"""Compute the five-column linear-spectrum terminal proxy and comparison plot.

The upper row propagates finite-N initial sampling
through the joint opinion--edge tangent operator and converts independent
Fourier-mode amplitudes into a largest-mode-above-threshold probability.  The
identity map ``winning mode m -> K=m`` is a declared terminal proxy, not an
exact committor. The lower row is either an explicit placeholder or a supplied
configuration-matched microscopic summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.integrate import quad
from scipy.linalg import expm

from theory.mesoscopic.joint_spectrum_operator import (
    JointSpectrumParameters,
    affine_jacobian_bases,
    build_joint_grid,
    discordant_orbit,
    interpolate_matrix,
    recommendation_slots,
)
from experiments.theory_guided.terminal_probability.paths import OUTPUT_DIR
from experiments.theory_guided.terminal_probability.scenarios import (
    PAPER_COMPARISON_CASES,
    PAPER_RATE_GRID,
    RECOMMENDATION_SCENARIOS,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
BASE_FRACTIONS = np.asarray((0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0))
RATE_GRID = np.asarray(PAPER_RATE_GRID)
CASES = tuple(
    (case.key, case.alpha_index, case.q_index)
    for case in PAPER_COMPARISON_CASES
)
CONFIGURATIONS = tuple(
    (item.key, item.spectrum_rule, item.steepness, item.display)
    for item in RECOMMENDATION_SCENARIOS
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _initial_covariance(parameters: JointSpectrumParameters, grid) -> np.ndarray:
    probabilities = np.asarray(grid.random[0] * grid.dx, dtype=float)
    edge = (
        np.diag(probabilities) - np.outer(probabilities, probabilities)
    ) / (parameters.mean_degree * grid.dx**2)
    covariance = np.zeros(
        (parameters.grid_size, parameters.grid_size), dtype=complex
    )
    covariance[0, 0] = 1.0
    covariance[1:, 1:] = edge[:-1, :-1]
    return covariance


def _winner_probabilities(
    mean_square_amplitudes: np.ndarray,
    *,
    threshold: float,
) -> tuple[float, np.ndarray, float]:
    """Independent-exponential order statistics above one amplitude cutoff."""

    means = np.asarray(mean_square_amplitudes, dtype=float)
    if np.any(means <= 0) or threshold <= 0:
        raise ValueError("mode means and threshold must be positive")
    threshold_squared = threshold**2
    exceedance = np.exp(-threshold_squared / means)
    no_onset = float(np.prod(1 - exceedance))
    winner = np.zeros(means.size)
    for index, mean in enumerate(means):
        def integrand(unit_exponential: float) -> float:
            amplitude = threshold_squared + mean * unit_exponential
            cdf = -np.expm1(-amplitude / means)
            return math.exp(-unit_exponential) * float(
                np.prod(np.delete(cdf, index))
            )

        breakpoints = {1.0, 5.0, 15.0}
        for other_mean in np.delete(means, index):
            for ratio in (0.05, 0.2, 1.0, 5.0):
                location = (ratio * other_mean - threshold_squared) / mean
                if 1e-12 < location < 60:
                    breakpoints.add(float(location))
        winner[index] = exceedance[index] * quad(
            integrand,
            0.0,
            60.0,
            epsabs=1e-12,
            epsrel=2e-10,
            limit=500,
            points=tuple(sorted(breakpoints)),
        )[0]
    raw_mass = float(no_onset + winner.sum())
    if raw_mass <= 0 or abs(raw_mass - 1) > 2e-6:
        raise RuntimeError(f"mode-race probability mass is {raw_mass:.12g}")
    return no_onset / raw_mass, winner / raw_mass, raw_mass - 1


def _matrix_tables(*, slots, parameters, grid, mode: int) -> tuple[np.ndarray, ...]:
    bases: list[np.ndarray] = []
    alphas: list[np.ndarray] = []
    rewiring: list[np.ndarray] = []
    for fraction in BASE_FRACTIONS:
        base, alpha_matrix, q_matrix = affine_jacobian_bases(
            slots=slots,
            mode=mode,
            discordant_fraction=grid.initial_discordant * float(fraction),
            parameters=parameters,
            grid=grid,
        )
        bases.append(base)
        alphas.append(alpha_matrix)
        rewiring.append(q_matrix)
    return np.asarray(bases), np.asarray(alphas), np.asarray(rewiring)


def _amplified_variance(
    *,
    alpha: float,
    rewiring: float,
    horizon: float,
    time_steps: int,
    parameters: JointSpectrumParameters,
    grid,
    tables: tuple[np.ndarray, ...],
) -> tuple[float, float]:
    base_table, alpha_table, q_table = tables
    dt = horizon / time_steps
    midpoint_times = (np.arange(time_steps, dtype=float) + 0.5) * dt
    discordant = discordant_orbit(
        midpoint_times,
        rewiring=rewiring,
        parameters=parameters,
        grid=grid,
    )
    observation = np.zeros(parameters.grid_size, dtype=complex)
    observation[0] = 1.0
    for value in discordant[::-1]:
        fraction = max(float(value / grid.initial_discordant), BASE_FRACTIONS[0])
        matrix = interpolate_matrix(BASE_FRACTIONS, base_table, fraction)
        matrix += alpha * interpolate_matrix(BASE_FRACTIONS, alpha_table, fraction)
        matrix += rewiring * interpolate_matrix(BASE_FRACTIONS, q_table, fraction)
        observation = expm(matrix.conj().T * dt) @ observation
    covariance = _initial_covariance(parameters, grid)
    variance = float(np.vdot(observation, covariance @ observation).real)
    final_discordant = float(
        discordant_orbit(
            np.asarray([horizon]),
            rewiring=rewiring,
            parameters=parameters,
            grid=grid,
        )[0]
    )
    return max(variance, np.finfo(float).tiny), final_discordant


def compute(
    *,
    population: int = 500,
    horizon: float = 100.0,
    time_steps: int = 40,
    threshold: float = 0.2,
    modes: int = 6,
    grid_size: int = 32,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if population < 2 or horizon <= 0 or time_steps < 1:
        raise ValueError("population, horizon, and time_steps must be positive")
    if threshold <= 0 or modes < 4 or grid_size < 16 or grid_size % 2:
        raise ValueError(
            "threshold must be positive; modes must be >= 4; "
            "grid_size must be an even integer >= 16"
        )
    parameters = JointSpectrumParameters(
        epsilon=0.45,
        mean_degree=15,
        recsys_count=10,
        opinion_tolerance=0.4,
        diffusion=1e-5,
        grid_size=grid_size,
        difference_step=1e-6,
    )
    probability_rows: list[dict[str, object]] = []
    mode_rows: list[dict[str, object]] = []

    for key, rule, steepness, _display in CONFIGURATIONS:
        slots = recommendation_slots(
            rule,
            recsys_count=parameters.recsys_count,
            steepness=steepness,
        )
        grid = build_joint_grid(slots, parameters)
        tables_by_mode = {
            mode: _matrix_tables(
                slots=slots, parameters=parameters, grid=grid, mode=mode
            )
            for mode in range(1, modes + 1)
        }
        for case_name, alpha_index, q_index in CASES:
            alpha = float(RATE_GRID[alpha_index])
            rewiring = float(RATE_GRID[q_index])
            mean_squares: list[float] = []
            final_discordant = math.nan
            for mode in range(1, modes + 1):
                clt_variance, final_discordant = _amplified_variance(
                    alpha=alpha,
                    rewiring=rewiring,
                    horizon=horizon,
                    time_steps=time_steps,
                    parameters=parameters,
                    grid=grid,
                    tables=tables_by_mode[mode],
                )
                mean_square = clt_variance / population
                mean_squares.append(mean_square)
                mode_rows.append(
                    {
                        "configuration": key,
                        "recommender": slots.name,
                        "steepness": steepness,
                        "case": case_name,
                        "alpha": alpha,
                        "q": rewiring,
                        "mode": mode,
                        "mean_square_amplitude": mean_square,
                    }
                )
            no_onset, winner, mass_error = _winner_probabilities(
                np.asarray(mean_squares), threshold=threshold
            )
            probability_rows.append(
                {
                    "configuration": key,
                    "recommender": slots.name,
                    "steepness": steepness,
                    "case": case_name,
                    "alpha": alpha,
                    "q": rewiring,
                    "p_k1_identity": float(winner[0]),
                    "p_k2_identity": float(winner[1]),
                    "p_k3_identity": float(winner[2]),
                    "p_k4plus_identity": float(winner[3:].sum()),
                    "p_no_linear_onset": no_onset,
                    "raw_probability_mass_error": mass_error,
                    "final_discordant_fraction_ratio": (
                        final_discordant / grid.initial_discordant
                    ),
                }
            )
        print(f"computed {slots.name}", flush=True)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "joint linear-spectrum terminal-probability proxy",
        "parameters": {
            **asdict(parameters),
            "population": population,
            "horizon": horizon,
            "time_steps": time_steps,
            "threshold": threshold,
            "modes": modes,
            "base_fraction_grid": BASE_FRACTIONS.tolist(),
            "rate_grid": RATE_GRID.tolist(),
            "selected_cases": [
                {"name": name, "alpha_index": ai, "q_index": qi}
                for name, ai, qi in CASES
            ],
        },
        "interpretation": {
            "probability_object": "largest amplified linear opinion mode above threshold",
            "terminal_mapping": "identity proxy K approximately equals winning mode; modes 4+ pooled",
            "exact_terminal_committor": False,
            "process_noise_included": False,
            "initial_opinion_and_random_graph_sampling_included": True,
            "structure_kernel": "L0 expected-overlap pair closure, not exact integer common-neighbor score",
            "lower_figure_row": "placeholder; production microscopic sweep not run",
        },
        "data_usage": {
            "microscopic_simulations_run": False,
            "microscopic_outcomes_read": False,
            "microscopic_calibration": False,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "matplotlib": plt.matplotlib.__version__,
        },
    }
    return pd.DataFrame(probability_rows), pd.DataFrame(mode_rows), metadata


def _microscopic_fields(frame: pd.DataFrame) -> tuple[tuple[str, str, str], ...]:
    missing = "p_incomplete" if "p_incomplete" in frame else "p_censored"
    required = {"p_k1", "p_k2", "p_k3", "p_k4plus", missing}
    absent = sorted(required - set(frame.columns))
    if absent:
        raise ValueError(
            "microscopic summary is missing probability columns: "
            + ", ".join(absent)
        )
    return (
        ("p_k1", r"$K=1$", "#3973ac"),
        ("p_k2", r"$K=2$", "#e79f3c"),
        ("p_k3", r"$K=3$", "#c95b5b"),
        ("p_k4plus", r"$K\geq4$", "#8b6bb1"),
        (missing, "incomplete" if missing == "p_incomplete" else "censored", "#b8b8b8"),
    )


def plot(
    probabilities: pd.DataFrame,
    output: Path,
    microscopic: pd.DataFrame | None = None,
) -> None:
    fields = (
        ("p_k1_identity", r"$K=1$", "#3973ac"),
        ("p_k2_identity", r"$K=2$", "#e79f3c"),
        ("p_k3_identity", r"$K=3$", "#c95b5b"),
        ("p_k4plus_identity", r"$K\geq4$", "#8b6bb1"),
        ("p_no_linear_onset", "no linear onset", "#b8b8b8"),
    )
    case_labels = {case.key: case.display for case in PAPER_COMPARISON_CASES}
    figure, axes = plt.subplots(
        2,
        5,
        figsize=(17.2, 6.0),
        sharey="row",
        constrained_layout=True,
        gridspec_kw={"height_ratios": (1.05, 0.82)},
    )
    for column, (key, _rule, _steepness, display) in enumerate(CONFIGURATIONS):
        selected = probabilities[probabilities["configuration"] == key].set_index("case")
        axis = axes[0, column]
        bottom = np.zeros(len(CASES))
        for field, label, color in fields:
            values = np.asarray(
                [float(selected.loc[case_name, field]) for case_name, _, _ in CASES]
            )
            axis.bar(
                np.arange(len(CASES)), values, bottom=bottom,
                width=0.72, color=color, label=label,
            )
            bottom += values
        axis.set_title(display)
        axis.set_xticks(
            np.arange(len(CASES)),
            [case_labels[name] for name, _, _ in CASES],
            rotation=28,
            ha="right",
        )
        axis.set_ylim(0, 1)
        axis.grid(axis="y", linewidth=0.5, alpha=0.22)

        lower = axes[1, column]
        lower.set_xlim(-0.5, len(CASES) - 0.5)
        lower.set_ylim(0, 1)
        lower.set_xticks(
            np.arange(len(CASES)),
            [case_labels[name] for name, _, _ in CASES],
            rotation=28,
            ha="right",
        )
        if microscopic is None:
            lower.set_facecolor("#f7f7f7")
            lower.text(
                0.5,
                0.52,
                "PLACEHOLDER\nconfiguration-matched\nGo ensemble pending",
                ha="center",
                va="center",
                transform=lower.transAxes,
                color="#666666",
                fontsize=8.5,
            )
        else:
            selected_micro = microscopic[
                microscopic["configuration"] == key
            ].set_index("case")
            absent_cases = [
                case_name for case_name, _, _ in CASES
                if case_name not in selected_micro.index
            ]
            if absent_cases:
                raise ValueError(
                    f"microscopic summary lacks {key}: {', '.join(absent_cases)}"
                )
            lower_bottom = np.zeros(len(CASES))
            for field, _label, color in _microscopic_fields(microscopic):
                values = np.asarray([
                    float(selected_micro.loc[case_name, field])
                    for case_name, _, _ in CASES
                ])
                lower.bar(
                    np.arange(len(CASES)), values, bottom=lower_bottom,
                    width=0.72, color=color,
                )
                lower_bottom += values
        lower.grid(axis="y", linewidth=0.5, alpha=0.15)
    axes[0, 0].set_ylabel("linear-spectrum proxy", labelpad=8)
    axes[1, 0].set_ylabel("microscopic probability", labelpad=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=5,
    )
    for suffix in (".pdf", ".png"):
        figure.savefig(output.with_suffix(suffix), dpi=300, bbox_inches="tight")
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=OUTPUT_DIR.resolve() / "spectrum_proxy",
    )
    parser.add_argument(
        "--microscopic-summary", type=Path,
        help="optional CSV with configuration/case and p_k* columns",
    )
    parser.add_argument("--population", type=int, default=500)
    parser.add_argument("--horizon", type=float, default=100.0)
    parser.add_argument("--time-steps", type=int, default=40)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--modes", type=int, default=6)
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--no-plot", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    microscopic = (
        pd.read_csv(args.microscopic_summary)
        if args.microscopic_summary is not None
        else None
    )
    probabilities, modes, metadata = compute(
        population=args.population,
        horizon=args.horizon,
        time_steps=args.time_steps,
        threshold=args.threshold,
        modes=args.modes,
        grid_size=args.grid_size,
    )
    probabilities.to_csv(
        output_dir / "linear_spectrum_terminal_proxy.csv", index=False
    )
    modes.to_csv(output_dir / "linear_spectrum_mode_variances.csv", index=False)
    if not args.no_plot:
        plot(
            probabilities,
            output_dir / "f_terminal_probability_two_row",
            microscopic,
        )

    script = Path(__file__).resolve()
    operator = REPOSITORY_ROOT / "theory" / "mesoscopic" / "joint_spectrum_operator.py"
    scenario_source = script.with_name("scenarios.py")
    metadata["command"] = [sys.executable, "-m", __package__ + ".spectrum", *sys.argv[1:]]
    metadata["interpretation"]["lower_figure_row"] = (
        f"microscopic summary: {args.microscopic_summary.resolve()}"
        if args.microscopic_summary is not None
        else "placeholder; production microscopic sweep not run"
    )
    metadata["data_usage"] = {
        "microscopic_simulations_run": False,
        "microscopic_outcomes_read": microscopic is not None,
        "microscopic_calibration": False,
    }
    metadata["sources"] = {
        str(script.relative_to(REPOSITORY_ROOT)): _sha256(script),
        str(scenario_source.relative_to(REPOSITORY_ROOT)): _sha256(scenario_source),
        str(operator.relative_to(REPOSITORY_ROOT)): _sha256(operator),
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
