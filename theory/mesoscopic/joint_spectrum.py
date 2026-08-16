"""Compute and plot the joint opinion--edge ``alpha/q`` linear spectrum.

The command writes the complete frozen-time eigenvalue spectrum at selected
points of the analytic rewiring base orbit and the complete finite-horizon
singular spectrum of the time-ordered tangent propagator.  It is a Section 3.6
analysis only: no committor, terminal community count, or microscopic outcome
frequency is read or produced.
"""

from __future__ import annotations

import argparse
import csv
import shlex
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from ehk.common.plotting import setup_paper_params
from theory.mesoscopic.cli_utils import write_run_metadata
from theory.mesoscopic.joint_spectrum_operator import (
    JointSpectrumParameters,
    affine_jacobian_bases,
    build_joint_grid,
    eigenspectrum,
    interpolate_matrix,
    recommendation_slots,
    time_ordered_spectrum,
)
from theory.paths import MESOSCOPIC_OUTPUT


DEFAULT_ALPHAS = (0.005, 0.05, 0.3)
DEFAULT_REWIRING = (0.0, 0.005, 0.05, 0.3)
DEFAULT_SNAPSHOTS = (1.0, 0.5, 0.1)
BASE_FRACTIONS = np.asarray(
    (0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0), dtype=float
)


def _float_tuple(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated numeric list")
    return values


def _name_tuple(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated name list")
    return values


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=rows[0].keys(), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _rows_matching(
    rows: Iterable[dict[str, object]],
    *,
    recommender: str,
    alpha: float,
    rewiring: float,
    snapshot: float | None = None,
) -> list[dict[str, object]]:
    selected = []
    for row in rows:
        if row["recommender"] != recommender:
            continue
        if not np.isclose(float(row["alpha"]), alpha):
            continue
        if not np.isclose(float(row["q"]), rewiring):
            continue
        if snapshot is not None and not np.isclose(
            float(row["base_discordant_fraction_ratio"]), snapshot
        ):
            continue
        selected.append(row)
    if selected and "mode" in selected[0]:
        return sorted(selected, key=lambda item: int(item["mode"]))
    return selected


def _plot_instantaneous(
    rows: list[dict[str, object]],
    *,
    recommenders: tuple[str, ...],
    snapshots: tuple[float, ...],
    rewiring_values: tuple[float, ...],
    alpha: float,
    epsilon: float,
    length: float,
    output: Path,
) -> None:
    setup_paper_params()
    figure, axes = plt.subplots(
        len(recommenders),
        len(snapshots),
        figsize=(10.4, 7.4),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    colors = plt.cm.viridis(np.linspace(0.08, 0.9, len(rewiring_values)))
    for row_index, recommender in enumerate(recommenders):
        for column_index, snapshot in enumerate(snapshots):
            axis = axes[row_index, column_index]
            axis.axhline(0, color="0.45", lw=0.7)
            for color, rewiring in zip(colors, rewiring_values):
                selected = _rows_matching(
                    rows,
                    recommender=recommender,
                    alpha=alpha,
                    rewiring=rewiring,
                    snapshot=snapshot,
                )
                modes = np.asarray([int(item["mode"]) for item in selected])
                z = 2 * np.pi * modes * epsilon / length
                rate = np.asarray(
                    [float(item["spectral_abscissa"]) for item in selected]
                )
                axis.plot(
                    z,
                    rate,
                    marker="o",
                    ms=3,
                    lw=1.0,
                    color=color,
                    label=rf"$q={rewiring:g}$",
                )
            if row_index == 0:
                axis.set_title(rf"frozen $d/d_0={snapshot:g}$")
            if column_index == 0:
                axis.set_ylabel(f"{recommender}\n" + r"$\max\Re\,\sigma(A_m)$")
            if row_index == len(recommenders) - 1:
                axis.set_xlabel(r"wave number $z=\kappa_m\epsilon$")
            axis.grid(alpha=0.18, linewidth=0.5)
    axes[0, -1].legend(frameon=False, fontsize=7, ncol=2)
    figure.suptitle(rf"Joint frozen-time spectrum, $\alpha={alpha:g}$")
    figure.savefig(output.with_suffix(".pdf"))
    figure.savefig(output.with_suffix(".png"), dpi=300)
    plt.close(figure)


def _plot_finite_time(
    rows: list[dict[str, object]],
    *,
    recommenders: tuple[str, ...],
    rewiring_values: tuple[float, ...],
    alpha: float,
    horizon: float,
    epsilon: float,
    length: float,
    output: Path,
) -> None:
    setup_paper_params()
    figure, axes = plt.subplots(
        1,
        len(recommenders),
        figsize=(10.4, 3.25),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    colors = plt.cm.viridis(np.linspace(0.08, 0.9, len(rewiring_values)))
    for column, recommender in enumerate(recommenders):
        axis = axes[0, column]
        axis.axhline(0, color="0.45", lw=0.7)
        for color, rewiring in zip(colors, rewiring_values):
            selected = _rows_matching(
                rows,
                recommender=recommender,
                alpha=alpha,
                rewiring=rewiring,
            )
            modes = np.asarray([int(item["mode"]) for item in selected])
            z = 2 * np.pi * modes * epsilon / length
            joint = np.asarray(
                [float(item["leading_singular_rate"]) for item in selected]
            )
            rho_seed = np.asarray(
                [float(item["rho_seed_rate"]) for item in selected]
            )
            axis.plot(
                z,
                joint,
                marker="o",
                ms=3,
                lw=1.0,
                color=color,
                label=rf"$q={rewiring:g}$ joint",
            )
            axis.plot(z, rho_seed, ls="--", lw=0.9, color=color)
        axis.set_title(recommender)
        axis.set_xlabel(r"wave number $z=\kappa_m\epsilon$")
        axis.grid(alpha=0.18, linewidth=0.5)
    axes[0, 0].set_ylabel(r"finite-time rate $T^{-1}\log G_m$")
    axes[0, -1].legend(frameon=False, fontsize=6.5, ncol=2)
    figure.suptitle(
        rf"Time-ordered joint spectrum, $\alpha={alpha:g}$, $T={horizon:g}$; "
        r"dashed: $\rho$-only seed",
    )
    figure.savefig(output.with_suffix(".pdf"))
    figure.savefig(output.with_suffix(".png"), dpi=300)
    plt.close(figure)


def _plot_alpha_q(
    rows: list[dict[str, object]],
    *,
    recommenders: tuple[str, ...],
    alphas: tuple[float, ...],
    rewiring_values: tuple[float, ...],
    horizon: float,
    output: Path,
) -> None:
    setup_paper_params()
    figure, axes = plt.subplots(
        2,
        len(recommenders),
        figsize=(10.2, 6.2),
        constrained_layout=True,
        squeeze=False,
    )
    rate_arrays = []
    advantage_arrays = []
    cutoff_arrays = []
    for recommender in recommenders:
        rate = np.empty((len(rewiring_values), len(alphas)))
        advantage = np.empty_like(rate)
        cutoff = np.empty_like(rate, dtype=bool)
        for q_index, rewiring in enumerate(rewiring_values):
            for alpha_index, alpha in enumerate(alphas):
                match = _rows_matching(
                    rows,
                    recommender=recommender,
                    alpha=alpha,
                    rewiring=rewiring,
                )[0]
                rate[q_index, alpha_index] = float(match["fastest_joint_rate"])
                advantage[q_index, alpha_index] = float(
                    match["edge_advantage_at_fastest_joint_mode"]
                )
                cutoff[q_index, alpha_index] = bool(
                    int(match["fastest_joint_mode_at_cutoff"])
                )
        rate_arrays.append(rate)
        advantage_arrays.append(advantage)
        cutoff_arrays.append(cutoff)
    rate_min = min(float(values.min()) for values in rate_arrays)
    rate_max = max(float(values.max()) for values in rate_arrays)
    advantage_max = max(float(values.max()) for values in advantage_arrays)
    for column, recommender in enumerate(recommenders):
        rate_image = axes[0, column].imshow(
            rate_arrays[column],
            origin="lower",
            aspect="auto",
            cmap="coolwarm",
            vmin=rate_min,
            vmax=rate_max,
        )
        advantage_image = axes[1, column].imshow(
            advantage_arrays[column],
            origin="lower",
            aspect="auto",
            cmap="magma",
            vmin=0,
            vmax=advantage_max,
        )
        axes[0, column].set_title(recommender)
        for q_index, alpha_index in zip(*np.nonzero(cutoff_arrays[column])):
            axes[0, column].add_patch(
                Rectangle(
                    (alpha_index - 0.5, q_index - 0.5),
                    1,
                    1,
                    fill=False,
                    hatch="///",
                    edgecolor="0.15",
                    linewidth=0.0,
                )
            )
        for row in range(2):
            axes[row, column].set_xticks(
                range(len(alphas)), [f"{value:g}" for value in alphas]
            )
            axes[row, column].set_yticks(
                range(len(rewiring_values)),
                [f"{value:g}" for value in rewiring_values],
            )
            axes[row, column].set_xlabel(r"influence $\alpha$")
        axes[0, column].set_ylabel(r"rewiring $q$")
        axes[1, column].set_ylabel(r"rewiring $q$")
    figure.colorbar(
        rate_image,
        ax=axes[0, :],
        shrink=0.82,
        label="max joint rate",
    )
    figure.colorbar(
        advantage_image,
        ax=axes[1, :],
        shrink=0.82,
        label="joint minus rho-seed rate",
    )
    figure.suptitle(
        rf"Joint $\alpha/q$ spectrum over fixed horizon $T={horizon:g}$; "
        "hatched: fastest mode at cutoff"
    )
    figure.savefig(output.with_suffix(".pdf"))
    figure.savefig(output.with_suffix(".png"), dpi=300)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recommenders",
        type=_name_tuple,
        default=("random", "opinion", "opinionm9"),
    )
    parser.add_argument(
        "--alphas",
        type=_float_tuple,
        default=DEFAULT_ALPHAS,
    )
    parser.add_argument(
        "--rewiring-values",
        type=_float_tuple,
        default=DEFAULT_REWIRING,
    )
    parser.add_argument(
        "--snapshots",
        type=_float_tuple,
        default=DEFAULT_SNAPSHOTS,
    )
    parser.add_argument("--length", type=float, default=2.0)
    parser.add_argument("--epsilon", type=float, default=0.45)
    parser.add_argument("--mean-degree", type=int, default=15)
    parser.add_argument("--recsys-count", type=int, default=10)
    parser.add_argument("--random-mix", type=float, default=0.1)
    parser.add_argument("--opinion-bandwidth", type=float, default=0.1)
    parser.add_argument("--diffusion", type=float, default=1e-5)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--difference-step", type=float, default=1e-6)
    parser.add_argument("--modes", type=int, default=20)
    parser.add_argument("--horizon", type=float, default=100.0)
    parser.add_argument("--time-steps", type=int, default=40)
    parser.add_argument("--plot-alpha", type=float, default=0.05)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MESOSCOPIC_OUTPUT.resolve() / "joint_spectrum",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.modes < 1:
        raise ValueError("modes must be positive")
    if args.modes > args.grid_size // 3:
        raise ValueError(
            "modes must not exceed grid_size // 3; the joint spectrum needs "
            "at least three relative-edge grid points per wavelength"
        )
    if any(not 0 <= value <= 1 for value in args.alphas):
        raise ValueError("all alpha values must lie in [0, 1]")
    if any(not 0 <= value <= 1 for value in args.rewiring_values):
        raise ValueError("all q values must lie in [0, 1]")
    if any(not 0 < value <= 1 for value in args.snapshots):
        raise ValueError("snapshot ratios must lie in (0, 1]")
    if not any(np.isclose(args.plot_alpha, value) for value in args.alphas):
        raise ValueError("plot-alpha must be present in alphas")

    parameters = JointSpectrumParameters(
        length=args.length,
        epsilon=args.epsilon,
        mean_degree=args.mean_degree,
        recsys_count=args.recsys_count,
        random_mix=args.random_mix,
        opinion_bandwidth=args.opinion_bandwidth,
        diffusion=args.diffusion,
        grid_size=args.grid_size,
        difference_step=args.difference_step,
    )
    parameters.validate()
    instantaneous_rows: list[dict[str, object]] = []
    instantaneous_summary: list[dict[str, object]] = []
    finite_rows: list[dict[str, object]] = []
    finite_summary: list[dict[str, object]] = []
    time_validation_rows: list[dict[str, object]] = []

    for recommender_name in args.recommenders:
        slots = recommendation_slots(
            recommender_name,
            recsys_count=args.recsys_count,
            random_mix=args.random_mix,
        )
        grid = build_joint_grid(slots, parameters)
        for mode in range(1, args.modes + 1):
            base_matrices = []
            alpha_matrices = []
            q_matrices = []
            for fraction in BASE_FRACTIONS:
                base, alpha_matrix, q_matrix = affine_jacobian_bases(
                    slots=slots,
                    mode=mode,
                    discordant_fraction=grid.initial_discordant * fraction,
                    parameters=parameters,
                    grid=grid,
                )
                base_matrices.append(base)
                alpha_matrices.append(alpha_matrix)
                q_matrices.append(q_matrix)
            base_table = np.asarray(base_matrices)
            alpha_table = np.asarray(alpha_matrices)
            q_table = np.asarray(q_matrices)

            for alpha in args.alphas:
                for rewiring in args.rewiring_values:
                    for snapshot in args.snapshots:
                        matrix = interpolate_matrix(
                            BASE_FRACTIONS, base_table, snapshot
                        )
                        matrix += alpha * interpolate_matrix(
                            BASE_FRACTIONS, alpha_table, snapshot
                        )
                        matrix += rewiring * interpolate_matrix(
                            BASE_FRACTIONS, q_table, snapshot
                        )
                        spectrum = eigenspectrum(
                            matrix, parameters=parameters, grid=grid
                        )
                        for rank, index in enumerate(spectrum.order, start=1):
                            instantaneous_rows.append(
                                {
                                    "recommender": slots.name,
                                    "alpha": alpha,
                                    "q": rewiring,
                                    "base_discordant_fraction_ratio": snapshot,
                                    "mode": mode,
                                    "rank_by_real_part": rank,
                                    "eigenvalue_real": float(
                                        spectrum.eigenvalues[index].real
                                    ),
                                    "eigenvalue_imag": float(
                                        spectrum.eigenvalues[index].imag
                                    ),
                                    "rho_energy_fraction": float(
                                        spectrum.rho_energy_fraction[index]
                                    ),
                                }
                            )
                        dominant = int(spectrum.order[0])
                        candidates = np.flatnonzero(
                            spectrum.rho_energy_fraction
                            >= 0.5
                            * float(spectrum.rho_energy_fraction.max())
                        )
                        rho_branch = int(
                            candidates[
                                np.argmax(spectrum.eigenvalues[candidates].real)
                            ]
                        )
                        instantaneous_summary.append(
                            {
                                "recommender": slots.name,
                                "alpha": alpha,
                                "q": rewiring,
                                "base_discordant_fraction_ratio": snapshot,
                                "mode": mode,
                                "spectral_abscissa": float(
                                    spectrum.eigenvalues[dominant].real
                                ),
                                "dominant_imaginary_part": float(
                                    spectrum.eigenvalues[dominant].imag
                                ),
                                "dominant_rho_energy_fraction": float(
                                    spectrum.rho_energy_fraction[dominant]
                                ),
                                "rho_supported_branch_real": float(
                                    spectrum.eigenvalues[rho_branch].real
                                ),
                                "rho_supported_branch_fraction": float(
                                    spectrum.rho_energy_fraction[rho_branch]
                                ),
                            }
                        )

                    finite = time_ordered_spectrum(
                        fractions=BASE_FRACTIONS,
                        base_table=base_table,
                        alpha_table=alpha_table,
                        q_table=q_table,
                        alpha=alpha,
                        rewiring=rewiring,
                        horizon=args.horizon,
                        time_steps=args.time_steps,
                        parameters=parameters,
                        grid=grid,
                    )
                    for rank, (log_gain, rate) in enumerate(
                        zip(
                            finite.log_singular_gains,
                            finite.finite_time_rates,
                        ),
                        start=1,
                    ):
                        finite_rows.append(
                            {
                                "recommender": slots.name,
                                "alpha": alpha,
                                "q": rewiring,
                                "mode": mode,
                                "horizon": args.horizon,
                                "singular_rank": rank,
                                "log_singular_gain": float(log_gain),
                                "finite_time_rate": float(rate),
                            }
                        )
                    finite_summary.append(
                        {
                            "recommender": slots.name,
                            "alpha": alpha,
                            "q": rewiring,
                            "mode": mode,
                            "horizon": args.horizon,
                            "leading_log_singular_gain": float(
                                finite.log_singular_gains[0]
                            ),
                            "leading_singular_rate": float(
                                finite.finite_time_rates[0]
                            ),
                            "rho_seed_log_gain": finite.rho_seed_log_gain,
                            "rho_seed_rate": finite.rho_seed_rate,
                            "edge_advantage_rate": float(
                                finite.finite_time_rates[0]
                                - finite.rho_seed_rate
                            ),
                            "dominant_output_rho_fraction": (
                                finite.dominant_output_rho_fraction
                            ),
                            "final_discordant_fraction_ratio": (
                                finite.final_discordant_fraction_ratio
                            ),
                        }
                    )
                    if (
                        np.isclose(alpha, args.plot_alpha)
                        and np.isclose(rewiring, max(args.rewiring_values))
                        and mode in {2, 4, 8, 11, 17}
                    ):
                        for validation_steps in sorted(
                            {20, args.time_steps, 80}
                        ):
                            validation = time_ordered_spectrum(
                                fractions=BASE_FRACTIONS,
                                base_table=base_table,
                                alpha_table=alpha_table,
                                q_table=q_table,
                                alpha=alpha,
                                rewiring=rewiring,
                                horizon=args.horizon,
                                time_steps=validation_steps,
                                parameters=parameters,
                                grid=grid,
                            )
                            time_validation_rows.append(
                                {
                                    "recommender": slots.name,
                                    "alpha": alpha,
                                    "q": rewiring,
                                    "mode": mode,
                                    "grid_size": args.grid_size,
                                    "horizon": args.horizon,
                                    "time_steps": validation_steps,
                                    "leading_singular_rate": float(
                                        validation.finite_time_rates[0]
                                    ),
                                    "rho_seed_rate": validation.rho_seed_rate,
                                    "edge_advantage_rate": float(
                                        validation.finite_time_rates[0]
                                        - validation.rho_seed_rate
                                    ),
                                }
                            )
            print(f"built {slots.name} mode {mode}/{args.modes}")

    case_rows: list[dict[str, object]] = []
    resolved_names = tuple(
        recommendation_slots(
            name,
            recsys_count=args.recsys_count,
            random_mix=args.random_mix,
        ).name
        for name in args.recommenders
    )
    for recommender in resolved_names:
        for alpha in args.alphas:
            for rewiring in args.rewiring_values:
                selected = _rows_matching(
                    finite_summary,
                    recommender=recommender,
                    alpha=alpha,
                    rewiring=rewiring,
                )
                fastest = max(
                    selected, key=lambda item: float(item["leading_singular_rate"])
                )
                edge_advantaged = max(
                    selected, key=lambda item: float(item["edge_advantage_rate"])
                )
                case_rows.append(
                    {
                        "recommender": recommender,
                        "alpha": alpha,
                        "q": rewiring,
                        "horizon": args.horizon,
                        "final_discordant_fraction_ratio": float(
                            fastest["final_discordant_fraction_ratio"]
                        ),
                        "fastest_joint_mode": int(fastest["mode"]),
                        "fastest_joint_mode_at_cutoff": int(
                            int(fastest["mode"]) == args.modes
                        ),
                        "fastest_joint_rate": float(
                            fastest["leading_singular_rate"]
                        ),
                        "rho_seed_rate_at_fastest_joint_mode": float(
                            fastest["rho_seed_rate"]
                        ),
                        "edge_advantage_at_fastest_joint_mode": float(
                            fastest["edge_advantage_rate"]
                        ),
                        "most_edge_advantaged_mode": int(
                            edge_advantaged["mode"]
                        ),
                        "most_edge_advantaged_mode_at_cutoff": int(
                            int(edge_advantaged["mode"]) == args.modes
                        ),
                        "max_edge_advantage_rate": float(
                            edge_advantaged["edge_advantage_rate"]
                        ),
                        "rho_seed_rate_at_most_edge_advantaged_mode": float(
                            edge_advantaged["rho_seed_rate"]
                        ),
                    }
                )

    _write_rows(
        args.output_dir / "joint_instantaneous_eigenvalues.csv",
        instantaneous_rows,
    )
    _write_rows(
        args.output_dir / "joint_instantaneous_summary.csv",
        instantaneous_summary,
    )
    _write_rows(
        args.output_dir / "joint_finite_time_singular_values.csv",
        finite_rows,
    )
    _write_rows(
        args.output_dir / "joint_finite_time_summary.csv",
        finite_summary,
    )
    if time_validation_rows:
        _write_rows(
            args.output_dir / "joint_time_step_validation.csv",
            time_validation_rows,
        )
    _write_rows(args.output_dir / "joint_alpha_q_summary.csv", case_rows)

    if not args.skip_plots:
        _plot_instantaneous(
            instantaneous_summary,
            recommenders=resolved_names,
            snapshots=args.snapshots,
            rewiring_values=args.rewiring_values,
            alpha=args.plot_alpha,
            epsilon=args.epsilon,
            length=args.length,
            output=args.output_dir / "f_joint_instantaneous_spectrum",
        )
        _plot_finite_time(
            finite_summary,
            recommenders=resolved_names,
            rewiring_values=args.rewiring_values,
            alpha=args.plot_alpha,
            horizon=args.horizon,
            epsilon=args.epsilon,
            length=args.length,
            output=args.output_dir / "f_joint_finite_time_spectrum",
        )
        _plot_alpha_q(
            case_rows,
            recommenders=resolved_names,
            alphas=args.alphas,
            rewiring_values=args.rewiring_values,
            horizon=args.horizon,
            output=args.output_dir / "f_joint_alpha_q_spectrum",
        )

    parameters_metadata = {
        key: value
        for key, value in vars(args).items()
        if key not in {"output_dir", "skip_plots"}
    }
    parameters_metadata["base_fraction_grid"] = BASE_FRACTIONS.tolist()
    write_run_metadata(
        args.output_dir / "run_metadata.json",
        analysis="joint alpha/q periodic pair spectrum",
        command=shlex.join(
            [sys.executable, "-m", "theory.mesoscopic.joint_spectrum", *sys.argv[1:]]
        ),
        parameters=parameters_metadata,
        configuration={
            "output_dir": str(args.output_dir.resolve()),
            "probability_or_terminal_outcomes_used": False,
            "finite_horizon_is_fixed_by_cli_not_committor": True,
        },
    )
    print(f"joint spectrum written to {args.output_dir}")


if __name__ == "__main__":
    main()
