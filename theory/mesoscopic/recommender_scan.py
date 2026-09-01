"""Compare five weighted-random recommendation configurations with the PDE.

The calculation is intentionally a no-repost/no-history closure.
``structure_random_l0`` uses the solver's outgoing-common-neighbor L0 pair
proxy, not the exact integer common-neighbor score in the microscopic runtime.
"""

from __future__ import annotations

import argparse
import csv
import shlex
import sys
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np

from ehk.common.plotting import setup_paper_params
from ehk.modeling.mesoscopic import (
    KineticParameters,
    ObservableResolution,
    ObservableThresholds,
    solve_observable_batch,
)
from theory.mesoscopic.phase_scan import (
    RATES,
    SweepResult,
    _path_label,
    _plot_path_grid,
    _plot_summary,
    _result_arrays,
    _save_data,
    _sweep_result,
)
from theory.mesoscopic.cli_utils import write_run_metadata
from theory.paths import MESOSCOPIC_OUTPUT


CONFIGURATIONS = (
    ("random", "random", 1.0),
    ("opinion_random_zeta1", "opinion_random", 1.0),
    ("opinion_random_zeta4", "opinion_random", 4.0),
    ("structure_random_l0_zeta1", "structure_random_l0", 1.0),
    ("structure_random_l0_zeta4", "structure_random_l0", 4.0),
)
RECOMMENDERS = tuple(configuration[0] for configuration in CONFIGURATIONS)
DISPLAY_NAMES = {
    "random": "Random",
    "opinion_random_zeta1": r"OpinionRandom ($\zeta=1$)",
    "opinion_random_zeta4": r"OpinionRandom ($\zeta=4$)",
    "structure_random_l0_zeta1": r"L0-StructureRandom ($\zeta=1$)",
    "structure_random_l0_zeta4": r"L0-StructureRandom ($\zeta=4$)",
}


def _precedence(arrays: dict[str, np.ndarray]) -> np.ndarray:
    t_p = arrays["t_Ip_0.5"]
    t_h = arrays["t_Ih_0.5"]
    return np.divide(
        t_h - t_p,
        t_h + t_p,
        out=np.full_like(t_p, np.nan),
        where=np.isfinite(t_p) & np.isfinite(t_h) & ((t_h + t_p) > 0),
    )


def _comparison_heatmap(
    axis,
    values: np.ndarray,
    title: str,
    cmap: str,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    norm=None,
    show_x: bool = True,
    show_y: bool = True,
):
    color_map = plt.get_cmap(cmap).copy()
    color_map.set_bad("#e6e6e6")
    image = axis.imshow(
        np.ma.masked_invalid(values),
        origin="lower",
        aspect="equal",
        cmap=color_map,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
    )
    rate_labels = [f"{value:.2g}" for value in RATES]
    axis.set_xticks(
        np.arange(RATES.size), labels=rate_labels if show_x else []
    )
    axis.set_yticks(
        np.arange(RATES.size), labels=rate_labels if show_y else []
    )
    axis.tick_params(
        axis="x", labelrotation=90, labelsize=7, length=2
    )
    axis.tick_params(axis="y", labelsize=7, length=2)
    if show_x:
        axis.set_xlabel(r"influence $\alpha$", fontsize=8)
    axis.set_title(title, fontsize=9)
    return image


def _plot_comparison(
    arrays_by_recsys: dict[str, dict[str, np.ndarray]],
    figure_path: Path,
) -> None:
    """Plot shared-scale outcomes, with systems aligned in columns."""

    setup_paper_params()
    fig, axes = plt.subplots(
        3,
        len(RECOMMENDERS),
        figsize=(17.2, 7.2),
        constrained_layout=False,
    )
    fig.subplots_adjust(
        left=0.065,
        right=0.92,
        bottom=0.13,
        top=0.94,
        wspace=0.18,
        hspace=0.16,
    )
    figures = []
    for column, recsys in enumerate(RECOMMENDERS):
        arrays = arrays_by_recsys[recsys]
        figures.append(
            _comparison_heatmap(
                axes[0, column],
                arrays["I_w"],
                DISPLAY_NAMES[recsys],
                "coolwarm",
                vmin=0,
                vmax=1,
                show_x=False,
                show_y=column == 0,
            )
        )
        _comparison_heatmap(
            axes[1, column],
            _precedence(arrays),
            "",
            "RdBu_r",
            norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1),
            show_x=False,
            show_y=column == 0,
        )
        _comparison_heatmap(
            axes[2, column],
            arrays["I_p"][:, :, -1],
            "",
            "magma",
            vmin=0,
            vmax=1,
            show_x=True,
            show_y=column == 0,
        )
    axes[0, 0].set_ylabel(r"pathway $I_w$" + "\n" + r"rewiring $q$", fontsize=8)
    axes[1, 0].set_ylabel("first-passage order\n" + r"rewiring $q$", fontsize=8)
    axes[2, 0].set_ylabel(r"final $I_p$" + "\n" + r"rewiring $q$", fontsize=8)
    # The PDE solver itself remains well defined at alpha=1 or q=1, but its
    # derivation from the synchronous microscopic update is no longer a
    # controlled small-step approximation.  Hatch that column and row so the
    # distinction survives grayscale printing.
    for axis in axes.ravel():
        axis.add_patch(
            Rectangle(
                (RATES.size - 1.5, -0.5),
                1,
                RATES.size,
                fill=False,
                hatch="////",
                edgecolor="0.25",
                linewidth=0.45,
                alpha=0.45,
            )
        )
        axis.add_patch(
            Rectangle(
                (-0.5, RATES.size - 1.5),
                RATES.size,
                1,
                fill=False,
                hatch="\\\\\\\\",
                edgecolor="0.25",
                linewidth=0.45,
                alpha=0.45,
            )
        )
    fig.canvas.draw()

    def color_axis(row: int):
        bounds = axes[row, -1].get_position()
        return fig.add_axes((0.94, bounds.y0, 0.009, bounds.height))

    fig.colorbar(
        figures[0], cax=color_axis(0), label=r"$I_w$"
    )
    order_scalar = plt.cm.ScalarMappable(
        cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    )
    fig.colorbar(
        order_scalar,
        cax=color_axis(1),
        label=r"$(t_H-t_P)/(t_H+t_P)$",
    )
    polarization_scalar = plt.cm.ScalarMappable(cmap="magma")
    polarization_scalar.set_clim(0, 1)
    fig.colorbar(
        polarization_scalar,
        cax=color_axis(2),
        label=r"final $I_p$",
    )
    for suffix in (".pdf", ".png"):
        fig.savefig(figure_path.with_suffix(suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_combined_summary(
    results_by_recsys: dict[str, list[SweepResult]],
    output_path: Path,
) -> None:
    fieldnames = (
        "configuration", "recsys", "steepness", "alpha", "q", "path",
        "I_w", "t_Ip_0.5", "t_Ih_0.5", "I_p_final", "I_h_final",
        "I_s_final",
    )
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        for configuration, recsys, steepness in CONFIGURATIONS:
            for result in sorted(
                results_by_recsys[configuration],
                key=lambda item: (item.q_index, item.alpha_index),
            ):
                writer.writerow({
                    "configuration": configuration,
                    "recsys": recsys,
                    "steepness": steepness,
                    "alpha": RATES[result.alpha_index],
                    "q": RATES[result.q_index],
                    "path": _path_label(result),
                    "I_w": result.pathway,
                    "t_Ip_0.5": result.t_polarization,
                    "t_Ih_0.5": result.t_homophily,
                    "I_p_final": result.polarization[-1],
                    "I_h_final": result.homophily[-1],
                    "I_s_final": result.subjective[-1],
                })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kinetic-binary", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--record-every", type=int, default=20)
    parser.add_argument("--grid-size", type=int, default=81)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.45)
    parser.add_argument("--noise", type=float, default=1e-5)
    parser.add_argument("--dynamics", choices=("hk", "deffuant"), default="hk")
    parser.add_argument(
        "--opinion-method",
        choices=("measure", "fokker_planck"),
        default="fokker_planck",
    )
    parser.add_argument("--mean-degree", type=int, default=15)
    parser.add_argument("--recsys-count", type=int, default=10)
    parser.add_argument("--opinion-tolerance", type=float, default=0.4)
    parser.add_argument("--random-ratio", type=float, default=0.0)
    parser.add_argument(
        "--confidence-mode",
        choices=("center", "cell_average"),
        default="cell_average",
    )
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--population", type=int, default=500)
    parser.add_argument("--opinion-min", type=float, default=-1.0)
    parser.add_argument("--opinion-max", type=float, default=1.0)
    parser.add_argument("--opinion-quadrature-points", type=int, default=5)
    parser.add_argument("--confidence-quadrature-points", type=int, default=5)
    parser.add_argument("--score-max", type=int, default=45)
    parser.add_argument("--distance-grid-size", type=int, default=256)
    parser.add_argument("--minimum-bandwidth", type=float, default=0.01)
    parser.add_argument("--objective-effective-samples", type=int, default=10_000)
    parser.add_argument("--polarization-threshold", type=float, default=0.5)
    parser.add_argument("--homophily-threshold", type=float, default=0.5)
    output_root = MESOSCOPIC_OUTPUT.resolve()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=output_root / "recsys_noise_1e-5",
    )
    parser.add_argument("--figure-dir", type=Path, default=output_root / "figures")
    parser.add_argument("--tag", default="noise_1e-5")
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="write numerical outputs without regenerating figures",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_plots:
        args.figure_dir.mkdir(parents=True, exist_ok=True)
    base = KineticParameters(
        epsilon=args.epsilon,
        influence=0.0,
        rewiring=0.0,
        dynamics=args.dynamics,
        opinion_method=args.opinion_method,
        mean_degree=float(args.mean_degree),
        recsys_count=args.recsys_count,
        opinion_tolerance=args.opinion_tolerance,
        recommendation_steepness=1.0,
        recommendation_random_ratio=args.random_ratio,
        noise_diffusion=args.noise,
        grid_size=args.grid_size,
        dt=args.dt,
        steps=args.steps,
        record_every=args.record_every,
        confidence_mode=args.confidence_mode,
    )
    indexed_cases = [
        (configuration, recsys, steepness, q_index, alpha_index)
        for configuration, recsys, steepness in CONFIGURATIONS
        for q_index in range(RATES.size)
        for alpha_index in range(RATES.size)
    ]
    results_by_recsys: dict[str, list[SweepResult]] = {
        recsys: [] for recsys in RECOMMENDERS
    }
    cases = [
        (
            f"{configuration}-q{q_index}-a{alpha_index}",
            replace(
                base,
                recsys=recsys,
                recommendation_steepness=steepness,
                influence=float(RATES[alpha_index]),
                rewiring=float(RATES[q_index]),
            ),
        )
        for configuration, recsys, steepness, q_index, alpha_index in indexed_cases
    ]
    resolution = ObservableResolution(
        population=args.population,
        opinion_min=args.opinion_min,
        opinion_max=args.opinion_max,
        opinion_quadrature_points=args.opinion_quadrature_points,
        confidence_quadrature_points=args.confidence_quadrature_points,
        score_max=args.score_max,
        distance_grid_size=args.distance_grid_size,
        minimum_bandwidth=args.minimum_bandwidth,
        objective_effective_samples=args.objective_effective_samples,
    )
    thresholds = ObservableThresholds(
        polarization=args.polarization_threshold,
        homophily=args.homophily_threshold,
    )
    series = solve_observable_batch(
        args.kinetic_binary, cases, resolution, thresholds, args.jobs, None
    )
    for count, (case, values) in enumerate(
        zip(indexed_cases, series, strict=True), start=1
    ):
        configuration, _, _, q_index, alpha_index = case
        result = _sweep_result(q_index, alpha_index, values)
        results_by_recsys[configuration].append(result)
        print(
            f"[{count:03d}/{len(cases)}] {configuration} "
            f"alpha={RATES[result.alpha_index]:g}, q={RATES[result.q_index]:g}, "
            f"I_w={result.pathway:.3f}"
        )

    arrays_by_recsys: dict[str, dict[str, np.ndarray]] = {}
    for configuration, recsys, steepness in CONFIGURATIONS:
        arrays = _result_arrays(results_by_recsys[configuration])
        arrays_by_recsys[configuration] = arrays
        params = replace(
            base,
            recsys=recsys,
            recommendation_steepness=steepness,
        )
        output_dir = args.output_dir / configuration
        _save_data(results_by_recsys[configuration], arrays, output_dir, params)
        if not args.skip_plots:
            _plot_path_grid(
                arrays,
                args.figure_dir / f"f_kinetic_pathway_grid_{configuration}_{args.tag}",
                params,
            )
            _plot_summary(
                arrays,
                args.figure_dir / f"f_kinetic_sweep_summary_{configuration}_{args.tag}",
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_combined_summary(
        results_by_recsys, args.output_dir / "recsys_summary.csv"
    )
    np.savez_compressed(
        args.output_dir / "recsys_indices.npz",
        alpha=RATES,
        q=RATES,
        configuration=np.asarray(RECOMMENDERS),
        **{
            f"{configuration}_{name}": values
            for configuration, arrays in arrays_by_recsys.items()
            for name, values in arrays.items()
        },
    )
    write_run_metadata(
        args.output_dir / "run_metadata.json",
        analysis="five-configuration weighted-random recommender mesoscopic scan",
        command=shlex.join(
            [sys.executable, "-m", "theory.mesoscopic.recommender_scan", *sys.argv[1:]]
        ),
        parameters=asdict(base),
        configuration={
            "rates": RATES.tolist(),
            "configurations": [
                {
                    "key": configuration,
                    "recsys": recsys,
                    "steepness": steepness,
                }
                for configuration, recsys, steepness in CONFIGURATIONS
            ],
            "jobs": args.jobs,
            "kinetic_binary": str(args.kinetic_binary.resolve()),
            "observable_resolution": asdict(resolution),
            "observable_thresholds": asdict(thresholds),
            "skip_plots": args.skip_plots,
            "output_dir": str(args.output_dir.resolve()),
            "figure_dir": str(args.figure_dir.resolve()),
            "tag": args.tag,
        },
    )
    if not args.skip_plots:
        _plot_comparison(
            arrays_by_recsys,
            args.figure_dir / f"f_kinetic_recsys_comparison_{args.tag}",
        )
    print(f"five-configuration mesoscopic sweep data written to {args.output_dir}")


if __name__ == "__main__":
    main()
