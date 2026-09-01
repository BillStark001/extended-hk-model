"""Solve the paper's 10x10 logarithmic alpha/q grid with the mesoscopic model.

This command invokes the Go kinetic solver through long-lived JSONL batch
processes. Only online density-observable series cross the process boundary.

Run as ``python -m theory.mesoscopic.phase_scan``.
"""

from __future__ import annotations

import argparse
import csv
import shlex
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np

from ehk.common.plotting import setup_paper_params
from ehk.modeling.mesoscopic import (
    KineticParameters,
    ObservableResolution,
    ObservableSeries,
    ObservableThresholds,
    solve_observable_batch,
)
from theory.mesoscopic.cli_utils import pathway_label, write_run_metadata
from theory.paths import MESOSCOPIC_OUTPUT


RATE_EXPONENTS = np.linspace(-3.0, 0.0, 10)
RATES = np.power(10.0, RATE_EXPONENTS)


@dataclass
class SweepResult:
    q_index: int
    alpha_index: int
    time: np.ndarray
    polarization: np.ndarray
    homophily: np.ndarray
    subjective: np.ndarray
    pathway: float
    t_polarization: float
    t_homophily: float


def _sweep_result(
    q_index: int,
    alpha_index: int,
    values: ObservableSeries,
) -> SweepResult:
    return SweepResult(
        q_index=q_index,
        alpha_index=alpha_index,
        time=values.time,
        polarization=values.polarization,
        homophily=values.homophily,
        subjective=values.subjective,
        pathway=values.pathway,
        t_polarization=values.polarization_first_passage,
        t_homophily=values.homophily_first_passage,
    )


def _path_label(result: SweepResult) -> str:
    return pathway_label(result.t_polarization, result.t_homophily)


def _result_arrays(results: list[SweepResult]) -> dict[str, np.ndarray]:
    first = results[0]
    shape_series = (RATES.size, RATES.size, first.time.size)
    shape_grid = (RATES.size, RATES.size)
    polarization = np.empty(shape_series)
    homophily = np.empty(shape_series)
    subjective = np.empty(shape_series)
    pathway = np.empty(shape_grid)
    t_polarization = np.empty(shape_grid)
    t_homophily = np.empty(shape_grid)
    for result in results:
        index = result.q_index, result.alpha_index
        polarization[index] = result.polarization
        homophily[index] = result.homophily
        subjective[index] = result.subjective
        pathway[index] = result.pathway
        t_polarization[index] = result.t_polarization
        t_homophily[index] = result.t_homophily
    return {
        "time": first.time,
        "I_p": polarization,
        "I_h": homophily,
        "I_s": subjective,
        "I_w": pathway,
        "t_Ip_0.5": t_polarization,
        "t_Ih_0.5": t_homophily,
    }


def _save_data(
    results: list[SweepResult],
    arrays: dict[str, np.ndarray],
    output_dir: Path,
    params: KineticParameters,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "sweep_indices.npz",
        alpha=RATES,
        q=RATES,
        epsilon=np.asarray(params.epsilon),
        noise_diffusion=np.asarray(params.noise_diffusion),
        recsys=np.asarray(params.recsys),
        **arrays,
    )
    fieldnames = [
        "alpha",
        "q",
        "path",
        "I_w",
        "t_Ip_0.5",
        "t_Ih_0.5",
        "I_p_final",
        "I_h_final",
        "I_s_final",
    ]
    with (output_dir / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        for result in sorted(
            results, key=lambda item: (item.q_index, item.alpha_index)
        ):
            writer.writerow(
                {
                    "alpha": RATES[result.alpha_index],
                    "q": RATES[result.q_index],
                    "path": _path_label(result),
                    "I_w": result.pathway,
                    "t_Ip_0.5": result.t_polarization,
                    "t_Ih_0.5": result.t_homophily,
                    "I_p_final": result.polarization[-1],
                    "I_h_final": result.homophily[-1],
                    "I_s_final": result.subjective[-1],
                }
            )


def _plot_path_grid(
    arrays: dict[str, np.ndarray],
    figure_path: Path,
    params: KineticParameters,
) -> None:
    setup_paper_params()
    fig, axes = plt.subplots(
        RATES.size,
        RATES.size,
        figsize=(16, 16),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.91,
        bottom=0.07,
        top=0.945,
        wspace=0.08,
        hspace=0.08,
    )
    cmap = plt.get_cmap("coolwarm")
    norm = Normalize(0.0, 1.0)
    for display_row, q_index in enumerate(reversed(range(RATES.size))):
        for alpha_index in range(RATES.size):
            ax = axes[display_row, alpha_index]
            p = arrays["I_p"][q_index, alpha_index]
            h = arrays["I_h"][q_index, alpha_index]
            pathway = arrays["I_w"][q_index, alpha_index]
            ax.plot(p, h, color=cmap(norm(pathway)), lw=1.35)
            ax.scatter(p[0], h[0], color="black", s=7, zorder=3)
            ax.scatter(p[-1], h[-1], color="black", marker="x", s=10, zorder=3)
            ax.text(
                0.04,
                0.92,
                rf"$I_w={pathway:.2f}$",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7,
            )
            ax.set_xlim(-0.03, 1.03)
            ax.set_ylim(-0.03, 1.03)
            ax.grid(alpha=0.18, linewidth=0.4)
            ax.tick_params(labelsize=7, length=2)
            if display_row == 0:
                ax.set_title(
                    rf"$\alpha={RATES[alpha_index]:g}$", fontsize=9
                )
            if alpha_index == 0:
                ax.set_ylabel(
                    rf"$q={RATES[q_index]:g}$" + "\n" + r"$I_h$",
                    fontsize=9,
                )
            if display_row == RATES.size - 1:
                ax.set_xlabel(r"$I_p$", fontsize=9)

    scalar = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    scalar.set_array([])
    color_axis = fig.add_axes((0.925, 0.18, 0.013, 0.64))
    colorbar = fig.colorbar(scalar, cax=color_axis)
    colorbar.set_label(r"pathway index $I_w$")
    fig.suptitle(
        r"Mesoscopic $I_p$--$I_h$ trajectories "
        rf"({params.recsys}; $p=0$, $k_h=0$, "
        rf"$D_0={params.noise_diffusion:g}$)",
        fontsize=14,
    )
    for suffix in (".pdf", ".png"):
        fig.savefig(figure_path.with_suffix(suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _heatmap(
    fig,
    ax,
    data: np.ndarray,
    title: str,
    cmap: str,
    norm=None,
    vmin=None,
    vmax=None,
):
    image = ax.imshow(
        data,
        origin="lower",
        aspect="equal",
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xticks(np.arange(RATES.size), labels=[f"{v:.2g}" for v in RATES])
    ax.set_yticks(np.arange(RATES.size), labels=[f"{v:.2g}" for v in RATES])
    ax.tick_params(axis="x", labelrotation=90)
    ax.set_xlabel(r"influence $\alpha$")
    ax.set_ylabel(r"rewiring $q$")
    ax.set_title(title, loc="left")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def _plot_summary(
    arrays: dict[str, np.ndarray],
    figure_path: Path,
) -> None:
    setup_paper_params()
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(9, 8),
        constrained_layout=True,
    )
    t_p = arrays["t_Ip_0.5"]
    t_h = arrays["t_Ih_0.5"]
    precedence = np.divide(
        t_h - t_p,
        t_h + t_p,
        out=np.full_like(t_p, np.nan),
        where=np.isfinite(t_p) & np.isfinite(t_h) & ((t_h + t_p) > 0),
    )
    _heatmap(
        fig,
        axes[0, 0],
        arrays["I_w"],
        r"(a) pathway index $I_w$",
        "coolwarm",
        vmin=0,
        vmax=1,
    )
    _heatmap(
        fig,
        axes[0, 1],
        precedence,
        r"(b) first-passage order $(t_H-t_P)/(t_H+t_P)$",
        "RdBu_r",
        norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1),
    )
    _heatmap(
        fig,
        axes[1, 0],
        arrays["I_p"][:, :, -1],
        r"(c) final polarization $I_p$",
        "magma",
        vmin=0,
        vmax=1,
    )
    _heatmap(
        fig,
        axes[1, 1],
        arrays["I_h"][:, :, -1],
        r"(d) final structural homophily $I_h$",
        "viridis",
        vmin=0,
        vmax=1,
    )
    for suffix in (".pdf", ".png"):
        fig.savefig(figure_path.with_suffix(suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)


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
    parser.add_argument(
        "--recsys",
        default="random",
        choices=(
            "random",
            "opinion_random",
            "structure_random_l0",
            "structure_random_l1",
        ),
    )
    parser.add_argument("--recommendation-steepness", type=float, default=1.0)
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
        default=output_root / "sweep_noise_1e-5",
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
    base = KineticParameters(
        epsilon=args.epsilon,
        influence=0.0,
        rewiring=0.0,
        dynamics=args.dynamics,
        opinion_method=args.opinion_method,
        mean_degree=float(args.mean_degree),
        recsys_count=args.recsys_count,
        recsys=args.recsys,
        recommendation_steepness=args.recommendation_steepness,
        recommendation_random_ratio=args.random_ratio,
        opinion_tolerance=args.opinion_tolerance,
        noise_diffusion=args.noise,
        grid_size=args.grid_size,
        dt=args.dt,
        steps=args.steps,
        record_every=args.record_every,
        confidence_mode=args.confidence_mode,
    )
    indexed_cases = [
        (q_index, alpha_index)
        for q_index in range(RATES.size)
        for alpha_index in range(RATES.size)
    ]
    cases = [
        (
            f"q{q_index}-a{alpha_index}",
            replace(
                base,
                influence=float(RATES[alpha_index]),
                rewiring=float(RATES[q_index]),
            ),
        )
        for q_index, alpha_index in indexed_cases
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
    results = [
        _sweep_result(q_index, alpha_index, values)
        for (q_index, alpha_index), values in zip(indexed_cases, series, strict=True)
    ]
    for count, result in enumerate(results, start=1):
        print(
            f"[{count:03d}/{len(results)}] alpha={RATES[result.alpha_index]:g}, "
            f"q={RATES[result.q_index]:g}, I_w={result.pathway:.3f}"
        )

    arrays = _result_arrays(results)
    _save_data(results, arrays, args.output_dir, base)
    write_run_metadata(
        args.output_dir / "run_metadata.json",
        analysis="single-recommender mesoscopic scan",
        command=shlex.join(
            [sys.executable, "-m", "theory.mesoscopic.phase_scan", *sys.argv[1:]]
        ),
        parameters=asdict(base),
        configuration={
            "rates": RATES.tolist(),
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
        args.figure_dir.mkdir(parents=True, exist_ok=True)
        _plot_path_grid(
            arrays,
            args.figure_dir / f"f_kinetic_pathway_grid_{args.tag}",
            base,
        )
        _plot_summary(
            arrays,
            args.figure_dir / f"f_kinetic_sweep_summary_{args.tag}",
        )
    print(f"mesoscopic sweep data written to {args.output_dir}")


if __name__ == "__main__":
    main()
