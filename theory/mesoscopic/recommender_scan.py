"""Compare five recommendation systems with the mesoscopic density solver.

The calculation is intentionally a no-repost/no-history closure.  ``structure``
and ``structurem9`` therefore use the solver's outgoing-common-neighbor pair
closure, not the exact directed in/out common-neighbor ranker in the
microscopic runtime.
"""

from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from ehk.common.plotting import setup_paper_params
from ehk.modeling.mesoscopic import KineticParameters
from theory.mesoscopic.phase_scan import (
    RATES,
    SweepResult,
    _path_label,
    _plot_path_grid,
    _plot_summary,
    _result_arrays,
    _save_data,
    _solve_case,
)
from theory.paths import MESOSCOPIC_OUTPUT


RECOMMENDERS = (
    "random",
    "opinion",
    "opinionm9",
    "structure",
    "structurem9",
)


def _solve_recommender_case(
    recsys: str,
    q_index: int,
    alpha_index: int,
    base: KineticParameters,
) -> tuple[str, SweepResult]:
    return recsys, _solve_case(
        q_index,
        alpha_index,
        replace(base, recsys=recsys),
    )


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
    fig,
    axis,
    values: np.ndarray,
    title: str,
    cmap: str,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    norm=None,
):
    image = axis.imshow(
        values,
        origin="lower",
        aspect="equal",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
    )
    axis.set_xticks(np.arange(RATES.size), labels=[f"{value:g}" for value in RATES])
    axis.set_yticks(np.arange(RATES.size), labels=[f"{value:g}" for value in RATES])
    axis.tick_params(axis="x", labelrotation=90, labelsize=7)
    axis.tick_params(axis="y", labelsize=7)
    axis.set_xlabel(r"influence $\alpha$", fontsize=8)
    axis.set_ylabel(r"rewiring $q$", fontsize=8)
    axis.set_title(title, fontsize=9)
    return image


def _plot_comparison(
    arrays_by_recsys: dict[str, dict[str, np.ndarray]],
    figure_path: Path,
    base: KineticParameters,
) -> None:
    """Plot shared-scale outcomes, with systems aligned in columns."""

    setup_paper_params()
    fig, axes = plt.subplots(
        3,
        len(RECOMMENDERS),
        figsize=(15, 8.5),
        constrained_layout=True,
    )
    figures = []
    for column, recsys in enumerate(RECOMMENDERS):
        arrays = arrays_by_recsys[recsys]
        figures.append(
            _comparison_heatmap(
                fig,
                axes[0, column],
                arrays["I_w"],
                recsys,
                "coolwarm",
                vmin=0,
                vmax=1,
            )
        )
        _comparison_heatmap(
            fig,
            axes[1, column],
            _precedence(arrays),
            "",
            "RdBu_r",
            norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1),
        )
        _comparison_heatmap(
            fig,
            axes[2, column],
            arrays["I_p"][:, :, -1],
            "",
            "magma",
            vmin=0,
            vmax=1,
        )
    axes[0, 0].set_ylabel(r"rewiring $q$\npathway $I_w$", fontsize=8)
    axes[1, 0].set_ylabel(r"rewiring $q$\nfirst-passage order", fontsize=8)
    axes[2, 0].set_ylabel(r"rewiring $q$\nfinal $I_p$", fontsize=8)
    fig.colorbar(figures[0], ax=axes[0, :], shrink=0.85, label=r"$I_w$")
    order_scalar = plt.cm.ScalarMappable(
        cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    )
    fig.colorbar(
        order_scalar,
        ax=axes[1, :],
        shrink=0.85,
        label=r"$(t_H-t_P)/(t_H+t_P)$",
    )
    polarization_scalar = plt.cm.ScalarMappable(cmap="magma")
    polarization_scalar.set_clim(0, 1)
    fig.colorbar(
        polarization_scalar,
        ax=axes[2, :],
        shrink=0.85,
        label=r"final $I_p$",
    )
    fig.suptitle(
        r"Mesoscopic recommendation comparison "
        rf"($p=0$, $k_h=0$, $D_0={base.noise_diffusion:g}$)",
        fontsize=14,
    )
    for suffix in (".pdf", ".png"):
        fig.savefig(figure_path.with_suffix(suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_combined_summary(
    results_by_recsys: dict[str, list[SweepResult]],
    output_path: Path,
) -> None:
    fieldnames = (
        "recsys", "alpha", "q", "path", "I_w", "t_Ip_0.5", "t_Ih_0.5",
        "I_p_final", "I_h_final", "I_s_final",
    )
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for recsys in RECOMMENDERS:
            for result in sorted(
                results_by_recsys[recsys],
                key=lambda item: (item.q_index, item.alpha_index),
            ):
                writer.writerow({
                    "recsys": recsys,
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
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--record-every", type=int, default=20)
    parser.add_argument("--grid-size", type=int, default=81)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.45)
    parser.add_argument("--noise", type=float, default=1e-5)
    parser.add_argument("--recsys-count", type=int, default=10)
    parser.add_argument("--random-mix", type=float, default=0.1)
    parser.add_argument("--jobs", type=int, default=min(os.cpu_count() or 1, 4))
    output_root = MESOSCOPIC_OUTPUT.resolve()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=output_root / "recsys_noise_1e-5",
    )
    parser.add_argument("--figure-dir", type=Path, default=output_root / "figures")
    parser.add_argument("--tag", default="noise_1e-5")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = KineticParameters(
        epsilon=args.epsilon,
        recsys_count=args.recsys_count,
        random_mix=args.random_mix,
        noise_diffusion=args.noise,
        grid_size=args.grid_size,
        dt=args.dt,
        steps=args.steps,
        record_every=args.record_every,
    )
    cases = [
        (recsys, q_index, alpha_index, base)
        for recsys in RECOMMENDERS
        for q_index in range(RATES.size)
        for alpha_index in range(RATES.size)
    ]
    results_by_recsys: dict[str, list[SweepResult]] = {
        recsys: [] for recsys in RECOMMENDERS
    }
    if args.jobs == 1:
        iterator = map(lambda case: _solve_recommender_case(*case), cases)
        for count, (recsys, result) in enumerate(iterator, start=1):
            results_by_recsys[recsys].append(result)
            print(f"[{count:03d}/320] {recsys} alpha={RATES[result.alpha_index]:g}, "
                  f"q={RATES[result.q_index]:g}, I_w={result.pathway:.3f}")
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = [
                executor.submit(_solve_recommender_case, *case) for case in cases
            ]
            for count, future in enumerate(as_completed(futures), start=1):
                recsys, result = future.result()
                results_by_recsys[recsys].append(result)
                print(f"[{count:03d}/320] {recsys} alpha={RATES[result.alpha_index]:g}, "
                      f"q={RATES[result.q_index]:g}, I_w={result.pathway:.3f}")

    arrays_by_recsys: dict[str, dict[str, np.ndarray]] = {}
    for recsys in RECOMMENDERS:
        arrays = _result_arrays(results_by_recsys[recsys])
        arrays_by_recsys[recsys] = arrays
        params = replace(base, recsys=recsys)
        output_dir = args.output_dir / recsys
        _save_data(results_by_recsys[recsys], arrays, output_dir, params)
        _plot_path_grid(
            arrays,
            args.figure_dir / f"f_kinetic_pathway_grid_{recsys}_{args.tag}",
            params,
        )
        _plot_summary(
            arrays,
            args.figure_dir / f"f_kinetic_sweep_summary_{recsys}_{args.tag}",
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_combined_summary(
        results_by_recsys, args.output_dir / "recsys_summary.csv"
    )
    np.savez_compressed(
        args.output_dir / "recsys_indices.npz",
        alpha=RATES,
        q=RATES,
        recsys=np.asarray(RECOMMENDERS),
        **{
            f"{recsys}_{name}": values
            for recsys, arrays in arrays_by_recsys.items()
            for name, values in arrays.items()
        },
    )
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    _plot_comparison(
        arrays_by_recsys,
        args.figure_dir / f"f_kinetic_recsys_comparison_{args.tag}",
        base,
    )
    print(f"five-system mesoscopic sweep data written to {args.output_dir}")


if __name__ == "__main__":
    main()
