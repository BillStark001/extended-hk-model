"""Select and plot informative four-way dynamics/method comparison points."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ehk.common.plotting import setup_paper_params
from experiments.theory_guided.multibarrier_dynamics_comparison.analyze import (
    analyze_output,
)
from experiments.theory_guided.multibarrier_dynamics_comparison.protocol import (
    EpsilonGrid,
)


def _scale(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    scale = float(np.quantile(finite, 0.9)) if finite.size else 0.0
    return values / scale if scale > 1e-15 else np.zeros_like(values)


def _candidate_scores(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    method_iw = np.asarray([float(row["method_gap_I_w_max"]) for row in rows])
    method_barrier = np.asarray([float(row["method_gap_barrier_max"]) for row in rows])
    dynamics_iw = np.asarray([float(row["dynamics_gap_I_w_max"]) for row in rows])
    dynamics_barrier = np.asarray(
        [float(row["dynamics_gap_barrier_max"]) for row in rows]
    )
    method_score = _scale(method_iw) + 0.5 * _scale(method_barrier)
    dynamics_score = _scale(dynamics_iw) + 0.5 * _scale(dynamics_barrier)
    multibarrier_score = np.asarray(
        [
            max(int(row["robust_well_count_max"]) - 2, 0)
            + int(row["dominant_pair_switch_count_max"])
            + 0.5 * max(int(row["overshoot_class_count"]) - 1, 0)
            for row in rows
        ],
        dtype=float,
    )
    result = []
    for index, row in enumerate(rows):
        result.append(
            {
                **row,
                "method_selection_score": float(method_score[index]),
                "dynamics_selection_score": float(dynamics_score[index]),
                "multibarrier_selection_score": float(multibarrier_score[index]),
                "agreement_selection_score": float(
                    method_score[index] + dynamics_score[index]
                ),
            }
        )
    return result


def select_points(
    rows: list[dict[str, object]],
    *,
    points_per_epsilon: int = 4,
) -> list[dict[str, object]]:
    """Choose disagreement extremes, a multibarrier case, and a control."""

    if not 1 <= points_per_epsilon <= 4:
        raise ValueError("points_per_epsilon must lie in [1, 4]")
    selected: list[dict[str, object]] = []
    for epsilon in sorted({float(row["epsilon"]) for row in rows}):
        candidates = _candidate_scores(
            [row for row in rows if float(row["epsilon"]) == epsilon]
        )
        objectives = (
            ("method_disagreement", "method_selection_score", True),
            ("dynamics_disagreement", "dynamics_selection_score", True),
            ("multibarrier", "multibarrier_selection_score", True),
            ("agreement_control", "agreement_selection_score", False),
        )
        used: set[tuple[str, str, str]] = set()
        count = min(points_per_epsilon, len(candidates))
        for label, field, descending in objectives[:count]:
            ordered = sorted(
                candidates,
                key=lambda row: float(row[field]),
                reverse=descending,
            )
            choice = next(
                row
                for row in ordered
                if (
                    str(row["configuration"]),
                    str(row["q_index"]),
                    str(row["alpha_index"]),
                )
                not in used
            )
            key = (
                str(choice["configuration"]),
                str(choice["q_index"]),
                str(choice["alpha_index"]),
            )
            used.add(key)
            selected.append({"selection_reason": label, **choice})
    return selected


def _checkpoint_path(
    scan_dir: Path, point: dict[str, object], dynamics: str, method: str
) -> Path:
    epsilon_grid = EpsilonGrid(float(point["epsilon"]), int(point["grid_size"]))
    return (
        scan_dir
        / "cells"
        / epsilon_grid.key
        / dynamics
        / method
        / str(point["configuration"])
        / f"q{int(point['q_index']):02d}_a{int(point['alpha_index']):02d}.npz"
    )


def _atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _plot_selected(scan_dir: Path, selected: list[dict[str, object]]) -> None:
    setup_paper_params()
    columns = min(3, len(selected))
    rows = math.ceil(len(selected) / columns)
    combinations = (
        ("hk", "measure", "tab:blue", "-", "HK / measure"),
        ("hk", "fokker_planck", "tab:blue", "--", "HK / F-P"),
        ("deffuant", "measure", "tab:orange", "-", "Deffuant / measure"),
        ("deffuant", "fokker_planck", "tab:orange", "--", "Deffuant / F-P"),
    )
    density_figure, density_axes = plt.subplots(
        rows,
        columns,
        figsize=(3.6 * columns, 2.8 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    barrier_figure, barrier_axes = plt.subplots(
        rows,
        columns,
        figsize=(3.6 * columns, 2.8 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    for index, point in enumerate(selected):
        density_axis = density_axes.flat[index]
        barrier_axis = barrier_axes.flat[index]
        for dynamics, method, color, linestyle, label in combinations:
            path = _checkpoint_path(scan_dir, point, dynamics, method)
            with np.load(path, allow_pickle=False) as arrays:
                density_axis.plot(
                    arrays["x"],
                    arrays["rho"][-1],
                    color=color,
                    linestyle=linestyle,
                    label=label,
                )
                barrier_axis.plot(
                    arrays["time"],
                    arrays["multi_dominant_barrier_height"],
                    color=color,
                    linestyle=linestyle,
                    label=label,
                )
        title = (
            f"{point['selection_reason']}; $\\epsilon={float(point['epsilon']):g}$\n"
            f"{point['configuration']}, q[{point['q_index']}], "
            f"$\\alpha$[{point['alpha_index']}]"
        )
        density_axis.set_title(title, fontsize=8)
        density_axis.set_xlabel("opinion")
        density_axis.set_ylabel("final cell mass")
        barrier_axis.set_title(title, fontsize=8)
        barrier_axis.set_xlabel("time")
        barrier_axis.set_ylabel("dominant barrier")
        density_axis.grid(alpha=0.2)
        barrier_axis.grid(alpha=0.2)
    for axis in density_axes.flat[len(selected) :]:
        axis.set_visible(False)
    for axis in barrier_axes.flat[len(selected) :]:
        axis.set_visible(False)
    density_axes.flat[0].legend(fontsize=7)
    barrier_axes.flat[0].legend(fontsize=7)
    for name, figure in (
        ("f_selected_final_density", density_figure),
        ("f_selected_barrier_evolution", barrier_figure),
    ):
        for suffix in ("pdf", "png"):
            figure.savefig(scan_dir / f"{name}.{suffix}")
        plt.close(figure)


def run_selection(scan_dir: Path, points_per_epsilon: int) -> list[dict[str, object]]:
    """Analyze if needed, select physical points, and plot all four operators."""

    rows = analyze_output(scan_dir)
    selected = select_points(rows, points_per_epsilon=points_per_epsilon)
    _atomic_csv(scan_dir / "selected_comparison_points.csv", selected)
    _plot_selected(scan_dir, selected)
    return selected


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scan_dir", type=Path)
    parser.add_argument("--points-per-epsilon", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    scan_dir = args.scan_dir.expanduser().resolve()
    selected = run_selection(scan_dir, args.points_per_epsilon)
    print(f"selected {len(selected)} four-way comparison points: {scan_dir}")


if __name__ == "__main__":
    main()
