"""Quantify double-well formation and barrier maturation in the PDE closure."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy

from ehk.common.plotting import setup_paper_params
from ehk.metrics import (
    IndexSeries,
    LandscapeSeries,
    calculate_index_series,
    potential_from_force,
    quantify_landscape_series,
)
from ehk.metrics.landscape_diagnostics import first_persistent_time
from ehk.modeling.mesoscopic import KineticParameters, KineticTrajectory, solve
from theory.mesoscopic.l1_comparison import CASES, CLOSURES, ComparisonCase

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class QuantificationResult:
    case: ComparisonCase
    level: str
    closure: str
    trajectory: KineticTrajectory
    indices: IndexSeries
    potential: np.ndarray
    landscape: LandscapeSeries


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def record_schedule(
    steps: int,
    *,
    early_until: int,
    early_every: int,
    late_every: int,
) -> tuple[int, ...]:
    """Dense early records followed by a coarser long-time schedule."""

    if steps < 1 or early_every < 1 or late_every < 1 or early_until < 0:
        raise ValueError("record schedule arguments are invalid")
    split = min(steps, early_until)
    selected = set(range(0, split + 1, early_every))
    selected.update(range(split, steps + 1, late_every))
    selected.update((0, split, steps))
    return tuple(sorted(selected))


def _crossing_time(time: np.ndarray, values: np.ndarray, threshold: float) -> float:
    return first_persistent_time(time, values >= threshold, persistence=3)


def _threshold_time(landscape: LandscapeSeries, threshold: float) -> float:
    matches = np.flatnonzero(
        np.isclose(landscape.barrier_thresholds, threshold, rtol=0.0, atol=1e-12)
    )
    return (
        float(landscape.barrier_crossing_times[matches[0]])
        if matches.size
        else float("nan")
    )


def run_quantification(
    base: KineticParameters,
    *,
    selected_cases: set[str],
    selected_levels: set[str],
    record_steps: tuple[int, ...],
    persistence: int,
) -> list[QuantificationResult]:
    results: list[QuantificationResult] = []
    for case in CASES:
        if case.key not in selected_cases:
            continue
        for level, recsys, closure in CLOSURES:
            if level not in selected_levels:
                continue
            params = replace(
                base,
                influence=case.influence,
                rewiring=case.rewiring,
                recsys=recsys,
                recommendation_steepness=1.0,
            )
            print(
                f"solving {case.key}/{level}: alpha={case.influence:g}, "
                f"q={case.rewiring:g}, B={params.grid_size}",
                flush=True,
            )
            trajectory = solve(params, record_steps=record_steps)
            indices = calculate_index_series(trajectory)
            potential = potential_from_force(
                trajectory.x, trajectory.velocity / case.influence
            )
            landscape = quantify_landscape_series(
                trajectory.x,
                trajectory.time,
                potential,
                barrier_thresholds=(0.01, 0.05),
                persistence=persistence,
            )
            results.append(
                QuantificationResult(
                    case,
                    level,
                    closure,
                    trajectory,
                    indices,
                    potential,
                    landscape,
                )
            )
    return results


def summary_row(result: QuantificationResult) -> dict[str, object]:
    landscape = result.landscape
    indices = result.indices
    final = -1
    t_p = _crossing_time(indices.time, indices.polarization, 0.5)
    t_h = _crossing_time(indices.time, indices.homophily, 0.5)
    return {
        "case": result.case.key,
        "level": result.level,
        "closure": result.closure,
        "influence": result.case.influence,
        "rewiring": result.case.rewiring,
        "q_over_alpha": result.case.rewiring / result.case.influence,
        "I_w": indices.pathway,
        "t_P_0.5": t_p,
        "t_H_0.5": t_h,
        "t_double_well": landscape.formation_time,
        "t_barrier_0.01": _threshold_time(landscape, 0.01),
        "t_barrier_0.05": _threshold_time(landscape, 0.05),
        "t_barrier_half_final": landscape.half_final_barrier_time,
        "barrier_final": landscape.barrier_height[final],
        "barrier_max": np.max(landscape.barrier_height),
        "left_depth_final": landscape.left_depth[final],
        "right_depth_final": landscape.right_depth[final],
        "left_position_final": landscape.left_position[final],
        "right_position_final": landscape.right_position[final],
        "well_separation_final": landscape.well_separation[final],
        "left_curvature_final": landscape.left_curvature[final],
        "right_curvature_final": landscape.right_curvature[final],
        "barrier_curvature_final": landscape.barrier_curvature[final],
        "t_double_well_minus_t_P": landscape.formation_time - t_p,
        "t_double_well_minus_t_H": landscape.formation_time - t_h,
        "t_barrier_0.05_minus_t_P": _threshold_time(landscape, 0.05) - t_p,
        "t_barrier_0.05_minus_t_H": _threshold_time(landscape, 0.05) - t_h,
    }


def _save_result(result: QuantificationResult, output_dir: Path) -> None:
    landscape = result.landscape
    indices = result.indices
    np.savez_compressed(
        output_dir / f"{result.case.key}_{result.level}.npz",
        parameters=json.dumps(asdict(result.trajectory.parameters), sort_keys=True),
        x=result.trajectory.x,
        time=result.trajectory.time,
        rho=result.trajectory.rho,
        velocity=result.trajectory.velocity,
        potential=result.potential,
        I_p=indices.polarization,
        I_h=indices.homophily,
        I_s=indices.subjective,
        I_w=np.asarray(indices.pathway),
        double_well=landscape.double_well,
        barrier_height=landscape.barrier_height,
        left_depth=landscape.left_depth,
        right_depth=landscape.right_depth,
        left_position=landscape.left_position,
        right_position=landscape.right_position,
        barrier_position=landscape.barrier_position,
        well_separation=landscape.well_separation,
        left_curvature=landscape.left_curvature,
        right_curvature=landscape.right_curvature,
        barrier_curvature=landscape.barrier_curvature,
        formation_time=np.asarray(landscape.formation_time),
        barrier_thresholds=landscape.barrier_thresholds,
        barrier_crossing_times=landscape.barrier_crossing_times,
        half_final_barrier_time=np.asarray(landscape.half_final_barrier_time),
    )


def _write_summary(
    results: list[QuantificationResult], output_dir: Path
) -> list[dict[str, object]]:
    rows = [summary_row(result) for result in results]
    with (output_dir / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _plot_barrier_maturation(
    results: list[QuantificationResult], output_dir: Path
) -> None:
    setup_paper_params()
    cases = [case for case in CASES if any(r.case.key == case.key for r in results)]
    figure, axes = plt.subplots(
        1,
        len(cases),
        figsize=(3.6 * len(cases), 3.2),
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    colors = {"l0": "tab:blue", "l1": "tab:orange"}
    for column, case in enumerate(cases):
        axis = axes[0, column]
        for result in results:
            if result.case.key != case.key:
                continue
            axis.plot(
                result.landscape.time,
                result.landscape.barrier_height,
                label=result.level.upper(),
                color=colors[result.level],
            )
            t_p = _crossing_time(
                result.indices.time, result.indices.polarization, 0.5
            )
            t_h = _crossing_time(result.indices.time, result.indices.homophily, 0.5)
            if np.isfinite(t_p):
                axis.axvline(t_p, color=colors[result.level], linestyle=":", alpha=0.5)
            if np.isfinite(t_h):
                axis.axvline(t_h, color=colors[result.level], linestyle="--", alpha=0.5)
        axis.axhline(0.05, color="black", linewidth=0.7, alpha=0.5)
        axis.set_title(case.title)
        axis.set_xlabel("time")
        axis.grid(alpha=0.2)
    axes[0, 0].set_ylabel(r"effective barrier $\Delta V$")
    axes[0, -1].legend(frameon=False)
    for suffix in ("pdf", "png"):
        figure.savefig(
            output_dir / f"barrier_maturation.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def _plot_timing_summary(
    rows: list[dict[str, object]], output_dir: Path
) -> None:
    setup_paper_params()
    short_case = {
        "no_rewiring": "no rw.",
        "balanced": "balanced",
        "influence_dominant": "influence",
        "rewiring_dominant": "rewiring",
    }
    labels = [
        f"{short_case[str(row['case'])]}\n{str(row['level']).upper()}" for row in rows
    ]
    positions = np.arange(len(rows))
    figure, axis = plt.subplots(figsize=(max(7.0, 1.05 * len(rows)), 3.7))
    for offset, (field, label, marker) in enumerate(
        (
            ("t_P_0.5", r"$t_P$", "o"),
            ("t_H_0.5", r"$t_H$", "s"),
            ("t_double_well", r"$t_{dw}$", "^"),
            ("t_barrier_0.05", r"$t_{\Delta V=0.05}$", "D"),
        )
    ):
        values = np.asarray([float(row[field]) for row in rows])
        axis.scatter(positions + 0.08 * (offset - 1.5), values, label=label, marker=marker)
    axis.set_xticks(positions, labels, fontsize=8)
    axis.set_ylabel("persistent first-passage time")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, ncol=4)
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(
            output_dir / f"landscape_timing_summary.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiments/theory_guided/potential_quantification"),
    )
    parser.add_argument("--grid-size", type=int, default=41)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--early-until", type=int, default=200)
    parser.add_argument("--early-every", type=int, default=1)
    parser.add_argument("--late-every", type=int, default=20)
    parser.add_argument("--persistence", type=int, default=3)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--noise", type=float, default=1e-5)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=[case.key for case in CASES],
        default=[case.key for case in CASES],
    )
    parser.add_argument(
        "--closures", nargs="+", choices=["l0", "l1"], default=["l0", "l1"]
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    schedule = record_schedule(
        args.steps,
        early_until=args.early_until,
        early_every=args.early_every,
        late_every=args.late_every,
    )
    base = KineticParameters(
        epsilon=0.45,
        mean_degree=15,
        recsys_count=10,
        recommendation_random_ratio=0.0,
        recommendation_steepness=1.0,
        grid_size=args.grid_size,
        steps=args.steps,
        record_every=args.late_every,
        dt=args.dt,
        noise_diffusion=args.noise,
    )
    results = run_quantification(
        base,
        selected_cases=set(args.cases),
        selected_levels=set(args.closures),
        record_steps=schedule,
        persistence=args.persistence,
    )
    if not results:
        raise RuntimeError("no quantification cells were selected")
    for result in results:
        _save_result(result, output_dir)
    rows = _write_summary(results, output_dir)
    _plot_barrier_maturation(results, output_dir)
    _plot_timing_summary(rows, output_dir)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": " ".join([sys.executable, *sys.argv]),
        "base_parameters": asdict(base),
        "record_steps": schedule,
        "persistence_records": args.persistence,
        "selected_cases": args.cases,
        "selected_closures": args.closures,
        "potential_scale": "velocity divided by influence",
        "barrier_definition": "minimum of left and right basin depths",
        "git_revision": _git_revision(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote potential-landscape quantification to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
