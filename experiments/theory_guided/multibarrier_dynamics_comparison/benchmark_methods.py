"""Benchmark the four kinetic operators at selected physical points."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from ehk.modeling.mesoscopic import KineticParameters, solve

COMBINATIONS = (
    ("hk", "measure"),
    ("hk", "fokker_planck"),
    ("deffuant", "measure"),
    ("deffuant", "fokker_planck"),
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _stratified_points(
    points: list[dict[str, str]], max_points: int
) -> list[dict[str, str]]:
    if max_points == 0 or max_points >= len(points):
        return points
    if max_points < 1:
        raise ValueError("max_points must be non-negative")
    by_epsilon: dict[float, list[dict[str, str]]] = defaultdict(list)
    for point in points:
        by_epsilon[float(point["epsilon"])].append(point)
    selected = []
    level = 0
    while len(selected) < max_points:
        added = False
        for epsilon in sorted(by_epsilon):
            group = by_epsilon[epsilon]
            if level < len(group):
                selected.append(group[level])
                added = True
                if len(selected) == max_points:
                    break
        if not added:
            break
        level += 1
    return selected


def _timed_solve(params: KineticParameters) -> float:
    started = time.perf_counter()
    solve(params, record_steps=(0, params.steps))
    return time.perf_counter() - started


def benchmark(
    scan_dir: Path,
    *,
    horizons: tuple[int, ...],
    repeats: int,
    max_points: int,
    jobs_for_estimate: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Time short runs, fit per-step costs, and extrapolate the scan protocol."""

    if len(horizons) < 2 or len(set(horizons)) != len(horizons):
        raise ValueError("at least two unique benchmark horizons are required")
    if min(horizons) < 1 or repeats < 1 or jobs_for_estimate < 1:
        raise ValueError("horizons, repeats, and jobs_for_estimate must be positive")
    protocol = json.loads((scan_dir / "protocol.json").read_text(encoding="utf-8"))
    points = _stratified_points(
        _read_csv(scan_dir / "selected_comparison_points.csv"), max_points
    )
    base = KineticParameters(**protocol["parameters"])
    raw_rows: list[dict[str, object]] = []
    total_runs = len(points) * len(COMBINATIONS) * len(horizons) * repeats
    completed = 0
    for point_index, point in enumerate(points):
        for dynamics, method in COMBINATIONS:
            for horizon in sorted(horizons):
                params = replace(
                    base,
                    epsilon=float(point["epsilon"]),
                    grid_size=int(point["grid_size"]),
                    dynamics=dynamics,
                    opinion_method=method,
                    recsys=point["recsys"],
                    recommendation_steepness=float(point["steepness"]),
                    influence=float(point["alpha"]),
                    rewiring=float(point["q"]),
                    steps=horizon,
                    record_every=horizon,
                )
                for repeat in range(repeats):
                    elapsed = _timed_solve(params)
                    completed += 1
                    row: dict[str, object] = {
                        "point_index": point_index,
                        "selection_reason": point["selection_reason"],
                        "epsilon": float(point["epsilon"]),
                        "grid_size": int(point["grid_size"]),
                        "configuration": point["configuration"],
                        "alpha_index": int(point["alpha_index"]),
                        "q_index": int(point["q_index"]),
                        "alpha": float(point["alpha"]),
                        "q": float(point["q"]),
                        "dynamics": dynamics,
                        "opinion_method": method,
                        "steps": horizon,
                        "repeat": repeat,
                        "elapsed_seconds": elapsed,
                    }
                    raw_rows.append(row)
                    print(
                        f"[{completed:03d}/{total_runs}] B={params.grid_size} "
                        f"{dynamics}/{method} steps={horizon} {elapsed:.3f}s",
                        flush=True,
                    )

    fitted_rows = []
    grouped: dict[tuple[int, str, str, int], list[dict[str, object]]] = defaultdict(
        list
    )
    for row in raw_rows:
        key = (
            int(row["point_index"]),
            str(row["dynamics"]),
            str(row["opinion_method"]),
            int(row["grid_size"]),
        )
        grouped[key].append(row)
    target_steps = int(protocol["parameters"]["steps"])
    for (point_index, dynamics, method, grid_size), group in grouped.items():
        median_by_horizon = {
            horizon: statistics.median(
                float(row["elapsed_seconds"])
                for row in group
                if int(row["steps"]) == horizon
            )
            for horizon in sorted(horizons)
        }
        x = np.asarray(list(median_by_horizon), dtype=float)
        y = np.asarray(list(median_by_horizon.values()), dtype=float)
        slope, intercept = np.polyfit(x, y, deg=1)
        slope = max(float(slope), 0.0)
        intercept = max(float(intercept), 0.0)
        estimated_case_seconds = (
            median_by_horizon[target_steps]
            if target_steps in median_by_horizon
            else intercept + target_steps * slope
        )
        point = points[point_index]
        fitted_rows.append(
            {
                "point_index": point_index,
                "epsilon": float(point["epsilon"]),
                "grid_size": grid_size,
                "configuration": point["configuration"],
                "alpha_index": int(point["alpha_index"]),
                "q_index": int(point["q_index"]),
                "dynamics": dynamics,
                "opinion_method": method,
                "setup_seconds": intercept,
                "seconds_per_step": slope,
                "estimated_case_seconds": estimated_case_seconds,
                "target_horizon_measured": target_steps in median_by_horizon,
            }
        )

    group_estimates = []
    epsilon_grids = {
        (float(item["epsilon"]), int(item["grid_size"]))
        for item in protocol["epsilon_grids"]
    }
    cases_per_combination = len(protocol["scenarios"]) * len(protocol["cells"])
    total_cpu_seconds = 0.0
    for epsilon, grid_size in sorted(epsilon_grids):
        for dynamics, method in COMBINATIONS:
            matching = [
                row
                for row in fitted_rows
                if float(row["epsilon"]) == epsilon
                and row["dynamics"] == dynamics
                and row["opinion_method"] == method
            ]
            if not matching:
                matching = [
                    row
                    for row in fitted_rows
                    if int(row["grid_size"]) == grid_size
                    and row["dynamics"] == dynamics
                    and row["opinion_method"] == method
                ]
            if not matching:
                matching = [
                    row
                    for row in fitted_rows
                    if row["dynamics"] == dynamics and row["opinion_method"] == method
                ]
            case_seconds = statistics.median(
                float(row["estimated_case_seconds"]) for row in matching
            )
            cpu_seconds = cases_per_combination * case_seconds
            total_cpu_seconds += cpu_seconds
            group_estimates.append(
                {
                    "epsilon": epsilon,
                    "grid_size": grid_size,
                    "dynamics": dynamics,
                    "opinion_method": method,
                    "cases": cases_per_combination,
                    "median_estimated_case_seconds": case_seconds,
                    "estimated_cpu_hours": cpu_seconds / 3600.0,
                }
            )
    ratios = {}
    for grid_size in sorted({int(row["grid_size"]) for row in group_estimates}):
        for dynamics in ("hk", "deffuant"):
            matching = [
                row
                for row in group_estimates
                if int(row["grid_size"]) == grid_size and row["dynamics"] == dynamics
            ]
            by_method = {
                method: statistics.median(
                    float(row["median_estimated_case_seconds"])
                    for row in matching
                    if row["opinion_method"] == method
                )
                for method in ("measure", "fokker_planck")
            }
            if set(by_method) == {"measure", "fokker_planck"}:
                ratios[f"B{grid_size}_{dynamics}_measure_over_fp"] = (
                    by_method["measure"] / by_method["fokker_planck"]
                    if by_method["fokker_planck"] > 0
                    else float("nan")
                )
    summary: dict[str, object] = {
        "target_steps": target_steps,
        "benchmark_horizons": sorted(horizons),
        "repeats": repeats,
        "benchmarked_points": len(points),
        "protocol_case_count": int(protocol["case_count"]),
        "estimated_total_cpu_hours": total_cpu_seconds / 3600.0,
        "idealized_wall_hours_at_requested_jobs": total_cpu_seconds
        / (3600.0 * jobs_for_estimate),
        "jobs_for_estimate": jobs_for_estimate,
        "measure_over_fp_ratios": ratios,
        "caveat": (
            "uses a measured target horizon when supplied, otherwise linear "
            "extrapolation; BLAS contention and physical-point variation can "
            "change realized wall time"
        ),
    }
    _atomic_csv(scan_dir / "benchmark_raw.csv", raw_rows)
    _atomic_csv(scan_dir / "benchmark_point_fits.csv", fitted_rows)
    _atomic_csv(scan_dir / "benchmark_group_estimates.csv", group_estimates)
    _atomic_json(scan_dir / "benchmark_summary.json", summary)
    return raw_rows, group_estimates, summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scan_dir", type=Path)
    parser.add_argument("--horizons", nargs="+", type=int, default=(200, 800, 4000))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--max-points",
        type=int,
        default=6,
        help="stratified cap; zero benchmarks every selected point",
    )
    parser.add_argument("--jobs-for-estimate", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    scan_dir = args.scan_dir.expanduser().resolve()
    _, _, summary = benchmark(
        scan_dir,
        horizons=tuple(args.horizons),
        repeats=args.repeats,
        max_points=args.max_points,
        jobs_for_estimate=args.jobs_for_estimate,
    )
    print(
        "estimated total CPU time: "
        f"{float(summary['estimated_total_cpu_hours']):.2f} h; "
        "idealized wall time: "
        f"{float(summary['idealized_wall_hours_at_requested_jobs']):.2f} h"
    )


if __name__ == "__main__":
    main()
