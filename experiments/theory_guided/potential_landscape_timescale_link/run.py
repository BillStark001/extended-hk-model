"""Run targeted B=81 potential landscapes selected by the time-scale scan."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from ehk.metrics import (
    calculate_index_series,
    potential_from_force,
    quantify_landscape_series,
)
from ehk.metrics.landscape_diagnostics import first_persistent_time
from ehk.modeling.mesoscopic import KineticParameters, solve
from experiments.theory_guided.macroscopic_timescale_ratio.scenarios import (
    SCENARIO_BY_KEY,
    SCENARIOS,
    record_schedule,
)
from theory.mesoscopic.cli_utils import write_run_metadata
from theory.paths import MESOSCOPIC_OUTPUT

DESIGNS = ("common_rates", "transition_center")


@dataclass(frozen=True)
class LandscapeCase:
    key: str
    design: str
    configuration: str
    recsys: str
    steepness: float
    display_name: str
    closure_note: str
    alpha: float
    q: float
    fitted_transition_ratio: float


def _transition_centers(path: Path) -> dict[str, float]:
    centers: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["predictor"] == "rate_ratio":
                centers[row["configuration"]] = float(row["transition"])
    missing = [scenario.key for scenario in SCENARIOS if scenario.key not in centers]
    if missing:
        raise ValueError(
            "rate-ratio transition centers missing for: " + ", ".join(missing)
        )
    return centers


def build_cases(
    transition_offsets: Path,
    *,
    designs: tuple[str, ...],
    configurations: tuple[str, ...],
    common_alpha: float,
    common_q: float,
    transition_alpha: float,
) -> tuple[LandscapeCase, ...]:
    """Build the two controlled designs from the fitted offset table."""

    centers = _transition_centers(transition_offsets)
    cases = []
    for design in designs:
        for configuration in configurations:
            scenario = SCENARIO_BY_KEY[configuration]
            if design == "common_rates":
                alpha = common_alpha
                q = common_q
            elif design == "transition_center":
                alpha = transition_alpha
                q = transition_alpha * centers[configuration]
            else:
                raise ValueError(f"unknown design: {design}")
            if not 0 < alpha <= 1 or not 0 <= q <= 1:
                raise ValueError(
                    f"{design}/{configuration} has invalid alpha={alpha}, q={q}"
                )
            cases.append(
                LandscapeCase(
                    key=f"{design}__{configuration}",
                    design=design,
                    configuration=configuration,
                    recsys=scenario.recsys,
                    steepness=scenario.steepness,
                    display_name=scenario.display_name,
                    closure_note=scenario.closure_note,
                    alpha=alpha,
                    q=q,
                    fitted_transition_ratio=centers[configuration],
                )
            )
    return tuple(cases)


def _persistent_crossing(
    time: np.ndarray,
    values: np.ndarray,
    threshold: float,
    persistence: int,
) -> float:
    return first_persistent_time(time, values >= threshold, persistence=persistence)


def _interpolate_at_time(
    time: np.ndarray,
    values: np.ndarray,
    target: float,
) -> float:
    if not np.isfinite(target):
        return float("nan")
    return float(np.interp(target, time, values))


def _value_at_progress(
    progress: np.ndarray,
    values: np.ndarray,
    threshold: float,
) -> tuple[float, float]:
    found = np.flatnonzero(progress >= threshold)
    if not found.size:
        return float("nan"), float("nan")
    upper = int(found[0])
    if upper == 0:
        return 0.0, float(values[0])
    lower = upper - 1
    before = float(progress[lower])
    after = float(progress[upper])
    fraction = (threshold - before) / (after - before) if after > before else 1.0
    fraction = float(np.clip(fraction, 0.0, 1.0))
    return fraction, float(values[lower] + fraction * (values[upper] - values[lower]))


def _threshold_time(
    thresholds: np.ndarray,
    crossing_times: np.ndarray,
    threshold: float,
) -> float:
    found = np.flatnonzero(np.isclose(thresholds, threshold, atol=1e-12, rtol=0))
    return float(crossing_times[found[0]]) if found.size else float("nan")


def _atomic_npz(path: Path, arrays: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def solve_case(
    case: LandscapeCase,
    base: KineticParameters,
    selected_steps: tuple[int, ...],
    persistence: int,
    protocol_digest: str,
    output_path: Path,
) -> dict[str, object]:
    """Solve, quantify, and checkpoint one targeted landscape case."""

    parameters = replace(
        base,
        influence=case.alpha,
        rewiring=case.q,
        recsys=case.recsys,
        recommendation_steepness=case.steepness,
    )
    trajectory = solve(parameters, record_steps=selected_steps)
    indices = calculate_index_series(trajectory)
    force = trajectory.velocity / case.alpha
    potential = potential_from_force(trajectory.x, force)
    landscape = quantify_landscape_series(
        trajectory.x,
        trajectory.time,
        potential,
        barrier_thresholds=(0.01, 0.05),
        persistence=persistence,
    )
    t_p = _persistent_crossing(indices.time, indices.polarization, 0.5, persistence)
    t_h = _persistent_crossing(indices.time, indices.homophily, 0.5, persistence)
    progress = indices.polarization + indices.homophily
    fraction, barrier_u0p1 = _value_at_progress(progress, landscape.barrier_height, 0.1)
    found = np.flatnonzero(progress >= 0.1)
    if not found.size:
        t_u0p1 = float("nan")
    else:
        upper = int(found[0])
        lower = max(upper - 1, 0)
        t_u0p1 = float(
            trajectory.time[lower]
            + fraction * (trajectory.time[upper] - trajectory.time[lower])
        )
    row: dict[str, object] = {
        **asdict(case),
        "rate_ratio": case.q / case.alpha,
        "I_w": indices.pathway,
        "t_P_0.5": t_p,
        "t_H_0.5": t_h,
        "t_u_0.1": t_u0p1,
        "t_double_well": landscape.formation_time,
        "t_barrier_0.01": _threshold_time(
            landscape.barrier_thresholds,
            landscape.barrier_crossing_times,
            0.01,
        ),
        "t_barrier_0.05": _threshold_time(
            landscape.barrier_thresholds,
            landscape.barrier_crossing_times,
            0.05,
        ),
        "t_barrier_half_final": landscape.half_final_barrier_time,
        "barrier_initial": float(landscape.barrier_height[0]),
        "barrier_at_u_0.1": barrier_u0p1,
        "barrier_at_t_P": _interpolate_at_time(
            trajectory.time, landscape.barrier_height, t_p
        ),
        "barrier_at_t_H": _interpolate_at_time(
            trajectory.time, landscape.barrier_height, t_h
        ),
        "barrier_final": float(landscape.barrier_height[-1]),
        "barrier_max": float(np.max(landscape.barrier_height)),
        "well_separation_final": float(landscape.well_separation[-1]),
        "left_curvature_final": float(landscape.left_curvature[-1]),
        "right_curvature_final": float(landscape.right_curvature[-1]),
        "barrier_curvature_final": float(landscape.barrier_curvature[-1]),
        "t_double_well_minus_t_P": landscape.formation_time - t_p,
        "t_double_well_minus_t_H": landscape.formation_time - t_h,
    }
    _atomic_npz(
        output_path,
        {
            "protocol_digest": np.asarray(protocol_digest),
            "case_json": np.asarray(json.dumps(asdict(case), sort_keys=True)),
            "parameters_json": np.asarray(
                json.dumps(asdict(parameters), sort_keys=True)
            ),
            "summary_json": np.asarray(json.dumps(row, sort_keys=True, allow_nan=True)),
            "x": trajectory.x,
            "time": trajectory.time,
            "rho": trajectory.rho,
            "velocity": trajectory.velocity,
            "force": force,
            "potential": potential,
            "I_p": indices.polarization,
            "I_h": indices.homophily,
            "I_s": indices.subjective,
            "I_w": np.asarray(indices.pathway),
            "double_well": landscape.double_well,
            "barrier_height": landscape.barrier_height,
            "left_depth": landscape.left_depth,
            "right_depth": landscape.right_depth,
            "left_position": landscape.left_position,
            "right_position": landscape.right_position,
            "barrier_position": landscape.barrier_position,
            "well_separation": landscape.well_separation,
            "left_curvature": landscape.left_curvature,
            "right_curvature": landscape.right_curvature,
            "barrier_curvature": landscape.barrier_curvature,
            "formation_time": np.asarray(landscape.formation_time),
            "barrier_thresholds": landscape.barrier_thresholds,
            "barrier_crossing_times": landscape.barrier_crossing_times,
            "half_final_barrier_time": np.asarray(landscape.half_final_barrier_time),
        },
    )
    return row


def _load_checkpoint(path: Path, protocol_digest: str) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as arrays:
        if str(arrays["protocol_digest"].item()) != protocol_digest:
            raise ValueError(f"incompatible checkpoint protocol: {path}")
        return json.loads(str(arrays["summary_json"].item()))


def _protocol_payload(
    base: KineticParameters,
    cases: tuple[LandscapeCase, ...],
    selected_steps: tuple[int, ...],
    persistence: int,
    transition_offsets: Path,
) -> dict[str, Any]:
    return {
        "parameters": asdict(base),
        "cases": [asdict(case) for case in cases],
        "record_steps": list(selected_steps),
        "persistence_records": persistence,
        "potential_scale": "velocity divided by influence",
        "barrier_definition": "minimum of left and right basin depths",
        "transition_offsets": str(transition_offsets.resolve()),
        "transition_offsets_sha256": hashlib.sha256(
            transition_offsets.read_bytes()
        ).hexdigest(),
    }


def _digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _write_summary(
    rows: list[dict[str, object]],
    cases: tuple[LandscapeCase, ...],
    path: Path,
) -> None:
    order = {case.key: index for index, case in enumerate(cases)}
    selected = sorted(rows, key=lambda row: order[str(row["key"])])
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(selected[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(selected)
    os.replace(temporary, path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transition-offsets", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(MESOSCOPIC_OUTPUT.resolve() / "potential_landscape_timescale_link"),
    )
    parser.add_argument("--designs", nargs="+", choices=DESIGNS, default=DESIGNS)
    parser.add_argument(
        "--configurations",
        nargs="+",
        choices=sorted(SCENARIO_BY_KEY),
        default=[scenario.key for scenario in SCENARIOS],
    )
    parser.add_argument("--common-alpha", type=float, default=0.1)
    parser.add_argument("--common-q", type=float, default=0.1)
    parser.add_argument("--transition-alpha", type=float, default=0.1)
    parser.add_argument("--grid-size", type=int, default=81)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--early-until", type=int, default=200)
    parser.add_argument("--early-every", type=int, default=1)
    parser.add_argument("--record-every", type=int, default=20)
    parser.add_argument("--persistence", type=int, default=3)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.45)
    parser.add_argument("--noise", type=float, default=1e-5)
    parser.add_argument("--jobs", type=int, default=min(os.cpu_count() or 1, 4))
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.jobs < 1 or args.persistence < 1:
        raise ValueError("jobs and persistence must be positive")
    transition_offsets = args.transition_offsets.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = build_cases(
        transition_offsets,
        designs=tuple(args.designs),
        configurations=tuple(args.configurations),
        common_alpha=args.common_alpha,
        common_q=args.common_q,
        transition_alpha=args.transition_alpha,
    )
    selected_steps = record_schedule(
        args.steps,
        early_until=args.early_until,
        early_every=args.early_every,
        late_every=args.record_every,
    )
    base = KineticParameters(
        epsilon=args.epsilon,
        mean_degree=15,
        recsys_count=10,
        random_mix=0.1,
        opinion_tolerance=0.4,
        recommendation_random_ratio=0.0,
        noise_diffusion=args.noise,
        grid_size=args.grid_size,
        dt=args.dt,
        steps=args.steps,
        record_every=args.record_every,
    )
    base.validate()
    protocol = _protocol_payload(
        base, cases, selected_steps, args.persistence, transition_offsets
    )
    protocol_digest = _digest(protocol)
    protocol_path = output_dir / "protocol.json"
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing.get("protocol_digest") != protocol_digest:
            raise ValueError(
                "output directory contains an incompatible protocol; "
                "choose another --output-dir"
            )
    else:
        _atomic_json(protocol_path, {"protocol_digest": protocol_digest, **protocol})

    completed: dict[str, dict[str, object]] = {}
    pending = []
    for case in cases:
        path = output_dir / "cases" / f"{case.key}.npz"
        if path.exists():
            completed[case.key] = _load_checkpoint(path, protocol_digest)
        else:
            pending.append((case, path))
    print(
        f"protocol={protocol_digest[:12]} completed={len(completed)} "
        f"pending={len(pending)} total={len(cases)}",
        flush=True,
    )
    if args.dry_run:
        return

    def accept(row: dict[str, object]) -> None:
        completed[str(row["key"])] = row
        print(
            f"[{len(completed):02d}/{len(cases)}] {row['key']} "
            f"I_w={float(row['I_w']):.4f} "
            f"t_dw={float(row['t_double_well']):g} "
            f"barrier_final={float(row['barrier_final']):.4g}",
            flush=True,
        )

    if args.jobs == 1:
        for case, path in pending:
            accept(
                solve_case(
                    case,
                    base,
                    selected_steps,
                    args.persistence,
                    protocol_digest,
                    path,
                )
            )
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = [
                executor.submit(
                    solve_case,
                    case,
                    base,
                    selected_steps,
                    args.persistence,
                    protocol_digest,
                    path,
                )
                for case, path in pending
            ]
            for future in as_completed(futures):
                accept(future.result())

    if len(completed) != len(cases):
        raise RuntimeError(
            f"landscape scan incomplete: {len(completed)} of {len(cases)}"
        )
    _write_summary(list(completed.values()), cases, output_dir / "summary.csv")
    raw_arguments = sys.argv[1:] if argv is None else argv
    write_run_metadata(
        output_dir / "run_metadata.json",
        analysis="targeted potential landscapes linked to time-scale offsets",
        command=shlex.join(
            [
                sys.executable,
                "-m",
                "experiments.theory_guided.potential_landscape_timescale_link.run",
                *raw_arguments,
            ]
        ),
        parameters=asdict(base),
        configuration={
            **protocol,
            "protocol_digest": protocol_digest,
            "jobs": args.jobs,
            "output_dir": str(output_dir),
        },
    )
    if not args.skip_analysis:
        from experiments.theory_guided.potential_landscape_timescale_link.analyze import (
            analyze_output,
        )

        analyze_output(output_dir)
    print(f"completed targeted potential landscapes: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
