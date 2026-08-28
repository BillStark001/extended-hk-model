"""Run the resumable epsilon/dynamics/solver multibarrier comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import sys
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from ehk.metrics import (
    calculate_index_series,
    potential_from_force,
    quantify_landscape_series,
    quantify_multiwell_series,
)
from ehk.metrics.landscape_diagnostics import first_persistent_time
from ehk.modeling.mesoscopic import KineticParameters, solve
from experiments.theory_guided.multibarrier_dynamics_comparison.protocol import (
    DEFAULT_CONFIGURATIONS,
    DEFAULT_DYNAMICS,
    DEFAULT_METHODS,
    PAPER_RATES,
    ScanCase,
    build_cases,
    parse_epsilon_grids,
    record_schedule,
    select_grid_cells,
    select_scenarios,
)
from theory.mesoscopic.cli_utils import pathway_label, write_run_metadata
from theory.paths import MESOSCOPIC_OUTPUT

SCHEMA_VERSION = 1
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
IMPLEMENTATION_SOURCES = (
    "src/ehk/modeling/mesoscopic/solver.py",
    "src/ehk/modeling/opinion_cells.py",
    "src/ehk/metrics/density_indices.py",
    "src/ehk/metrics/landscape_diagnostics.py",
    "experiments/theory_guided/multibarrier_dynamics_comparison/protocol.py",
    "experiments/theory_guided/multibarrier_dynamics_comparison/run.py",
)


def _implementation_digest() -> str:
    digest = hashlib.sha256()
    for relative in IMPLEMENTATION_SOURCES:
        digest.update(relative.encode("utf-8"))
        digest.update((REPOSITORY_ROOT / relative).read_bytes())
    return digest.hexdigest()


def _persistent_crossing(
    time: np.ndarray,
    values: np.ndarray,
    threshold: float,
    persistence: int,
) -> float:
    return first_persistent_time(time, values >= threshold, persistence=persistence)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _checkpoint_path(output_dir: Path, case: ScanCase) -> Path:
    return output_dir / "cells" / f"{case.key}.npz"


def _protocol_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _multiwell_arrays(series: Any) -> dict[str, Any]:
    """Serialize all padded multibarrier diagnostics without pickled objects."""

    return {
        "multi_well_count": series.well_count,
        "multi_well_position": series.well_position,
        "multi_well_potential": series.well_potential,
        "multi_basin_mass": series.basin_mass,
        "multi_basin_left": series.basin_left,
        "multi_basin_right": series.basin_right,
        "multi_well_id": series.well_id,
        "multi_barrier_count": series.barrier_count,
        "multi_barrier_position": series.barrier_position,
        "multi_barrier_height": series.barrier_height,
        "multi_barrier_macro_score": series.barrier_macro_score,
        "multi_barrier_left_id": series.barrier_left_id,
        "multi_barrier_right_id": series.barrier_right_id,
        "multi_robust_well_count": series.robust_well_count,
        "multi_robust_barrier_count": series.robust_barrier_count,
        "multi_effective_well_count": series.effective_well_count,
        "multi_dominant_barrier_height": series.dominant_barrier_height,
        "multi_dominant_macro_score": series.dominant_macro_score,
        "multi_dominant_barrier_position": series.dominant_barrier_position,
        "multi_dominant_left_id": series.dominant_left_id,
        "multi_dominant_right_id": series.dominant_right_id,
        "multi_dominant_switch": series.dominant_switch,
        "multi_well_birth_count": series.well_birth_count,
        "multi_well_death_count": series.well_death_count,
        "multi_overshoot_class": np.asarray(series.overshoot_class),
    }


def solve_case(
    case: ScanCase,
    base: KineticParameters,
    selected_steps: tuple[int, ...],
    persistence: int,
    min_basin_mass: float,
    min_barrier_height: float,
    max_well_displacement: float,
    overshoot_tolerance: float,
    protocol_digest: str,
    output_path: Path,
) -> dict[str, object]:
    """Solve, reduce, and atomically checkpoint one factorial case."""

    parameters = replace(
        base,
        epsilon=case.epsilon,
        grid_size=case.grid_size,
        dynamics=case.dynamics,
        opinion_method=case.opinion_method,
        recsys=case.recsys,
        recommendation_steepness=case.steepness,
        influence=case.alpha,
        rewiring=case.q,
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
    multiwell = quantify_multiwell_series(
        trajectory.x,
        trajectory.time,
        potential,
        trajectory.rho,
        min_basin_mass=min_basin_mass,
        min_barrier_height=min_barrier_height,
        max_well_displacement=max_well_displacement,
        overshoot_tolerance=overshoot_tolerance,
    )
    t_polarization = _persistent_crossing(
        indices.time, indices.polarization, 0.5, persistence
    )
    t_homophily = _persistent_crossing(
        indices.time, indices.homophily, 0.5, persistence
    )
    variance_mean = np.sum(trajectory.rho * trajectory.displacement_variance, axis=1)
    second_moment_mean = np.sum(
        trajectory.rho * trajectory.displacement_second_moment, axis=1
    )
    diffusion_mean = np.sum(trajectory.rho * trajectory.endogenous_diffusion, axis=1)
    row: dict[str, object] = {
        **case.payload(),
        "case_key": case.key,
        "rate_ratio": case.q / case.alpha,
        "path": pathway_label(t_polarization, t_homophily),
        "I_w": indices.pathway,
        "t_Ip_0.5": t_polarization,
        "t_Ih_0.5": t_homophily,
        "I_p_final": float(indices.polarization[-1]),
        "I_h_final": float(indices.homophily[-1]),
        "I_s_final": float(indices.subjective[-1]),
        "double_well_formation_time": landscape.formation_time,
        "double_well_barrier_peak": float(np.max(landscape.barrier_height)),
        "double_well_barrier_final": float(landscape.barrier_height[-1]),
        "multiwell_peak_time": multiwell.peak_time,
        "multiwell_barrier_peak": multiwell.peak_barrier_height,
        "multiwell_barrier_final": multiwell.final_barrier_height,
        "multiwell_overshoot_absolute": multiwell.overshoot_absolute,
        "multiwell_overshoot_relative": multiwell.overshoot_relative,
        "multiwell_overshoot_class": multiwell.overshoot_class,
        "robust_well_count_peak": int(np.max(multiwell.robust_well_count)),
        "robust_well_count_final": int(multiwell.robust_well_count[-1]),
        "robust_barrier_count_peak": int(np.max(multiwell.robust_barrier_count)),
        "robust_barrier_count_final": int(multiwell.robust_barrier_count[-1]),
        "effective_well_count_peak": float(np.max(multiwell.effective_well_count)),
        "effective_well_count_final": float(multiwell.effective_well_count[-1]),
        "dominant_pair_switch_count": int(np.sum(multiwell.dominant_switch)),
        "well_birth_count_after_initial": int(np.sum(multiwell.well_birth_count[1:])),
        "well_death_count": int(np.sum(multiwell.well_death_count)),
        "conditional_displacement_variance_peak": float(np.max(variance_mean)),
        "conditional_displacement_variance_final": float(variance_mean[-1]),
        "displacement_second_moment_peak": float(np.max(second_moment_mean)),
        "displacement_second_moment_final": float(second_moment_mean[-1]),
        "endogenous_diffusion_peak": float(np.max(diffusion_mean)),
        "endogenous_diffusion_final": float(diffusion_mean[-1]),
    }
    arrays = {
        "schema_version": np.asarray(SCHEMA_VERSION),
        "protocol_digest": np.asarray(protocol_digest),
        "case_json": np.asarray(json.dumps(case.payload(), sort_keys=True)),
        "parameters_json": np.asarray(json.dumps(asdict(parameters), sort_keys=True)),
        "summary_json": np.asarray(json.dumps(row, sort_keys=True, allow_nan=True)),
        "x": trajectory.x,
        "time": trajectory.time,
        "rho": trajectory.rho,
        "velocity": trajectory.velocity,
        "force": force,
        "potential": potential,
        "conditional_displacement_variance": trajectory.displacement_variance,
        "displacement_second_moment": trajectory.displacement_second_moment,
        "endogenous_diffusion": trajectory.endogenous_diffusion,
        "conditional_displacement_variance_mean": variance_mean,
        "displacement_second_moment_mean": second_moment_mean,
        "endogenous_diffusion_mean": diffusion_mean,
        "I_p": indices.polarization,
        "I_h": indices.homophily,
        "I_s": indices.subjective,
        "I_w": np.asarray(indices.pathway),
        "double_well": landscape.double_well,
        "double_well_barrier_height": landscape.barrier_height,
        "double_well_left_position": landscape.left_position,
        "double_well_right_position": landscape.right_position,
        "double_well_barrier_position": landscape.barrier_position,
        **_multiwell_arrays(multiwell),
    }
    _atomic_npz(output_path, arrays)
    return row


def _load_checkpoint(path: Path, protocol_digest: str) -> dict[str, object]:
    try:
        with np.load(path, allow_pickle=False) as arrays:
            if int(arrays["schema_version"].item()) != SCHEMA_VERSION:
                raise ValueError(f"unsupported checkpoint schema: {path}")
            if str(arrays["protocol_digest"].item()) != protocol_digest:
                raise ValueError(f"incompatible checkpoint protocol: {path}")
            return json.loads(str(arrays["summary_json"].item()))
    except (OSError, KeyError, ValueError) as error:
        raise ValueError(f"invalid checkpoint {path}: {error}") from error


def _write_summary(
    rows: list[dict[str, object]],
    cases: tuple[ScanCase, ...],
    path: Path,
) -> None:
    order = {case.key: index for index, case in enumerate(cases)}
    ordered = sorted(rows, key=lambda row: order[str(row["case_key"])])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(ordered[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(ordered)
    os.replace(temporary, path)


def _terminate_executor(
    executor: ProcessPoolExecutor,
    futures: list[Future[dict[str, object]]],
) -> None:
    """Terminate only this worker pool while preserving completed checkpoints."""

    for future in futures:
        future.cancel()
    process_map = getattr(executor, "_processes", None)
    processes = tuple(process_map.values()) if process_map else ()
    for process in processes:
        if process.is_alive():
            process.terminate()
    executor.shutdown(wait=True, cancel_futures=True)


def _parse_explicit_cell(value: str) -> tuple[int, int]:
    try:
        q_text, alpha_text = value.split(":", maxsplit=1)
        return int(q_text), int(alpha_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "explicit cells must be Q_INDEX:ALPHA_INDEX"
        ) from error


def _selected_rates(values: list[float] | None) -> np.ndarray:
    rates = PAPER_RATES.copy() if values is None else np.asarray(values, dtype=float)
    if (
        rates.ndim != 1
        or rates.size < 1
        or np.any(~np.isfinite(rates))
        or np.any(rates <= 0)
        or np.any(rates > 1)
        or np.unique(rates).size != rates.size
    ):
        raise ValueError("rates must be unique and lie in (0, 1]")
    return rates


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--epsilon-grid",
        action="append",
        help="repeat EPSILON:GRID; defaults to 0.2:121, 0.4:81, 0.8:81",
    )
    parser.add_argument("--rates", nargs="+", type=float)
    parser.add_argument(
        "--grid-selection",
        choices=("anti_diagonal_band", "full", "explicit"),
        default="anti_diagonal_band",
    )
    parser.add_argument("--band-offsets", nargs="+", type=int, default=(-1, 0, 1))
    parser.add_argument(
        "--explicit-cell",
        action="append",
        type=_parse_explicit_cell,
        default=[],
        metavar="Q_INDEX:ALPHA_INDEX",
    )
    parser.add_argument(
        "--dynamics", nargs="+", choices=DEFAULT_DYNAMICS, default=DEFAULT_DYNAMICS
    )
    parser.add_argument(
        "--methods", nargs="+", choices=DEFAULT_METHODS, default=DEFAULT_METHODS
    )
    parser.add_argument(
        "--configurations",
        nargs="+",
        choices=sorted(scenario.key for scenario in select_scenarios(None)),
        default=DEFAULT_CONFIGURATIONS,
    )
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--early-until", type=int, default=200)
    parser.add_argument("--early-every", type=int, default=1)
    parser.add_argument("--record-every", type=int, default=20)
    parser.add_argument("--persistence", type=int, default=3)
    parser.add_argument("--noise", type=float, default=1e-5)
    parser.add_argument("--mean-degree", type=float, default=15.0)
    parser.add_argument("--recsys-count", type=int, default=10)
    parser.add_argument("--random-mix", type=float, default=0.1)
    parser.add_argument("--opinion-tolerance", type=float, default=0.4)
    parser.add_argument("--random-ratio", type=float, default=0.0)
    parser.add_argument("--min-basin-mass", type=float, default=0.02)
    parser.add_argument("--min-barrier-height", type=float, default=1e-4)
    parser.add_argument("--max-well-displacement", type=float, default=0.25)
    parser.add_argument("--overshoot-tolerance", type=float, default=1e-4)
    parser.add_argument("--jobs", type=int, default=min(os.cpu_count() or 1, 4))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(MESOSCOPIC_OUTPUT.resolve() / "multibarrier_dynamics_comparison"),
    )
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.jobs < 1 or args.persistence < 1:
        raise ValueError("jobs and persistence must be positive")
    if len(set(args.dynamics)) != len(args.dynamics):
        raise ValueError("dynamics must not be repeated")
    if len(set(args.methods)) != len(args.methods):
        raise ValueError("methods must not be repeated")
    epsilon_grids = parse_epsilon_grids(args.epsilon_grid)
    rates = _selected_rates(args.rates)
    scenarios = select_scenarios(list(args.configurations))
    cells = select_grid_cells(
        rates,
        selection=args.grid_selection,
        band_offsets=tuple(args.band_offsets),
        explicit_cells=tuple(args.explicit_cell),
    )
    cases = build_cases(
        epsilon_grids,
        tuple(args.dynamics),
        tuple(args.methods),
        scenarios,
        cells,
    )
    selected_steps = record_schedule(
        args.steps,
        early_until=args.early_until,
        early_every=args.early_every,
        late_every=args.record_every,
    )
    base = KineticParameters(
        mean_degree=args.mean_degree,
        recsys_count=args.recsys_count,
        random_mix=args.random_mix,
        opinion_tolerance=args.opinion_tolerance,
        recommendation_random_ratio=args.random_ratio,
        noise_diffusion=args.noise,
        dt=args.dt,
        steps=args.steps,
        record_every=args.record_every,
    )
    base.validate()
    protocol: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "implementation_digest": _implementation_digest(),
        "parameters": asdict(base),
        "epsilon_grids": [asdict(item) for item in epsilon_grids],
        "rates": rates.tolist(),
        "grid_selection": args.grid_selection,
        "band_offsets": list(args.band_offsets),
        "cells": [asdict(cell) for cell in cells],
        "dynamics": list(args.dynamics),
        "methods": list(args.methods),
        "scenarios": [asdict(scenario) for scenario in scenarios],
        "record_steps": list(selected_steps),
        "persistence_records": args.persistence,
        "min_basin_mass": args.min_basin_mass,
        "min_barrier_height": args.min_barrier_height,
        "max_well_displacement": args.max_well_displacement,
        "overshoot_tolerance": args.overshoot_tolerance,
        "potential_scale": "velocity divided by influence",
        "case_count": len(cases),
    }
    digest = _protocol_digest(protocol)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = output_dir / "protocol.json"
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing.get("protocol_digest") != digest:
            raise ValueError(
                "output directory contains an incompatible protocol; "
                "choose another --output-dir"
            )
    else:
        _atomic_json(protocol_path, {"protocol_digest": digest, **protocol})

    completed: dict[str, dict[str, object]] = {}
    pending: list[tuple[ScanCase, Path]] = []
    for case in cases:
        path = _checkpoint_path(output_dir, case)
        if path.exists():
            completed[case.key] = _load_checkpoint(path, digest)
        else:
            pending.append((case, path))
    print(
        f"protocol={digest[:12]} completed={len(completed)} "
        f"pending={len(pending)} total={len(cases)}",
        flush=True,
    )
    if args.dry_run:
        return

    def accept(row: dict[str, object]) -> None:
        key = str(row["case_key"])
        completed[key] = row
        print(
            f"[{len(completed):04d}/{len(cases)}] {key} "
            f"I_w={float(row['I_w']):.4f} "
            f"wells={int(row['robust_well_count_peak'])} "
            f"overshoot={row['multiwell_overshoot_class']}",
            flush=True,
        )

    arguments = (
        base,
        selected_steps,
        args.persistence,
        args.min_basin_mass,
        args.min_barrier_height,
        args.max_well_displacement,
        args.overshoot_tolerance,
        digest,
    )
    if args.jobs == 1:
        for case, path in pending:
            accept(solve_case(case, *arguments, path))
    else:
        executor = ProcessPoolExecutor(max_workers=args.jobs)
        futures = [
            executor.submit(solve_case, case, *arguments, path)
            for case, path in pending
        ]
        try:
            for future in as_completed(futures):
                accept(future.result())
        except KeyboardInterrupt:
            _terminate_executor(executor, futures)
            print(
                "interrupted; all completed case files remain resumable",
                flush=True,
            )
            raise SystemExit(130) from None
        except Exception:
            _terminate_executor(executor, futures)
            raise
        else:
            executor.shutdown(wait=True)

    if len(completed) != len(cases):
        raise RuntimeError(f"scan incomplete: {len(completed)} of {len(cases)}")
    _write_summary(list(completed.values()), cases, output_dir / "summary.csv")
    raw_arguments = sys.argv[1:] if argv is None else argv
    write_run_metadata(
        output_dir / "run_metadata.json",
        analysis="epsilon/dynamics/solver multibarrier comparison",
        command=shlex.join(
            [
                sys.executable,
                "-m",
                "experiments.theory_guided.multibarrier_dynamics_comparison.run",
                *raw_arguments,
            ]
        ),
        parameters=asdict(base),
        configuration={
            **protocol,
            "protocol_digest": digest,
            "jobs": args.jobs,
            "output_dir": str(output_dir),
        },
    )
    if not args.skip_analysis:
        from experiments.theory_guided.multibarrier_dynamics_comparison.analyze import (
            analyze_output,
        )

        analyze_output(output_dir)
    print(f"completed multibarrier comparison: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
