"""Run the resumable seven-scenario B=81 macro-time-scale scan."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from ehk.metrics import (
    calculate_channel_progress_snapshots,
    calculate_index_series,
)
from ehk.modeling.mesoscopic import KineticParameters, solve
from experiments.theory_guided.macroscopic_timescale_ratio.scenarios import (
    PAPER_RATES,
    RecommendationScenario,
    record_schedule,
    select_scenarios,
)
from theory.mesoscopic.cli_utils import (
    first_crossing_or_nan,
    pathway_label,
    write_run_metadata,
)
from theory.paths import MESOSCOPIC_OUTPUT


@dataclass(frozen=True)
class TimescaleCell:
  """Scalar results retained from one alpha/q trajectory."""

  configuration: str
  recsys: str
  steepness: float
  closure_note: str
  alpha_index: int
  q_index: int
  alpha: float
  q: float
  path: str
  pathway: float
  t_polarization: float
  t_homophily: float
  precedence: float
  polarization_final: float
  homophily_final: float
  subjective_final: float
  opinion_rate_gamma_0: float
  rewiring_rate_gamma_0: float
  gamma_0: float
  gamma_0p1_time: float
  gamma_0p1_reached: bool
  opinion_rate_gamma_0p1: float
  rewiring_rate_gamma_0p1: float
  gamma_0p1: float

  @property
  def rate_ratio(self) -> float:
    return self.q / self.alpha

  def row(self) -> dict[str, object]:
    return {
        **asdict(self),
        "I_w": self.pathway,
        "t_Ip_0.5": self.t_polarization,
        "t_Ih_0.5": self.t_homophily,
        "rate_ratio": self.rate_ratio,
        "log10_rate_ratio": math.log10(self.rate_ratio),
    }


def _precedence(t_polarization: float, t_homophily: float) -> float:
  if not np.isfinite(t_polarization) or not np.isfinite(t_homophily):
    return float("nan")
  total = t_polarization + t_homophily
  return (
      (t_homophily - t_polarization) / total
      if total > 0
      else float("nan")
  )


def solve_cell(
    scenario: RecommendationScenario,
    alpha_index: int,
    q_index: int,
    alpha: float,
    q: float,
    base: KineticParameters,
    selected_steps: tuple[int, ...],
    probe_dt: float,
) -> TimescaleCell:
  """Solve and reduce one grid cell; suitable for a worker process."""

  parameters = replace(
      base,
      recsys=scenario.recsys,
      recommendation_steepness=scenario.steepness,
      influence=alpha,
      rewiring=q,
  )
  trajectory = solve(parameters, record_steps=selected_steps)
  indices = calculate_index_series(trajectory)
  gamma_0, gamma_0p1 = calculate_channel_progress_snapshots(
      trajectory,
      indices,
      progress_thresholds=(0.0, 0.1),
      probe_dt=probe_dt,
  )
  t_polarization = first_crossing_or_nan(
      indices.time, indices.polarization
  )
  t_homophily = first_crossing_or_nan(indices.time, indices.homophily)
  return TimescaleCell(
      configuration=scenario.key,
      recsys=scenario.recsys,
      steepness=scenario.steepness,
      closure_note=scenario.closure_note,
      alpha_index=alpha_index,
      q_index=q_index,
      alpha=alpha,
      q=q,
      path=pathway_label(t_polarization, t_homophily),
      pathway=indices.pathway,
      t_polarization=t_polarization,
      t_homophily=t_homophily,
      precedence=_precedence(t_polarization, t_homophily),
      polarization_final=float(indices.polarization[-1]),
      homophily_final=float(indices.homophily[-1]),
      subjective_final=float(indices.subjective[-1]),
      opinion_rate_gamma_0=gamma_0.opinion_polarization_rate,
      rewiring_rate_gamma_0=gamma_0.rewiring_homophily_rate,
      gamma_0=gamma_0.gamma,
      gamma_0p1_time=gamma_0p1.time,
      gamma_0p1_reached=gamma_0p1.reached,
      opinion_rate_gamma_0p1=gamma_0p1.opinion_polarization_rate,
      rewiring_rate_gamma_0p1=gamma_0p1.rewiring_homophily_rate,
      gamma_0p1=gamma_0p1.gamma,
  )


def _protocol_payload(
    base: KineticParameters,
    rates: np.ndarray,
    scenarios: tuple[RecommendationScenario, ...],
    selected_steps: tuple[int, ...],
    probe_dt: float,
) -> dict[str, Any]:
  return {
      "parameters": asdict(base),
      "rates": rates.tolist(),
      "scenarios": [asdict(scenario) for scenario in scenarios],
      "record_steps": list(selected_steps),
      "gamma_progress_levels": [0.0, 0.1],
      "probe_dt": probe_dt,
  }


def _protocol_digest(payload: dict[str, Any]) -> str:
  encoded = json.dumps(
      payload, sort_keys=True, separators=(",", ":"), allow_nan=False
  ).encode("utf-8")
  return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
  temporary.write_text(
      json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n",
      encoding="utf-8",
  )
  os.replace(temporary, path)


def _checkpoint_path(
    output_dir: Path,
    configuration: str,
    q_index: int,
    alpha_index: int,
) -> Path:
  return (
      output_dir
      / "cells"
      / configuration
      / f"q{q_index:02d}_a{alpha_index:02d}.json"
  )


def _load_checkpoint(path: Path, expected_digest: str) -> TimescaleCell:
  payload = json.loads(path.read_text(encoding="utf-8"))
  if payload.get("protocol_digest") != expected_digest:
    raise ValueError(f"incompatible checkpoint protocol: {path}")
  return TimescaleCell(**payload["result"])


def _write_checkpoint(
    path: Path,
    cell: TimescaleCell,
    protocol_digest: str,
) -> None:
  _atomic_json(
      path,
      {
          "protocol_digest": protocol_digest,
          "result": asdict(cell),
      },
  )


def _sort_cells(
    cells: list[TimescaleCell],
    scenarios: tuple[RecommendationScenario, ...],
) -> list[TimescaleCell]:
  scenario_index = {
      scenario.key: index for index, scenario in enumerate(scenarios)
  }
  return sorted(
      cells,
      key=lambda cell: (
          scenario_index[cell.configuration],
          cell.q_index,
          cell.alpha_index,
      ),
  )


def write_summary(
    cells: list[TimescaleCell],
    scenarios: tuple[RecommendationScenario, ...],
    rates: np.ndarray,
    output_dir: Path,
) -> None:
  """Write the canonical CSV and compact grid array after a complete scan."""

  ordered = _sort_cells(cells, scenarios)
  rows = [cell.row() for cell in ordered]
  summary_path = output_dir / "summary.csv"
  temporary = summary_path.with_suffix(".csv.tmp")
  with temporary.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
  os.replace(temporary, summary_path)

  shape = (len(scenarios), rates.size, rates.size)
  fields = {
      name: np.full(shape, np.nan, dtype=float)
      for name in (
          "I_w",
          "t_Ip_0.5",
          "t_Ih_0.5",
          "precedence",
          "I_p_final",
          "I_h_final",
          "I_s_final",
          "gamma_0",
          "gamma_0p1",
          "gamma_0p1_time",
      )
  }
  reached = np.zeros(shape, dtype=bool)
  configuration_index = {
      scenario.key: index for index, scenario in enumerate(scenarios)
  }
  for cell in ordered:
    index = (
        configuration_index[cell.configuration],
        cell.q_index,
        cell.alpha_index,
    )
    fields["I_w"][index] = cell.pathway
    fields["t_Ip_0.5"][index] = cell.t_polarization
    fields["t_Ih_0.5"][index] = cell.t_homophily
    fields["precedence"][index] = cell.precedence
    fields["I_p_final"][index] = cell.polarization_final
    fields["I_h_final"][index] = cell.homophily_final
    fields["I_s_final"][index] = cell.subjective_final
    fields["gamma_0"][index] = cell.gamma_0
    fields["gamma_0p1"][index] = cell.gamma_0p1
    fields["gamma_0p1_time"][index] = cell.gamma_0p1_time
    reached[index] = cell.gamma_0p1_reached
  np.savez_compressed(
      output_dir / "timescale_ratio.npz",
      alpha=rates,
      q=rates,
      configuration=np.asarray([scenario.key for scenario in scenarios]),
      gamma_0p1_reached=reached,
      **fields,
  )


def _selected_rates(values: list[float] | None) -> np.ndarray:
  if values is None:
    return PAPER_RATES.copy()
  rates = np.asarray(values, dtype=float)
  if rates.ndim != 1 or rates.size < 1 or np.any(~np.isfinite(rates)):
    raise ValueError("--rates must contain finite values")
  if np.any(rates <= 0) or np.any(rates > 1) or np.unique(rates).size != rates.size:
    raise ValueError("--rates must be unique and lie in (0, 1]")
  return rates


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--grid-size", type=int, default=81)
  parser.add_argument("--steps", type=int, default=4000)
  parser.add_argument("--dt", type=float, default=1.0)
  parser.add_argument("--early-until", type=int, default=200)
  parser.add_argument("--early-every", type=int, default=1)
  parser.add_argument("--record-every", type=int, default=20)
  parser.add_argument("--epsilon", type=float, default=0.45)
  parser.add_argument("--noise", type=float, default=1e-5)
  parser.add_argument("--mean-degree", type=float, default=15.0)
  parser.add_argument("--recsys-count", type=int, default=10)
  parser.add_argument("--random-mix", type=float, default=0.1)
  parser.add_argument("--opinion-tolerance", type=float, default=0.4)
  parser.add_argument("--random-ratio", type=float, default=0.0)
  parser.add_argument("--probe-dt", type=float, default=1.0)
  parser.add_argument("--rates", nargs="+", type=float)
  parser.add_argument(
      "--configurations",
      nargs="+",
      choices=sorted(scenario.key for scenario in select_scenarios(None)),
  )
  parser.add_argument(
      "--jobs", type=int, default=min(os.cpu_count() or 1, 8)
  )
  parser.add_argument(
      "--output-dir",
      type=Path,
      default=(
          MESOSCOPIC_OUTPUT.resolve() / "macroscopic_timescale_ratio_b81"
      ),
  )
  parser.add_argument(
      "--skip-analysis",
      action="store_true",
      help="write numerical outputs without fitting offsets or plotting",
  )
  return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
  args = parse_args(argv)
  if args.jobs < 1:
    raise ValueError("--jobs must be positive")
  rates = _selected_rates(args.rates)
  scenarios = select_scenarios(args.configurations)
  selected_steps = record_schedule(
      args.steps,
      early_until=args.early_until,
      early_every=args.early_every,
      late_every=args.record_every,
  )
  base = KineticParameters(
      epsilon=args.epsilon,
      mean_degree=args.mean_degree,
      recsys_count=args.recsys_count,
      random_mix=args.random_mix,
      opinion_tolerance=args.opinion_tolerance,
      recommendation_random_ratio=args.random_ratio,
      noise_diffusion=args.noise,
      grid_size=args.grid_size,
      dt=args.dt,
      steps=args.steps,
      record_every=args.record_every,
  )
  base.validate()
  protocol = _protocol_payload(
      base, rates, scenarios, selected_steps, args.probe_dt
  )
  digest = _protocol_digest(protocol)
  args.output_dir.mkdir(parents=True, exist_ok=True)
  protocol_path = args.output_dir / "protocol.json"
  if protocol_path.exists():
    existing = json.loads(protocol_path.read_text(encoding="utf-8"))
    if existing.get("protocol_digest") != digest:
      raise ValueError(
          "output directory contains an incompatible protocol; "
          "choose another --output-dir"
      )
  else:
    _atomic_json(
        protocol_path,
        {"protocol_digest": digest, **protocol},
    )

  completed: dict[tuple[str, int, int], TimescaleCell] = {}
  pending: list[
      tuple[RecommendationScenario, int, int, float, float]
  ] = []
  for scenario in scenarios:
    for q_index, q in enumerate(rates):
      for alpha_index, alpha in enumerate(rates):
        key = scenario.key, q_index, alpha_index
        checkpoint = _checkpoint_path(
            args.output_dir, scenario.key, q_index, alpha_index
        )
        if checkpoint.exists():
          completed[key] = _load_checkpoint(checkpoint, digest)
        else:
          pending.append(
              (
                  scenario,
                  alpha_index,
                  q_index,
                  float(alpha),
                  float(q),
              )
          )
  total = len(completed) + len(pending)
  print(
      f"protocol={digest[:12]} completed={len(completed)} "
      f"pending={len(pending)} total={total}",
      flush=True,
  )

  def accept(cell: TimescaleCell) -> None:
    key = cell.configuration, cell.q_index, cell.alpha_index
    _write_checkpoint(
        _checkpoint_path(
            args.output_dir,
            cell.configuration,
            cell.q_index,
            cell.alpha_index,
        ),
        cell,
        digest,
    )
    completed[key] = cell
    print(
        f"[{len(completed):03d}/{total}] {cell.configuration} "
        f"alpha={cell.alpha:g} q={cell.q:g} I_w={cell.pathway:.4f} "
        f"Gamma(0.1)={cell.gamma_0p1:.4g}",
        flush=True,
    )

  if args.jobs == 1:
    for case in pending:
      accept(
          solve_cell(
              *case,
              base,
              selected_steps,
              args.probe_dt,
          )
      )
  else:
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
      future_to_case = {
          executor.submit(
              solve_cell,
              *case,
              base,
              selected_steps,
              args.probe_dt,
          ): case
          for case in pending
      }
      for future in as_completed(future_to_case):
        accept(future.result())

  cells = list(completed.values())
  if len(cells) != total:
    raise RuntimeError(f"scan incomplete: retained {len(cells)} of {total} cells")
  write_summary(cells, scenarios, rates, args.output_dir)
  raw_arguments = sys.argv[1:] if argv is None else argv
  write_run_metadata(
      args.output_dir / "run_metadata.json",
      analysis="seven-configuration B=81 macroscopic time-scale scan",
      command=shlex.join(
          [
              sys.executable,
              "-m",
              "experiments.theory_guided.macroscopic_timescale_ratio.run",
              *raw_arguments,
          ]
      ),
      parameters=asdict(base),
      configuration={
          **protocol,
          "protocol_digest": digest,
          "jobs": args.jobs,
          "output_dir": str(args.output_dir.resolve()),
          "resume_checkpoint_count": total,
      },
  )
  if not args.skip_analysis:
    from experiments.theory_guided.macroscopic_timescale_ratio.analyze import (
        analyze_output,
    )

    analyze_output(args.output_dir)
  print(f"completed seven-scenario scan: {args.output_dir}", flush=True)


if __name__ == "__main__":
  main()
