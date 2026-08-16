"""Shared helpers for kinetic command-line analyses."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REPRODUCIBILITY_SOURCES = (
    "src/ehk/modeling/mesoscopic/solver.py",
    "src/ehk/metrics/density_indices.py",
    "theory/mesoscopic/cli_utils.py",
    "theory/mesoscopic/phase_scan.py",
    "theory/mesoscopic/recommender_scan.py",
    "theory/mesoscopic/spectrum_check.py",
    "theory/mesoscopic/joint_spectrum_operator.py",
    "theory/mesoscopic/joint_spectrum.py",
    "theory/mesoscopic/joint_spectrum_convergence.py",
    "theory/mesoscopic/spectrum_steady_states.py",
    "theory/mesoscopic/convergence_report.py",
)


def _package_version(name: str) -> str:
  try:
    return importlib.metadata.version(name)
  except importlib.metadata.PackageNotFoundError:
    return "not-installed"


def _git_output(*args: str) -> str | None:
  try:
    completed = subprocess.run(
        ("git", *args),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
  except (OSError, subprocess.CalledProcessError):
    return None
  return completed.stdout.strip()


def write_run_metadata(
    output_path: Path,
    *,
    analysis: str,
    command: str,
    parameters: dict[str, Any],
    configuration: dict[str, Any],
) -> None:
  """Write enough environment and source information to audit a scan."""

  source_hashes = {}
  for relative in REPRODUCIBILITY_SOURCES:
    source_path = REPOSITORY_ROOT / relative
    source_hashes[relative] = hashlib.sha256(source_path.read_bytes()).hexdigest()
  status = _git_output("status", "--porcelain")
  payload = {
      "schema_version": 1,
      "analysis": analysis,
      "generated_at_utc": datetime.now(timezone.utc).isoformat(),
      "command": command,
      "parameters": parameters,
      "configuration": configuration,
      "environment": {
          "python": platform.python_version(),
          "numpy": _package_version("numpy"),
          "scipy": _package_version("scipy"),
          "matplotlib": _package_version("matplotlib"),
      },
      "source": {
          "git_commit": _git_output("rev-parse", "HEAD"),
          "git_dirty": bool(status) if status is not None else None,
          "sha256": source_hashes,
      },
  }
  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )


def first_crossing(
    time: np.ndarray,
    values: np.ndarray,
    threshold: float = 0.5,
) -> float | None:
  """Return the linearly interpolated first crossing, or None if unresolved."""
  found = np.flatnonzero(values >= threshold)
  if not found.size:
    return None
  index = int(found[0])
  if index == 0:
    return float(time[0])
  value_before = float(values[index - 1])
  value_after = float(values[index])
  if value_after <= value_before:
    return float(time[index])
  fraction = (threshold - value_before) / (value_after - value_before)
  fraction = min(max(fraction, 0.0), 1.0)
  return float(
      time[index - 1] + fraction * (time[index] - time[index - 1])
  )


def first_crossing_or_nan(
    time: np.ndarray,
    values: np.ndarray,
    threshold: float = 0.5,
) -> float:
  """Return the first crossing or NaN for array-oriented sweep outputs."""
  value = first_crossing(time, values, threshold)
  return float("nan") if value is None else value


def pathway_label(
    polarization_time: float | None,
    homophily_time: float | None,
) -> str:
  """Classify observed crossing order while accepting None or NaN.

  If only one threshold is reached within the horizon, that crossing is
  assigned precedence. 'unresolved' is reserved for reaching neither.
  """

  def resolved(value: float | None) -> bool:
    return value is not None and math.isfinite(value)

  has_polarization = resolved(polarization_time)
  has_homophily = resolved(homophily_time)
  if not has_polarization and not has_homophily:
    return "unresolved"
  if not has_homophily:
    return "PbS"
  if not has_polarization:
    return "SbP"
  assert polarization_time is not None and homophily_time is not None
  if polarization_time < homophily_time:
    return "PbS"
  if homophily_time < polarization_time:
    return "SbP"
  return "simultaneous"
