"""Shared helpers for kinetic command-line analyses."""

from __future__ import annotations

import math

import numpy as np


def first_crossing(
    time: np.ndarray,
    values: np.ndarray,
    threshold: float = 0.5,
) -> float | None:
  """Return the first threshold-crossing time, or None when unresolved."""
  found = np.flatnonzero(values >= threshold)
  return float(time[found[0]]) if found.size else None


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
  """Classify crossing order while accepting None or NaN as unresolved."""

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
