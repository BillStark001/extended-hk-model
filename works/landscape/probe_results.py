"""Load and plot frozen-state probe landscapes for mechanism-map figures."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal, get_args

import numpy as np
from matplotlib.axes import Axes

from works.landscape.config import DEFAULT_RESULT_DIR, SCENARIOS
from works.landscape.plot_utils import (
    plot_potential_curves,
    potential_from_force,
)


LandscapeKind = Literal["observational", "probe"]
LANDSCAPE_CHOICES = get_args(LandscapeKind)


@dataclass(frozen=True)
class ProbeLandscape:
  x: np.ndarray
  normalized_time: np.ndarray
  total_force: np.ndarray
  neighbor_force: np.ndarray

  def force(self, *, neighbor_only: bool) -> np.ndarray:
    return self.neighbor_force if neighbor_only else self.total_force

  def potential(self, *, neighbor_only: bool) -> np.ndarray:
    return potential_from_force(
        self.x,
        self.force(neighbor_only=neighbor_only),
    )


@lru_cache(maxsize=None)
def load_probe_landscape(
    scenario_key: str,
    result_dir: str | Path = DEFAULT_RESULT_DIR,
) -> ProbeLandscape:
  """Load one probe NPZ and validate that it belongs to the requested record."""
  if scenario_key not in SCENARIOS:
    raise KeyError(f"unknown mechanism scenario: {scenario_key}")

  path = Path(result_dir) / f"counterfactual_probe_{scenario_key}.npz"
  if not path.is_file():
    raise FileNotFoundError(
        f"probe landscape not found: {path}; run "
        f"`python -m works.landscape.counterfactual_probe "
        f"--scenario {scenario_key} --no-plot` first"
    )

  with np.load(path) as result:
    result_key = str(result["scenario_key"].item())
    if result_key != scenario_key:
      raise ValueError(
          f"{path} contains scenario {result_key!r}, "
          f"expected {scenario_key!r}"
      )
    return ProbeLandscape(
        x=np.asarray(result["x"], dtype=float),
        normalized_time=np.asarray(result["normalized_time"], dtype=float),
        total_force=np.asarray(result["social_force_mean"], dtype=float),
        neighbor_force=np.asarray(
            result["social_force_neighbor_mean"],
            dtype=float,
        ),
    )


def plot_probe_potential(
    potential_axis: Axes,
    scenario_key: str,
    *,
    neighbor_only: bool,
    xlabel: bool,
    ylabel: bool,
    result_dir: str | Path = DEFAULT_RESULT_DIR,
) -> np.ndarray:
  """Plot a probe potential while leaving the trajectory panel unchanged."""
  result = load_probe_landscape(scenario_key, result_dir)
  potential = result.potential(neighbor_only=neighbor_only)

  plot_potential_curves(
      potential_axis,
      result.x,
      potential,
      result.normalized_time,
      xlabel=xlabel,
      ylabel=ylabel,
  )
  return potential
