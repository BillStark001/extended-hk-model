"""Shared numerical and visual helpers for potential landscapes."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ehk.metrics.landscape_diagnostics import potential_from_force as _potential_from_force


CMAP_NAME = "managua"


def potential_from_force(
    x: np.ndarray,
    force: np.ndarray,
    *,
    center: bool = True,
) -> np.ndarray:
  """Backward-compatible plotting-level alias for the numerical helper."""
  return _potential_from_force(x, force, center=center)


def format_landscape_axis(
    axis: Axes,
    *,
    xlabel: str | None,
    ylabel: str | None,
    ylim: tuple[float, float] | None = None,
) -> None:
  """Apply the existing paper landscape axis style."""
  axis.set_xlim(-1, 1)
  if ylim is not None:
    axis.set_ylim(*ylim)
  axis.grid(True, linestyle="--", alpha=0.5)
  if xlabel is None:
    axis.set_xticklabels([])
  else:
    axis.set_xlabel(xlabel)
  if ylabel is None:
    axis.set_yticklabels([])
  else:
    axis.set_ylabel(ylabel)


def plot_time_curves(
    axis: Axes,
    x: np.ndarray,
    curves: Sequence[np.ndarray] | np.ndarray,
    normalized_times: Sequence[float] | np.ndarray,
) -> None:
  """Plot normalized-time curves using the original landscape styling."""
  cmap = plt.get_cmap(CMAP_NAME)
  for curve, normalized_time in zip(curves, normalized_times):
    axis.plot(
        x,
        curve,
        color=cmap(float(normalized_time)),
        linewidth=1,
        alpha=0.8,
    )


def plot_potential_curves(
    axis: Axes,
    x: np.ndarray,
    potentials: Sequence[np.ndarray] | np.ndarray,
    normalized_times: Sequence[float] | np.ndarray,
    *,
    xlabel: bool = True,
    ylabel: bool = True,
) -> None:
  plot_time_curves(axis, x, potentials, normalized_times)
  format_landscape_axis(
      axis,
      xlabel=r"$x$" if xlabel else None,
      ylabel=r"$V(x)$" if ylabel else None,
  )


def plot_normalized_time_colorbar(
    figure: Figure,
    axes: Axes | Iterable[Axes],
    *,
    pad: float = 0.01,
):
  norm = mpl.colors.Normalize(vmin=0, vmax=1)
  scalar = plt.cm.ScalarMappable(cmap=CMAP_NAME, norm=norm)
  scalar.set_array([])
  colorbar = figure.colorbar(
      scalar,
      ax=axes,
      orientation="vertical",
      aspect=30,
      pad=pad,
  )
  colorbar.set_label(r"$t_n$")
  return colorbar


def set_shared_y_limits(
    axes: Iterable[Axes],
    values: Sequence[np.ndarray],
    *,
    padding_ratio: float = 0.04,
) -> None:
  lower = min(float(np.min(value)) for value in values)
  upper = max(float(np.max(value)) for value in values)
  padding = padding_ratio * max(upper - lower, 1e-12)
  for axis in axes:
    axis.set_ylim(lower - padding, upper + padding)


def save_landscape_figure(
    figure: Figure,
    output_stem: str | Path,
) -> tuple[Path, Path]:
  """Save the standard PDF/PNG pair without opening an interactive window."""
  stem = Path(output_stem)
  stem.parent.mkdir(parents=True, exist_ok=True)
  png_path = stem.with_suffix(".png")
  pdf_path = stem.with_suffix(".pdf")
  figure.savefig(pdf_path, dpi=300, bbox_inches="tight")
  figure.savefig(png_path, dpi=300, bbox_inches="tight")
  plt.close(figure)
  return png_path, pdf_path
