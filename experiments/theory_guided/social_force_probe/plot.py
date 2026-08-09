"""Plot force fields and potentials produced by the counterfactual probe."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from ehk.common.plotting import plt_figure, setup_paper_params
from experiments.theory_guided.social_force_probe.scenarios import OUTPUT_DIR, SCENARIOS
from ehk.common.plotting.landscape import (
    format_landscape_axis,
    plot_normalized_time_colorbar,
    plot_potential_curves,
    plot_time_curves,
    save_landscape_figure,
    set_shared_y_limits,
)


def plot_results(
    result_paths: Sequence[Path],
    output_stem: Path,
) -> tuple[Path, Path]:
  if not result_paths:
    raise ValueError("at least one result file is required")

  records = [np.load(path) for path in result_paths]
  try:
    setup_paper_params()
    fig, axes_raw = plt_figure(
        len(records),
        2,
        total_width=8,
        hw_ratio=0.68,
        squeeze=False,
    )
    axes = np.asarray(axes_raw)
    forces = [record["social_force_mean"] for record in records]
    potentials = [record["potential"] for record in records]
    force_limit = 1.05 * max(
        float(np.max(np.abs(force)))
        for force in forces
    )

    for row, record in enumerate(records):
      x = record["x"]
      normalized_time = record["normalized_time"]
      force = forces[row]
      potential = potentials[row]
      key = str(record["scenario_key"].item())
      label = (
          SCENARIOS[key].label
          if key in SCENARIOS
          else str(record["scenario_name"].item())
      )

      force_ax = axes[row, 0]
      potential_ax = axes[row, 1]
      show_xlabels = row == len(records) - 1
      panel = chr(ord("a") + row)
      plot_time_curves(force_ax, x, force, normalized_time)
      force_ax.axhline(0.0, color="0.35", linewidth=0.7, linestyle="--")
      format_landscape_axis(
          force_ax,
          xlabel=r"$x_i(t)$" if show_xlabels else None,
          ylabel=r"$\Delta_{\mathrm{CF}} x_i(t)$",
          ylim=(-force_limit, force_limit),
      )
      plot_potential_curves(
          potential_ax,
          x,
          potential,
          normalized_time,
          xlabel=show_xlabels,
          ylabel=True,
      )
      force_ax.set_title(
          f"({panel}) {label}, counterfactual NOD",
          loc="left",
      )
      potential_ax.set_title(
          f"({panel}') Potential of ({panel})",
          loc="left",
      )

    set_shared_y_limits(
        axes[:, 1],
        potentials,
        padding_ratio=0.04,
    )
    plot_normalized_time_colorbar(fig, axes.ravel().tolist(), pad=0.03)
    return save_landscape_figure(fig, output_stem)
  finally:
    for record in records:
      record.close()


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("results", nargs="+", type=Path)
  parser.add_argument(
      "--output-stem",
      type=Path,
      default=OUTPUT_DIR.resolve() / "figures" / "counterfactual_probe_landscape",
  )
  return parser.parse_args()


if __name__ == "__main__":
  args = parse_args()
  paths = plot_results(args.results, args.output_stem)
  for path in paths:
    print(path)
