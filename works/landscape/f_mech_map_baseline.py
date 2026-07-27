from typing import List, Sequence
import argparse
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec

from smp_bindings import RawSimulationRecord
from utils.plot import (
    plt_figure,
    setup_paper_params,
    plot_network_snapshot,
    plot_opinion_colorbar,
)
import works.config as cfg
from works.landscape.config import SCENARIOS
from works.landscape.plot_utils import save_landscape_figure
from works.landscape.probe_results import (
    LANDSCAPE_CHOICES,
    LandscapeKind,
    plot_probe_potential,
)

# Import reusable functions from f_supp_mech_map
from works.landscape.f_supp_mech_map import (
    ParsedRecord,
    get_basic_data,
    compute_segmented_potentials,
    eval_rec_dx_r_map,
    eval_rec_dx_n_map,
    plot_potential_segments,
    set_ax_format,
    plot_colorbar,
)


def create_subplot_layout(total_width=12, height_scale=1.0):
  """
  Create a figure with 2 rows and 6 columns of subplots.
  Left 4 columns are square, right 2 columns have 4:3 aspect ratio.

  Parameters:
  -----------
  total_width : float
      Total width of the figure in inches

  Returns:
  --------
  fig : matplotlib.figure.Figure
      The figure object
  axes : list
      List of subplot axes (12 subplots total)
  """
  # Calculate dimensions
  # For square subplots: width = height = h
  # For 4:3 subplots: width = 4h/3
  # Total width: 4h + 2(4h/3) = 20h/3
  height_per_row = 3 * total_width / 20
  total_height = 2 * height_per_row * height_scale

  # Create figure
  fig = plt.figure(figsize=(total_width, total_height))

  # GridSpec with width ratios [3, 3, 3, 3, 4, 4] for proper aspect ratios
  gs = GridSpec(
      1,
      2,
      figure=fig,
      width_ratios=[12, 8],
      wspace=0.05,
      hspace=0,
  )
  gs_left = gs[0].subgridspec(
      2,
      4,
      hspace=0.2,
      wspace=0.1,
  )
  gs_right = gs[1].subgridspec(
      2,
      2,
      hspace=0.4,
      wspace=0.6,
  )

  # Create subplots
  axes_r1: List[Axes] = []
  axes_r2: List[Axes] = []
  for j in range(4):
    ax1 = fig.add_subplot(gs_left[0, j])
    axes_r1.append(ax1)
    ax2 = fig.add_subplot(gs_left[1, j])
    axes_r2.append(ax2)

  for j in range(2):
    ax1 = fig.add_subplot(gs_right[0, j])
    axes_r1.append(ax1)
    ax2 = fig.add_subplot(gs_right[1, j])
    axes_r2.append(ax2)

  return fig, [axes_r1, axes_r2]


def plot_network_snapshots(
    rec: RawSimulationRecord,
    axes: Sequence[Axes],
    *,
    active_step: int,
    title_prefix="",
):
  """Plot network snapshots at different time points (row 1)"""
  # Calculate snapshot time steps
  snapshot_times_normalized = [0, 0.1, 0.25, 0.5]
  snapshot_times = [int(t * active_step) for t in snapshot_times_normalized]

  # Get initial graph and compute spring layout
  G0 = rec.get_graph(0)
  poses = {}
  pos = nx.spring_layout(G0, seed=42)
  poses[0] = pos

  # Iteratively refine layout for consistency across snapshots
  for t in snapshot_times[1:]:
    G = rec.get_graph(t)
    pos = nx.spring_layout(G, pos=pos, iterations=10, seed=42)
    poses[t] = pos

  # Plot network snapshots
  for i, t in enumerate(snapshot_times):
    G = rec.get_graph(t)
    opinions = rec.opinions[t]
    plot_network_snapshot(
        poses[t],
        opinions,
        G,
        ax=axes[i],
        colorbar=False,
        node_size=25,
        alpha=0.2,
        rasterized=True,
    )
    t_n = snapshot_times_normalized[i]
    axes[i].set_title(f"({title_prefix}{i+1}) $t_n={t_n}$", loc="left")


def plot_dynamics_maps(
    rec_parsed: ParsedRecord,
    axes: Sequence[Axes],
    scenario_key: str,
    landscape: LandscapeKind,
    use_neighbor_diff=False,
    title_prefix="",
    xlabel=True,
):
  """Plot dynamics maps and potentials (row 2)"""
  ax_x, ax_v_x = axes

  # Keep the original all-agent trajectories for either potential estimator.
  set_ax_format(
      ax_x, xlabel=xlabel, ylabel=True, dt="F" if use_neighbor_diff else "N"
  )
  if use_neighbor_diff:
    eval_rec_dx_n_map(ax_x, rec_parsed)
  else:
    eval_rec_dx_r_map(ax_x, rec_parsed)

  if landscape == "observational":
    x_grid_x, _, V_segments_x = compute_segmented_potentials(
        rec_parsed,
        use_neighbor_diff=use_neighbor_diff,
        normalize_to_zero=True,
        n_segments=10,
    )
    plot_potential_segments(
        ax_v_x,
        x_grid_x,
        V_segments_x,
        xlabel=xlabel,
        ylabel=True,
    )
  else:
    plot_probe_potential(
        ax_v_x,
        scenario_key,
        neighbor_only=use_neighbor_diff,
        xlabel=xlabel,
        ylabel=True,
    )

  # Set titles
  ax_x.set_title(
      f"({title_prefix}) {'NOD' if not use_neighbor_diff else 'FOD'}", loc="left"
  )
  ax_v_x.set_title(
      f"({title_prefix}') Potential of ({title_prefix})", loc="left")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--version",
      type=int,
      choices=(0, 1),
      default=0,
      help="0: baseline NOD/FOD; 1: PbS/SbP comparison (default: 0)",
  )
  parser.add_argument(
      "--landscape",
      choices=LANDSCAPE_CHOICES,
      default="observational",
      help="potential landscape estimator (default: observational)",
  )
  return parser.parse_args()

if __name__ == "__main__":
  args = parse_args()
  setup_paper_params()

  if args.version == 0:
    fig, axes = plt_figure(n_row=2, n_col=4, total_width=14)
    axes_r1, axes_r2 = axes
    ax_x, ax_v_x, ax_dx, ax_v_dx = axes_r2

    rec_baseline = RawSimulationRecord(
        cfg.get_workspace_dir(name="local_mech"), cfg.all_scenarios_mech[0]
    )
    with rec_baseline:
      rec_parsed = get_basic_data(rec_baseline)
      plot_network_snapshots(
          rec_baseline,
          axes_r1,
          active_step=SCENARIOS["baseline"].active_step,
          title_prefix="a",
      )
      plot_dynamics_maps(
          rec_parsed,
          [ax_x, ax_v_x],
          "baseline",
          args.landscape,
          use_neighbor_diff=False,
          title_prefix="b",
      )
      plot_dynamics_maps(
          rec_parsed,
          [ax_dx, ax_v_dx],
          "baseline",
          args.landscape,
          use_neighbor_diff=True,
          title_prefix="c",
      )
    plot_opinion_colorbar(fig, list(axes_r1), label="$x$")
    plot_colorbar(fig, list(axes_r2))

    save_landscape_figure(fig, "fig/landscape/f_mech_map_baseline")

    exit(0)

  # Load baseline scenario (first record)
  rec_baseline_pbs = RawSimulationRecord(
      cfg.get_workspace_dir(name="local_mech"), cfg.all_scenarios_mech[1]
  )
  rec_baseline_sbp = RawSimulationRecord(
      cfg.get_workspace_dir(name="local_mech"), cfg.all_scenarios_mech[2]
  )

  # Create figure and axes
  fig, axes = create_subplot_layout(total_width=16, height_scale=0.85)
  axes_r1, axes_r2 = axes

  # Process data within context
  with rec_baseline_pbs:
    rec_parsed = get_basic_data(rec_baseline_pbs)
    plot_network_snapshots(
        rec_baseline_pbs,
        axes_r1[:4],
        active_step=SCENARIOS["pbs"].active_step,
        title_prefix="a",
    )
    plot_dynamics_maps(
        rec_parsed,
        axes_r1[4:],
        "pbs",
        args.landscape,
        use_neighbor_diff=False,
        title_prefix="b",
        xlabel=False,
    )
  with rec_baseline_sbp:
    rec_parsed = get_basic_data(rec_baseline_sbp)
    plot_network_snapshots(
        rec_baseline_sbp,
        axes_r2[:4],
        active_step=SCENARIOS["sbp"].active_step,
        title_prefix="c",
    )
    plot_dynamics_maps(
        rec_parsed,
        axes_r2[4:],
        "sbp",
        args.landscape,
        use_neighbor_diff=False,
        title_prefix="d",
    )

  for ax in axes_r1[4:] + axes_r2[4:]:
    ax.tick_params(axis="y", pad=0)

  # Add colorbars with unified style
  plot_opinion_colorbar(fig, axes_r1[:4] + axes_r2[:4], label="$x$")
  plot_colorbar(fig, axes_r1[4:] + axes_r2[4:], pad=0.03)

  save_landscape_figure(fig, "fig/landscape/f_mech_map_baseline_cmp")
