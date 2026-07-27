from typing import Tuple, List, Dict
import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.interpolate import interp1d
from dataclasses import dataclass, asdict

from smp_bindings import RawSimulationRecord
from utils.plot import plt_figure, setup_paper_params
from stats import estimate_force_field_kde
import works.config as cfg
from works.landscape.plot_utils import (
    CMAP_NAME,
    format_landscape_axis,
    plot_normalized_time_colorbar,
    plot_potential_curves,
    potential_from_force,
    save_landscape_figure,
)
from works.landscape.probe_results import (
    LANDSCAPE_CHOICES,
    LandscapeKind,
    plot_probe_potential,
)
from works.stat.context import c

CACHE_FILE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))
                    ), "basic_data_cache.pkl"
)
PLOT_RES = 100
EXTRAPOLATE_FILL: float = "extrapolate"  # type: ignore


@dataclass
class ParsedRecord:
  t_seq: np.ndarray
  x_seq: np.ndarray
  active_step: int
  dx_n_seq: np.ndarray
  dx_r_seq: np.ndarray
  dx_nr_seq: np.ndarray
  kn_seq: np.ndarray
  kr_seq: np.ndarray
  knr_seq: np.ndarray

# region Data Loading and Caching


def load_cache() -> dict:
  if os.path.exists(CACHE_FILE_PATH):
    try:
      with open(CACHE_FILE_PATH, "rb") as f:
        return pickle.load(f)
    except Exception as e:
      print(f"Warning: Failed to load cache: {e}")
  return {}


def save_cache(cache: dict):
  try:
    with open(CACHE_FILE_PATH, "wb") as f:
      pickle.dump(cache, f)
  except Exception as e:
    print(f"Warning: Failed to save cache: {e}")


def get_basic_data(rec: RawSimulationRecord, force_reload=False) -> ParsedRecord:
  if globals().get("basic_data_cache", None) is None:
    globals()["basic_data_cache"] = load_cache()

  cache: Dict[str, dict] = globals()["basic_data_cache"]
  rec_key = rec.unique_name

  if rec_key in cache and not force_reload:
    cached = cache[rec_key]
    try:
      return ParsedRecord(**cached)
    except Exception as e:
      print(f"Warning: Failed to parse cached data for {rec_key}: {e}")
      del cache[rec_key]

  c.set_state(active_threshold=0.98,
              min_inactive_value=0.75, scenario_record=rec)
  c.debug = True

  active_step = c.active_step
  t_seq = np.arange(rec.opinions.shape[0]) / active_step
  x_seq = rec.opinions[:, :]

  kn_seq = rec.agent_numbers[:, :, 0]
  kn_seq_l_1 = np.maximum(kn_seq, 1)
  kr_seq = rec.agent_numbers[:, :, 1]
  kr_seq_l_1 = np.maximum(kr_seq, 1)
  knr_seq = kn_seq + kr_seq
  knr_seq_l_1 = np.maximum(knr_seq, 1)

  dx_n_seq = rec.agent_opinion_sums[:, :, 0] / kn_seq_l_1
  dx_r_seq = rec.agent_opinion_sums[:, :, 1] / kr_seq_l_1
  dx_nr_seq = (
      rec.agent_opinion_sums[:, :, 0] +
      rec.agent_opinion_sums[:, :, 1]
  ) / knr_seq_l_1

  print(rec_key, rec.max_step, active_step)

  record = ParsedRecord(
      t_seq=t_seq,
      x_seq=x_seq,
      active_step=active_step,
      dx_n_seq=dx_n_seq,
      dx_r_seq=dx_r_seq,
      dx_nr_seq=dx_nr_seq,
      kn_seq=kn_seq,
      kr_seq=kr_seq,
      knr_seq=knr_seq,
  )

  cache[rec_key] = asdict(record)
  save_cache(cache)

  return record


# endregion

# region Data Processing


def resample_sequence(seq: np.ndarray, num_points: int) -> np.ndarray:
  xx = np.linspace(0, seq.shape[0] - 1, num=num_points)
  return interp1d(
      np.arange(seq.shape[0]), seq, axis=0, kind="linear", fill_value=EXTRAPOLATE_FILL
  )(xx)


def compute_differentials(
    r: ParsedRecord,
    differential_sequence: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  x_seq_resampled = resample_sequence(r.x_seq, PLOT_RES)
  t_seq_resampled = resample_sequence(r.t_seq, PLOT_RES)
  dx_seq_resampled = resample_sequence(differential_sequence, PLOT_RES)
  return t_seq_resampled, x_seq_resampled, dx_seq_resampled


def compute_trajectory_differentials(
    r: ParsedRecord,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  return compute_differentials(r, r.dx_nr_seq)


def compute_neighbor_differentials(
    r: ParsedRecord,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  return compute_differentials(r, r.dx_n_seq)


def seq_to_map(seq: np.ndarray, diff=False) -> np.ndarray:
  map_raw = np.array(
      [seq[:-1, :], np.diff(seq, axis=0) if diff else seq[1:, :]])
  return map_raw.transpose((2, 1, 0))


def k_points_to_map(k_points: np.ndarray, max_val=15) -> np.ndarray:
  cnt_mat = np.zeros((max_val + 1, max_val + 1), dtype=int)
  for x in range(max_val + 1):
    for y in range(max_val + 1):
      cnt_mat[y, x] = np.sum((k_points[:, 0] == x) & (k_points[:, 1] == y))
  return cnt_mat


def compute_segmented_potentials(
    rec_parsed: ParsedRecord,
    n_segments: int = 5,
    x_min: float = -1.0,
    x_max: float = 1.0,
    n_grid: int = 200,
    h: float = 0.1,
    k: float = 1.0,
    use_neighbor_diff: bool = False,
    normalize_to_zero: bool = False,
) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:

  if use_neighbor_diff:
    t_seq, x_seq, dx_seq = compute_neighbor_differentials(rec_parsed)
    dx_data, x_data = dx_seq[1:, :], x_seq[:-1, :]
  else:
    t_seq, x_seq, dx_seq = compute_trajectory_differentials(rec_parsed)
    # dx_data, x_data = dx_seq, x_seq[:-1, :]
    dx_data, x_data = dx_seq[1:, :], x_seq[:-1, :]

  t_normalized = np.clip(t_seq[:-1], 0, 1)
  x_grid = np.linspace(x_min, x_max, n_grid)
  segment_edges = np.linspace(0, 1, n_segments + 1)

  F_segments, V_segments = [], []

  for i in range(n_segments):
    t_start, t_end = segment_edges[i], segment_edges[i + 1]
    mask = (t_normalized >= t_start) & (
        t_normalized <= t_end if i == n_segments - 1 else t_normalized < t_end
    )

    x_segment = x_data[mask, :].flatten()
    dx_segment = dx_data[mask, :].flatten()

    if len(x_segment) == 0:
      F_segments.append(np.zeros_like(x_grid))
      V_segments.append(np.zeros_like(x_grid))
      continue

    F_grid = estimate_force_field_kde(x_segment, dx_segment, x_grid, h, k)
    V_grid = potential_from_force(
        x_grid,
        F_grid,
        center=normalize_to_zero,
    )
    F_segments.append(F_grid)
    V_segments.append(V_grid)

  return x_grid, F_segments, V_segments


# endregion

# region Plotting


def plot_colorbar(fig: Figure, axes: List[Axes], pad=0.01):
  return plot_normalized_time_colorbar(fig, axes, pad=pad)


def set_ax_format(ax: Axes, dt: float | str = 0.02, ylim=0.4, xlabel=True, ylabel=True):
  format_landscape_axis(
      ax,
      xlabel=r"$x_i(t)$" if xlabel else None,
      ylabel=rf"$\Delta_{{{dt}}} x_i(t)$" if ylabel else None,
      ylim=(-ylim, ylim),
  )


def plot_trajectory_map(ax: Axes, x_map: np.ndarray, t_seq: np.ndarray):
  cmap = plt.get_cmap(CMAP_NAME)
  for time_idx in range(x_map.shape[1] - 1):
    color = cmap(t_seq[time_idx])
    alpha = min(0.1 + t_seq[time_idx] * 0.5, 1)
    ax.plot(
        x_map[:, time_idx: time_idx + 2, 0].T,
        x_map[:, time_idx: time_idx + 2, 1].T,
        color=color,
        linewidth=0.5,
        alpha=alpha,
        rasterized=True,
    )


def eval_rec_k_map(ax_k: Axes, r: ParsedRecord):
  k_points = seq_to_map(r.kn_seq).reshape(-1, 2)
  k_points = np.clip(k_points, 0, 15)
  k_map = np.log1p(k_points_to_map(k_points, max_val=15))
  ax_k.imshow(k_map, origin="lower", cmap="YlGnBu")


def eval_rec_dx_r_map(ax_x: Axes, rec_parsed: ParsedRecord):
  t_seq_resampled, x_seq_resampled, dx_r_seq_resampled = compute_trajectory_differentials(
      rec_parsed)
  # x_map = seq_to_map(x_seq_resampled, diff=True)
  dx_r_map = np.array([x_seq_resampled[:-1], dx_r_seq_resampled[1:]]).transpose(
      (2, 1, 0)
  )
  plot_trajectory_map(ax_x, dx_r_map, t_seq_resampled)


def eval_rec_dx_n_map(ax_dx: Axes, rec_parsed: ParsedRecord):
  t_seq_resampled, x_seq_resampled, dx_n_seq_resampled = (
      compute_neighbor_differentials(rec_parsed)
  )
  dx_n_map = np.array([x_seq_resampled[:-1], dx_n_seq_resampled[1:]]).transpose(
      (2, 1, 0)
  )
  plot_trajectory_map(ax_dx, dx_n_map, t_seq_resampled)


def plot_potential_segments(
    ax: Axes, x_grid: np.ndarray, V_segments: List[np.ndarray], xlabel=True, ylabel=True
):
  n_segments = len(V_segments)
  plot_potential_curves(
      ax,
      x_grid,
      V_segments,
      (np.arange(n_segments) + 0.5) / n_segments,
      xlabel=xlabel,
      ylabel=ylabel,
  )


def plot_group(
    recs: List[RawSimulationRecord],
    scenario_keys: List[str],
    n_col: int,
    titles: List[str],
    save_name: str,
    *,
    landscape: LandscapeKind,
):
  fig, axes_all = plt_figure(n_row=4, n_col=n_col, total_width=n_col * 3)
  axes_r1, axes_r1_v, axes_r2, axes_r2_v = axes_all

  # Collect all potential values to determine unified y-axis ranges
  all_V_x = []
  all_V_dx = []

  for i, rec in enumerate(recs):
    set_ax_format(axes_r1[i], xlabel=False, ylabel=(
        i == 0), dt="n")  # 1 / PLOT_RES
    set_ax_format(axes_r2[i], xlabel=False, ylabel=(i == 0), dt="f")  # "N"

    with rec:
      rec_parsed = get_basic_data(rec)

    # The dynamics panels always retain the original all-agent trajectories.
    eval_rec_dx_r_map(axes_r1[i], rec_parsed)
    eval_rec_dx_n_map(axes_r2[i], rec_parsed)

    if landscape == "observational":
      x_grid_x, _, V_segments_x = compute_segmented_potentials(
          rec_parsed, use_neighbor_diff=False, normalize_to_zero=True,
          n_segments=10,
      )
      x_grid_dx, _, V_segments_dx = compute_segmented_potentials(
          rec_parsed, use_neighbor_diff=True, normalize_to_zero=True,
          n_segments=10,
      )
      plot_potential_segments(
          axes_r1_v[i],
          x_grid_x,
          V_segments_x,
          xlabel=False,
          ylabel=(i == 0),
      )
      plot_potential_segments(
          axes_r2_v[i],
          x_grid_dx,
          V_segments_dx,
          xlabel=True,
          ylabel=(i == 0),
      )
    else:
      V_segments_x = plot_probe_potential(
          axes_r1_v[i],
          scenario_keys[i],
          neighbor_only=False,
          xlabel=False,
          ylabel=(i == 0),
      )
      V_segments_dx = plot_probe_potential(
          axes_r2_v[i],
          scenario_keys[i],
          neighbor_only=True,
          xlabel=True,
          ylabel=(i == 0),
      )

    # Collect potential values
    for V in V_segments_x:
      all_V_x.extend(V)
    for V in V_segments_dx:
      all_V_dx.extend(V)

    char = chr(ord("a") + i)
    title = f" {titles[i]}" if i < len(titles) else ""
    axes_r1[i].set_title(f"({char}1){title}", loc="left")
    axes_r1_v[i].set_title(f"({char}1')", loc="left")
    axes_r2[i].set_title(f"({char}2)", loc="left")
    axes_r2_v[i].set_title(f"({char}2')", loc="left")

  # Set unified y-axis ranges for potential plots
  if all_V_x:
    v_min_x, v_max_x = np.min(all_V_x), np.max(all_V_x)
    v_range_x = v_max_x - v_min_x
    y_margin_x = v_range_x * 0.1  # 10% margin
    for ax in axes_r1_v:
      ax.set_ylim(v_min_x - y_margin_x, v_max_x + y_margin_x)

  if all_V_dx:
    v_min_dx, v_max_dx = np.min(all_V_dx), np.max(all_V_dx)
    v_range_dx = v_max_dx - v_min_dx
    y_margin_dx = v_range_dx * 0.1  # 10% margin
    for ax in axes_r2_v:
      ax.set_ylim(v_min_dx - y_margin_dx, v_max_dx + y_margin_dx)

  plot_colorbar(fig, [*axes_r1, *axes_r1_v, *axes_r2, *axes_r2_v])
  save_landscape_figure(fig, save_name)


# endregion


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
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

  recs = [
      RawSimulationRecord(cfg.get_workspace_dir(name="local_mech"), d)
      for d in cfg.all_scenarios_mech
  ]

  plot_group(
      recs[:1] + recs[3:6],
      ["baseline", "influence", "retweet", "op_recsys"],
      4,
      ["baseline", "+influence", "+retweet", "opinion rec."],
      "fig/landscape/f_supp_mech_map_g1",
      landscape=args.landscape,
  )
  plot_group(
      recs[6:],
      ["phase1", "phase2", "phase3", "phase4", "phase5"],
      5,
      [],
      "fig/landscape/f_supp_mech_map_g2",
      landscape=args.landscape,
  )
