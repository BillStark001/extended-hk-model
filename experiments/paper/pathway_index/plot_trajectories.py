from typing import Dict, Iterable, List, Tuple

import os

import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

from ehk.common.plotting import plt_figure, plt_save_and_close, setup_paper_params
from ehk.common.storage.sqlite import create_db_engine_and_session
import experiments.paper.scenarios as cfg
from experiments.paper.paths import STAT_INPUT, PATHWAY_OUTPUT
from experiments.paper.analysis.records import ScenarioStatistics

from sklearn.cluster import KMeans


plot_path = str(STAT_INPUT.resolve())
stats_db_path = os.path.join(plot_path, "stats.db")

retweet_groups: List[float] = cfg.retweet_rate_array.tolist()
rs_groups: List[Tuple[str, int]] = [
    (rs, int(retain)) for rs, retain in cfg.rs_names.values()
]

HEATMAP_BINS = 128
SUMMARY_CURVE_SAMPLES = 256
SUMMARY_CURVE_QUANTILES = 16


def _safe_curve(
    p_index: np.ndarray | None,
    h_index: np.ndarray | None,
) -> np.ndarray | None:
  if p_index is None or h_index is None:
    return None
  if p_index.size < 2 or h_index.size < 2:
    return None

  n = min(p_index.size, h_index.size)
  p = p_index[:n]
  h = h_index[:n]

  mask = np.isfinite(p) & np.isfinite(h)
  if np.count_nonzero(mask) < 2:
    return None

  p_valid = np.minimum(np.maximum(p[mask], 0.0), 1.0)
  h_valid = np.minimum(np.maximum(h[mask], 0.0), 1.0)
  points = np.column_stack((p_valid, h_valid))
  if points.shape[0] < 2:
    return None
  return points


def _group_data(
    full_data: Iterable[ScenarioStatistics],
) -> Dict[Tuple[float, str, int], List[Tuple[np.ndarray, float]]]:
  grouped: Dict[Tuple[float, str, int], List[Tuple[np.ndarray, float]]] = {}
  for rt in retweet_groups:
    for rs_type, retain_count in rs_groups:
      grouped[(rt, rs_type, retain_count)] = []

  for datum in full_data:
    if datum.grad_index is None:
      continue
    if datum.p_index is None or datum.h_index is None:
      continue

    retain_count = int(round(float(datum.tweet_retain_count)))
    key = (float(datum.retweet), str(datum.recsys_type), retain_count)
    if key not in grouped:
      continue

    points = _safe_curve(datum.p_index, datum.h_index)
    if points is None:
      continue

    grad = min(max(float(datum.grad_index), 0.0), 1.0)
    grouped[key].append((points, grad))

  return grouped


def _density_image(
    curves: List[Tuple[np.ndarray, float]],
    cmap,
    norm: Normalize,
    bins: int = HEATMAP_BINS,
) -> np.ndarray | None:
  if not curves:
    return None

  counts = np.zeros((bins, bins), dtype=float)
  grad_sums = np.zeros((bins, bins), dtype=float)

  for points, grad in curves:
    hist, x_edges, y_edges = np.histogram2d(
        points[:, 0],
        points[:, 1],
        bins=bins,
        range=((0.0, 1.0), (0.0, 1.0)),
    )
    weighted_hist, _, _ = np.histogram2d(
        points[:, 0],
        points[:, 1],
        bins=bins,
        range=((0.0, 1.0), (0.0, 1.0)),
        weights=np.full(points.shape[0], grad, dtype=float),
    )
    counts += hist
    grad_sums += weighted_hist

  if counts.max() <= 0:
    return None

  mean_grad = np.full_like(counts, 0.5)
  valid = counts > 0
  mean_grad[valid] = grad_sums[valid] / counts[valid]

  density = np.log1p(counts)
  density /= density.max()
  density = density.T
  mean_grad = mean_grad.T

  rgba = cmap(norm(mean_grad))
  rgba[..., :3] = 1.0 - (1.0 - rgba[..., :3]) * density[..., None]
  rgba[..., 3] = np.where(density > 0, 0.95, 0.0)
  return rgba


def resample_curve(points: np.ndarray, n_samples: int = SUMMARY_CURVE_SAMPLES) -> np.ndarray:
  if points.shape[0] == n_samples:
    return points

  src = np.linspace(0.0, 1.0, points.shape[0])
  dst = np.linspace(0.0, 1.0, n_samples)
  return np.column_stack((
      np.interp(dst, src, points[:, 0]),
      np.interp(dst, src, points[:, 1]),
  ))


def summary_curves_clustered(
    curves: List[Tuple[np.ndarray, float]],
    n_groups: int = SUMMARY_CURVE_QUANTILES,
) -> List[Tuple[np.ndarray, float, int]]:
  if not curves:
    return []

  # Resample every curve to a common 512-point coordinate system.
  resampled_curves = [resample_curve(points) for points, _ in curves]
  resampled_array = np.stack(resampled_curves, axis=0)

  n_curves, n_samples, n_dims = resampled_array.shape

  # Flatten coordinates so K-means uses pointwise Euclidean distance.
  features = resampled_array.reshape(n_curves, n_samples * n_dims)

  # Do not request more spatial trajectory clusters than samples.
  n_clusters = min(n_groups, n_curves)
  kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
  labels = kmeans.fit_predict(features)

  # Aggregate each trajectory cluster.
  grads = [grad for _, grad in curves]
  summaries: List[Tuple[np.ndarray, float, int]] = []

  for cluster_idx in range(n_clusters):
    # Select curves in this cluster.
    indices = np.where(labels == cluster_idx)[0]
    if len(indices) == 0:
      continue

    # Collect their resampled coordinates.
    cluster_curves = resampled_array[indices]

    # Use the median curve as the cluster backbone.
    median_curve = np.median(cluster_curves, axis=0)

    # Retain the mean pathway index for scientific color encoding.
    mean_grad = float(np.mean([grads[i] for i in indices]))

    summaries.append((median_curve, mean_grad, len(indices)))

  # Sort by pathway index for stable layering and legends.
  summaries.sort(key=lambda x: x[1])

  return summaries


def plot_p_h_index_trajectories() -> Figure:
  setup_paper_params()

  engine, session = create_db_engine_and_session(
      stats_db_path, ScenarioStatistics.Base
  )

  full_data: Iterable[ScenarioStatistics] = session.query(ScenarioStatistics).filter(
      ScenarioStatistics.name.startswith("s_grad"),
      ScenarioStatistics.p_index.is_not(None),
      ScenarioStatistics.h_index.is_not(None),
      ScenarioStatistics.grad_index.is_not(None),
  )

  grouped = _group_data(full_data)

  fig, axes = plt_figure(
      n_row=4,
      n_col=4,
      hw_ratio=1,
      total_width=13.5,
      constrained_layout=False,
  )
  fig.subplots_adjust(
      left=0.07,
      right=0.89,
      bottom=0.08,
      top=0.96,
      wspace=0.10,
      hspace=0.14,
  )

  axes_arr = np.array(axes)
  cmap = cm.get_cmap("coolwarm")
  norm = Normalize(vmin=0, vmax=1)

  for i_rt, rt in enumerate(retweet_groups):
    for i_rs, (rs_type, retain_count) in enumerate(rs_groups):
      ax = axes_arr[i_rt, i_rs]
      curves = grouped[(rt, rs_type, retain_count)]

      density_img = _density_image(curves, cmap, norm)
      if density_img is not None:
        ax.imshow(
            density_img,
            extent=(0, 1, 0, 1),
            origin="lower",
            interpolation="bilinear",
            aspect="equal",
            zorder=1,
        )

      for points, grad, _ in summary_curves_clustered(curves):
        ax.plot(
            points[:, 0],
            points[:, 1],
            color=cmap(norm(grad)),
            linewidth=1.1,
            alpha=0.95,
            solid_capstyle="round",
            zorder=2,
        )

      eps = 0.05

      ax.set_xlim(0 - eps, 1 + eps)
      ax.set_ylim(0 - eps, 1 + eps)
      ax.set_facecolor("white")
      ax.grid(True, linestyle="--", linewidth=0.5)

      title_rs = "St" if rs_type == "StructureM9" else "Op"
      ax.set_title(
          f"p={rt:g}, {title_rs}/k={retain_count} (n={len(curves)})",
          loc="left",
          fontsize=9,
      )

      if i_rt == 3:
        ax.set_xlabel(r"$I_p$")
      else:
        ax.set_xticklabels([])

      if i_rs == 0:
        ax.set_ylabel(r"$I_h$")
      else:
        ax.set_yticklabels([])

  sm = cm.ScalarMappable(norm=norm, cmap=cmap)
  sm.set_array([])
  cax = fig.add_axes((0.91, 0.17, 0.015, 0.66))
  cbar = fig.colorbar(sm, cax=cax)
  cbar.set_label(r"mean $I_w$")

  session.close()
  engine.dispose()
  return fig


if __name__ == "__main__":
  figure_dir = PATHWAY_OUTPUT.resolve() / "figures"
  figure_dir.mkdir(parents=True, exist_ok=True)
  fig = plot_p_h_index_trajectories()
  plt_save_and_close(
      fig,
      str(figure_dir / "f_p_h_index_trajectory"),
      jpg=True,
  )
