from typing import List, Callable, Tuple, Any

import os

import numpy as np
from matplotlib.axes import Axes

from scipy.stats import gaussian_kde

from ehk.common.plotting import plt_figure, plt_save_and_close, setup_paper_params
from ehk.common.numerics import adaptive_moving_stats
from ehk.common.storage.sqlite import create_db_engine_and_session
from experiments.paper.paths import STAT_INPUT, MECHANISM_OUTPUT
from experiments.paper.analysis.records import ScenarioStatistics


plot_path = str(STAT_INPUT.resolve())
stats_db_path = os.path.join(plot_path, "stats.db")

setup_paper_params()


def kde_min_bw_factory(min_bandwidth):
  def min_bw_factor(kde_obj):
    default_factor = kde_obj.scotts_factor()
    min_factor = min_bandwidth / np.std(kde_obj.dataset, ddof=1)
    return max(default_factor, min_factor)

  return min_bw_factor


def fix_margin_stat(arr: np.ndarray, x_left, x_right, x_mid):
  arr_left = arr[x_left][::-1]
  arr_right = arr[x_right][::-1]
  arr_mid = arr[x_mid]

  arr_mid[: arr_left.size] += arr_left
  arr_mid[-arr_right.size:] += arr_right

  return arr_mid


def draw_prob(
    filter_1: np.ndarray,
    filter_2: np.ndarray,
    filter_3: np.ndarray,
    total_size: int,
    ax: Axes,
    stats_grad: np.ndarray,
    x_axis: np.ndarray,
    x: np.ndarray,
    x_left: np.ndarray,
    x_right: np.ndarray,
    x_mid: np.ndarray,
):
  bw_method = kde_min_bw_factory(0.025)

  def _f(filter: np.ndarray):
    data_x = stats_grad[filter]
    if data_x.size == 0:
      return x_axis * 0
    return fix_margin_stat(
        gaussian_kde(data_x, bw_method=bw_method)(x), x_left, x_right, x_mid
    )

  div_ratio_1 = np.sum(filter_1) / total_size
  div_ratio_2 = np.sum(filter_2) / total_size
  div_ratio_3 = np.sum(filter_3) / total_size

  div_group_1 = _f(filter_1) * div_ratio_1
  div_group_2 = _f(filter_2) * div_ratio_2
  div_group_3 = _f(filter_3) * div_ratio_3

  ax.fill_between(x_axis, x_axis * 0, div_group_1, label="#peaks = 1")
  ax.fill_between(x_axis, div_group_1, div_group_1 +
                  div_group_2, label="#peaks = 2")
  ax.fill_between(
      x_axis,
      div_group_1 + div_group_2,
      div_group_1 + div_group_2 + div_group_3,
      label="#peaks >= 3",
  )


def plot_bar(
    ax: Axes,
    all_data: List[List[Tuple[float, float]]] | np.ndarray,
    k_labels: List[str],
    m_labels: List[str],
    m_colors: List[Any],
):
  all_data_arr = np.array(all_data, dtype=float)  # (k, m, [mean, std])
  k, m, _ = all_data_arr.shape
  means = all_data_arr[:, :, 0]
  stds = all_data_arr[:, :, 1]

  width = 0.6 / m  # width of the bars
  x = np.arange(k)

  for i in range(m):
    ax.bar(
        x + i * width,
        means[:, i],
        width,
        yerr=stds[:, i],
        label=m_labels[i],
        capsize=4,
        edgecolor="black",
        color=m_colors[i],
        alpha=0.5,
    )

  ax.set_xticks(x + width * (m - 1) / 2)
  ax.set_xticklabels(k_labels)

  return ax


if __name__ == "__main__":

  engine, session = create_db_engine_and_session(
      stats_db_path, ScenarioStatistics.Base
  )

  # Load cluster data.
  stats_f1 = session.query(ScenarioStatistics).filter(
      ScenarioStatistics.grad_index.is_not(None),
      ScenarioStatistics.last_opinion_peak_count.is_not(None),
  )

  stats_grad, stats_peak, stats_retweet = np.array(
      [(x.grad_index, x.last_opinion_peak_count, x.retweet) for x in stats_f1],
      dtype=np.float64,
  ).T

  stats_peak = stats_peak.astype(int)

  # Configure cluster axes.
  resolution = 0.005
  margin = 0.3
  x = np.arange(-margin, 1 + margin + resolution, resolution)
  x_left = x < -resolution * 0.5
  x_right = x > 1 + resolution * 0.5
  x_mid = np.logical_not(np.logical_or(x_left, x_right))
  x_axis = x[x_mid]

  # Load repost data.
  stats_raw = session.query(ScenarioStatistics).filter(
      ScenarioStatistics.grad_index.is_not(None),
      ScenarioStatistics.opinion_diff_seg_mean.is_not(None),
      ScenarioStatistics.opinion_diff_seg_std.is_not(None),
  )

  def _filter(f: Callable[[ScenarioStatistics], bool]):
    stats_filtered = [x for x in stats_raw if f(x)]

    raw_mean, raw_std = np.array(
        [(x.opinion_diff_seg_mean, x.opinion_diff_seg_std)
         for x in stats_filtered]
    ).transpose(1, 0, -1)
    opinion_diff_mean = np.mean(raw_mean, axis=0)

    std_term_1 = raw_std**2
    std_term_2 = (raw_mean - opinion_diff_mean) ** 2

    var_term = np.mean((std_term_1 + std_term_2), axis=0)
    opinion_diff_std = var_term**0.5

    return opinion_diff_mean, opinion_diff_std

  od_p2_r_mean, od_p2_r_std = _filter(
      lambda x: x.grad_index is not None and x.grad_index < 0.6 and x.retweet != 0
  )
  od_p2_nr_mean, od_p2_nr_std = _filter(
      lambda x: x.grad_index is not None and x.grad_index < 0.6 and x.retweet == 0
  )
  od_p1_r_mean, od_p1_r_std = _filter(
      lambda x: x.grad_index is not None and x.grad_index >= 0.6 and x.retweet != 0
  )
  od_p1_nr_mean, od_p1_nr_std = _filter(
      lambda x: x.grad_index is not None and x.grad_index >= 0.6 and x.retweet == 0
  )

  # Compare repost conditions.
  x1, y1, y_std_1 = np.array(
      [
          (x.grad_index, x.opinion_diff_seg_mean[1], x.opinion_diff_seg_std[1])
          for x in stats_raw
          if x.retweet == 0
          and x.opinion_diff_seg_mean is not None
          and x.opinion_diff_seg_std is not None
      ]
  ).T
  x1_f, y1_f, _ = adaptive_moving_stats(x1, y1, 0.2, min=0, max=1)
  _, y1_std_f, _ = adaptive_moving_stats(x1, y_std_1, 0.2, min=0, max=1)

  x2, y2, y_std_2 = np.array(
      [
          (x.grad_index, x.opinion_diff_seg_mean[1], x.opinion_diff_seg_std[1])
          for x in stats_raw
          if x.retweet != 0
          and x.opinion_diff_seg_mean is not None
          and x.opinion_diff_seg_std is not None
      ]
  ).T
  x2_f, y2_f, _ = adaptive_moving_stats(x2, y2, 0.2)
  _, y2_std_f, _ = adaptive_moving_stats(x2, y_std_2, 0.2, min=0, max=1)

  # Combine both studies in a two-by-three layout.
  fig, axes = plt_figure(n_row=2, n_col=3)

  # First row: cluster panels.
  ax1, ax2, ax3 = axes[0]

  draw_prob(
      stats_peak == 1,
      stats_peak == 2,
      stats_peak > 2,
      stats_peak.size,
      ax1,
      stats_grad,
      x_axis,
      x,
      x_left,
      x_right,
      x_mid,
  )
  draw_prob(
      np.logical_and(stats_peak == 1, stats_retweet == 0),
      np.logical_and(stats_peak == 2, stats_retweet == 0),
      np.logical_and(stats_peak > 2, stats_retweet == 0),
      np.sum(stats_retweet == 0),
      ax2,
      stats_grad,
      x_axis,
      x,
      x_left,
      x_right,
      x_mid,
  )
  draw_prob(
      np.logical_and(stats_peak == 1, stats_retweet != 0),
      np.logical_and(stats_peak == 2, stats_retweet != 0),
      np.logical_and(stats_peak > 2, stats_retweet != 0),
      np.sum(stats_retweet != 0),
      ax3,
      stats_grad,
      x_axis,
      x,
      x_left,
      x_right,
      x_mid,
  )

  for ax in [ax1, ax2, ax3]:
    ax.grid(True)
    ax.legend()
    ax.set_xlabel(r"$I_w$")

  ax1.set_ylabel(r"P(#peaks)")
  ax1.set_title("(a) gross", loc="left")
  ax2.set_title("(b) no retweet", loc="left")
  ax3.set_title("(c) with retweet", loc="left")

  # Second row: repost panels.
  ax4, ax5, ax6 = axes[1]

  x_labels = ["0~.2", ".2~.4", ".4~.6", ".6~.8", ".8~1"]

  plot_bar(
      ax4,
      np.array(
          [[od_p2_nr_mean, od_p2_nr_std], [od_p2_r_mean, od_p2_r_std]]
      ).transpose(2, 0, 1),
      x_labels,
      ["no retweet", "retweet"],
      ["tab:blue", "tab:cyan"],
  )

  plot_bar(
      ax5,
      np.array(
          [[od_p1_nr_mean, od_p1_nr_std], [od_p1_r_mean, od_p1_r_std]]
      ).transpose(2, 0, 1),
      x_labels,
      ["no retweet", "retweet"],
      ["tab:red", "tab:orange"],
  )

  ax6.plot(x1_f, y1_f, label="no retweet")
  ax6.fill_between(x1_f, y1_f - y1_std_f, y1_f + y1_std_f, alpha=0.1)
  ax6.plot(x2_f, y2_f, label="retweet")
  ax6.fill_between(x2_f, y2_f - y2_std_f, y2_f + y2_std_f, alpha=0.1)

  for ax in [ax4, ax5, ax6]:
    ax.grid(True)
    ax.legend()

  for ax in [ax4, ax5]:
    ax.set_xlabel(r"$t_n$")
  ax6.set_xlabel(r"$I_w$")

  ax4.set_ylabel("mean opinion difference")

  ax4.set_title("(d) PbS", loc="left")
  ax5.set_title("(e) SbP", loc="left")
  ax6.set_title(r"(f) mean op. diff. ($t_n \in [0.2, 0.4)$)", loc="left")

  ax4.set_ylim(0, 0.6)
  ax5.set_ylim(0, 0.5)

  figure_dir = MECHANISM_OUTPUT.resolve() / "figures"
  figure_dir.mkdir(parents=True, exist_ok=True)
  plt_save_and_close(fig, str(figure_dir / "f_cluster_and_retweet"))

  session.close()
  engine.dispose()
