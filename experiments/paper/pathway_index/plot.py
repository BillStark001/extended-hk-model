from typing import Dict, List, Iterable, Callable, TypeVar, Tuple

import os

import numpy as np
from sqlalchemy import ColumnExpressionArgument
from sqlalchemy.orm import Session
from matplotlib.axes import Axes

from ehk.common.plotting import plt_figure, plt_save_and_close, setup_paper_params, PAPER_WIDTH
from ehk.common.storage.sqlite import create_db_engine_and_session
import experiments.paper.scenarios as cfg
from experiments.paper.paths import STAT_INPUT, PATHWAY_OUTPUT
from experiments.paper.pathway_index.data_utils import piecewise_linear_integral_trapz
from experiments.paper.analysis.records import ScenarioStatistics


T = TypeVar("T")
T1 = TypeVar("T1")
T2 = TypeVar("T2")

# parameters

plot_path = str(STAT_INPUT.resolve())
stats_db_path = os.path.join(plot_path, 'stats.db')

rs_keys = list(cfg.rs_names)
tw_keys = cfg.retweet_rate_array.tolist()


setup_paper_params()

# utilities

# build scenario
short_progress_bar = "{l_bar}{bar:10}{r_bar}{bar:-10b}"


bc_inst_orig: List[List[List[ScenarioStatistics]]] = []
rw_rev_dict: Dict[float, int] = {}
dc_rev_dict: Dict[float, int] = {}
for i, rw in enumerate(cfg.rewiring_rate_array):
  rw_rev_dict[rw] = i
  lst = []
  for j, dc in enumerate(cfg.decay_rate_array):
    dc_rev_dict[dc] = j
    lst.append([])
  bc_inst_orig.append(lst)

# First row filters: distinguish by recommendation strategy (St vs Op), ignore retweet status
k_filters_recsys = [
    (
        ScenarioStatistics.recsys_type == 'StructureM9',
    ),
    (
        ScenarioStatistics.recsys_type == 'OpinionM9',
    ),
]

k_labels_recsys = [
    'St.',
    'Op.',
]

# Second row filters: distinguish by retweet status (!R vs R), ignore recommendation strategy
k_filters_retweet = [
    (
        ScenarioStatistics.retweet == 0,
    ),
    (
        ScenarioStatistics.retweet != 0,
    ),
]

k_labels_retweet = [
    '!R',
    'R',
]


m_filters = [
    (
        ScenarioStatistics.grad_index.is_not(None),
        ScenarioStatistics.grad_index < 0.6,
    ),
    (
        ScenarioStatistics.grad_index.is_not(None),
        ScenarioStatistics.grad_index >= 0.6,
    ),
]

m_labels = [
    'PbS',
    'SbP',
]

m_colors = [
    'tab:blue',
    'tab:red'
]


def evaluate_bar_on_filters(
    session: Session,
    filters: List[ColumnExpressionArgument[bool]],
    f_ext: Callable[[ScenarioStatistics], float],
):
  full_data: Iterable[ScenarioStatistics] = session.query(
      ScenarioStatistics).filter(*filters)

  data_arr: List[float] = []
  for datum in full_data:
    data_arr.append(f_ext(datum))

  return np.mean(data_arr), np.std(data_arr)


def plot_bar(
    ax: Axes,
    f: Callable[[ScenarioStatistics], float],
    use_retweet_filters=False,
    plot_title: str = "",
) -> List[str]:

  ret: List[str] = []

  k_filters = k_filters_retweet if use_retweet_filters else k_filters_recsys
  k_labels = k_labels_retweet if use_retweet_filters else k_labels_recsys

  k = len(k_filters)
  m = len(m_filters)

  all_data: List[List[Tuple[float, float]]] = []

  for k_filter in k_filters:
    row_data: List[Tuple[float, float]] = []
    for m_filter in m_filters:
      filters = [*k_filter, *m_filter]
      mean, std = evaluate_bar_on_filters(session, filters, f)
      row_data.append((mean, std))  # type: ignore
    all_data.append(row_data)

  all_data_arr = np.array(all_data, dtype=float)  # (k, m, [mean, std])
  means = all_data_arr[:, :, 0]
  stds = all_data_arr[:, :, 1]

  # Print data for each bar
  if plot_title:
    ret.append(f"\n=== {plot_title} ===")
  for k_idx in range(k):
    for m_idx in range(m):
      ret.append(
          f"{k_labels[k_idx]} - {m_labels[m_idx]}: {means[k_idx, m_idx]:.4f} ± {stds[k_idx, m_idx]:.4f}")

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
        edgecolor='black',
        color=m_colors[i],
        alpha=0.5,
    )

  ax.set_xticks(x + width*(m-1)/2)
  ax.set_xticklabels(k_labels)

  return ret


if __name__ == '__main__':

  engine, session = create_db_engine_and_session(
      stats_db_path, ScenarioStatistics.Base)

  # Create figure with 2 rows, 4 columns for 8 subplots (a-h)
  fig, axes = plt_figure(n_row=2, n_col=4, total_width=PAPER_WIDTH *
                         0.8, hw_ratio=4/3, constrained_layout=False)

  def f_ev_count(x: ScenarioStatistics) -> float:
    assert x.event_count is not None
    return x.event_count

  def f_ev_step(x: ScenarioStatistics) -> float:
    assert x.event_step_mean is not None
    assert x.active_step is not None
    return x.event_step_mean / max(x.active_step, 1)

  def f_auc_subj_pol_index(x: ScenarioStatistics) -> float:
    assert x.x_indices is not None
    assert x.active_step is not None
    assert x.g_index is not None
    f = piecewise_linear_integral_trapz(
        x.x_indices / x.active_step, x.g_index, 0, 1)
    return f

  def f_triads(x: ScenarioStatistics) -> float:
    assert x.triads is not None
    return x.triads

  # First row (a-d): comparing recommendation strategies (St vs Op)
  # Plot a: Event count by recommendation strategy
  data_a = plot_bar(axes[0][0], f=f_ev_count,
                    use_retweet_filters=False, plot_title="Event Count by RecSys")
  axes[0][0].set_title('(a)', loc='left')
  axes[0][0].set_ylabel('#rewiring events')
  axes[0][0].grid(True, linestyle='--', alpha=0.5)
  # axes[0][0].legend()
  axes[0][0].set_yscale('log')
  axes[0][0].set_ylim(1e3, 1e4)

  # Plot b: Event step by recommendation strategy
  data_b = plot_bar(axes[0][1], f=f_ev_step,
                    use_retweet_filters=False, plot_title="Event Step by RecSys")
  axes[0][1].set_title('(b)', loc='left')
  axes[0][1].set_ylabel('norm. occurrence time')
  axes[0][1].grid(True, linestyle='--', alpha=0.5)
  # axes[0][1].legend()
  axes[0][1].set_ylim(0, 0.4)

  # Plot c: AUC of Subjective Polarization Index by recommendation strategy
  data_c = plot_bar(axes[0][2], f=f_auc_subj_pol_index, use_retweet_filters=False,
                    plot_title="AUC of Subjective Polarization Index by RecSys")
  axes[0][2].set_title('(c)', loc='left')
  axes[0][2].set_ylabel(r"AUC of $I_s$")
  axes[0][2].grid(True, linestyle='--', alpha=0.5)
  # axes[0][2].legend()
  axes[0][2].set_ylim(0.3, 0.8)

  # Plot d: Triads by recommendation strategy
  data_d = plot_bar(axes[0][3], f=f_triads,
                    use_retweet_filters=False, plot_title="Triads by RecSys")
  axes[0][3].set_title('(d)', loc='left')
  axes[0][3].set_ylabel('#triads')
  axes[0][3].grid(True, linestyle='--', alpha=0.5)
  axes[0][3].legend(fontsize=8)
  axes[0][3].set_yscale('log')
  axes[0][3].set_ylim(1000, 200000)

  # Second row (e-h): comparing retweet status (!R vs R)
  # Plot e: Event count by retweet status
  data_e = plot_bar(axes[1][0], f=f_ev_count,
                    use_retweet_filters=True, plot_title="Event Count by Retweet")
  axes[1][0].set_title('(e)', loc='left')
  axes[1][0].set_ylabel('#rewiring events')
  axes[1][0].grid(True, linestyle='--', alpha=0.5)
  # axes[1][0].legend()
  axes[1][0].set_yscale('log')
  axes[1][0].set_ylim(1e3, 1e4)

  # Plot f: Event step by retweet status
  data_f = plot_bar(axes[1][1], f=f_ev_step,
                    use_retweet_filters=True, plot_title="Event Step by Retweet")
  axes[1][1].set_title('(f)', loc='left')
  axes[1][1].set_ylabel('norm. occurrence time')
  axes[1][1].grid(True, linestyle='--', alpha=0.5)
  # axes[1][1].legend()
  axes[1][1].set_ylim(0, 0.4)

  # Plot g: AUC of Subjective Polarization Index by retweet status
  data_g = plot_bar(axes[1][2], f=f_auc_subj_pol_index, use_retweet_filters=True,
                    plot_title="AUC of Subjective Polarization Index by Retweet")
  axes[1][2].set_title('(g)', loc='left')
  axes[1][2].set_ylabel(r"AUC of $I_s$")
  axes[1][2].grid(True, linestyle='--', alpha=0.5)
  # axes[1][2].legend()
  axes[1][2].set_ylim(0.3, 0.8)

  # Plot h: Triads by retweet status
  data_h = plot_bar(axes[1][3], f=f_triads,
                    use_retweet_filters=True, plot_title="Triads by Retweet")
  axes[1][3].set_title('(h)', loc='left')
  axes[1][3].set_ylabel('#triads')
  axes[1][3].grid(True, linestyle='--', alpha=0.5)
  axes[1][3].legend(fontsize=8)
  axes[1][3].set_yscale('log')
  axes[1][3].set_ylim(1000, 200000)

  # Optimize layout before saving
  # Save the combined figure
  fig.tight_layout()
  figure_dir = PATHWAY_OUTPUT.resolve() / "figures"
  figure_dir.mkdir(parents=True, exist_ok=True)
  plt_save_and_close(fig, str(figure_dir / "f_grad_index_interpret"))

  with (figure_dir / "f_grad_index_interpret_data.txt").open("w") as f:
    for line in data_a + data_b + data_c + data_d + data_e + data_f + data_g + data_h:
      f.write(line + '\n')

  session.close()
  engine.dispose()
