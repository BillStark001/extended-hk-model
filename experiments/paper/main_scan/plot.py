from typing import Dict, List, Iterable, Callable, Sequence, TypeVar, Tuple
from matplotlib.axes import Axes
from numpy.typing import NDArray

import os
import copy

import numpy as np
from sqlalchemy.orm import Session
from tqdm import tqdm

from ehk.common.plotting import plt_figure, plt_save_and_close, setup_paper_params
from ehk.common.storage.sqlite import create_db_engine_and_session
import experiments.paper.scenarios as cfg
from experiments.paper.paths import STAT_INPUT, MAIN_SCAN_OUTPUT
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


def _create_evaluator(
    full_data_selector: Iterable[ScenarioStatistics]
):
  bc_inst = copy.deepcopy(bc_inst_orig)
  for datum in tqdm(full_data_selector, bar_format=short_progress_bar):
    if datum is None:
      continue
    rw_index = rw_rev_dict[datum.rewiring]
    dc_index = dc_rev_dict[datum.decay]
    bc_inst[rw_index][dc_index].append(datum)

  def eval_func(
      f_ext: Callable[[ScenarioStatistics], float]
  ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:

    bc_inst_num: List[List[List[float]]] = []
    for l_rw in bc_inst:
      bc_inst_num_row = []
      for l_dc_rw in l_rw:
        bc_inst_num_cell: List[float] = [f_ext(x) for x in l_dc_rw]
        bc_inst_num_row.append([
            np.mean(bc_inst_num_cell) if bc_inst_num_cell else None,
            np.std(bc_inst_num_cell) if bc_inst_num_cell else None,
        ])
      bc_inst_num.append(bc_inst_num_row)

    # (rewiring, influence, [mean, std])
    bc_arr = np.array(bc_inst_num, dtype=float)
    bc_mean, bc_std = bc_arr[:, :, 0], bc_arr[:, :, 1]
    return bc_mean, bc_std

  return eval_func


def create_heatmap_evaluator(
    session: Session,
    rs_key: str,
    opinion_peak_count: int | None = None,
):
  full_data_selector: Iterable[ScenarioStatistics] = session.query(ScenarioStatistics).filter(
      *((ScenarioStatistics.last_opinion_peak_count == opinion_peak_count,)
        if opinion_peak_count is not None else ()),
      ScenarioStatistics.name.startswith('s_grad'),
      ScenarioStatistics.recsys_type == rs_key,
      ScenarioStatistics.grad_index.is_not(None),
  )
  return _create_evaluator(full_data_selector)


def create_heatmap_evaluator_retweet(
    session: Session,
    retweet_rate: float,
    rs_key: str | None = None,
):
  full_data_selector: Iterable[ScenarioStatistics] = session.query(ScenarioStatistics).filter(
      ScenarioStatistics.name.startswith('s_grad'),
      *((ScenarioStatistics.recsys_type == rs_key,) if rs_key else ()),
      ScenarioStatistics.retweet == retweet_rate,
      ScenarioStatistics.grad_index.is_not(None),
  )
  return _create_evaluator(full_data_selector)


def _plot_single_heatmap(ax, data_mean, data_std, title, cmap, vmin, vmax, with_std=True):
  """Helper function to plot a single heatmap with optional standard deviation circles."""
  im = ax.imshow(data_mean, cmap=cmap, vmin=vmin, vmax=vmax)
  ax.set_title(title, loc='left')

  # Add standard deviation circles
  if with_std:
    for i in range(data_std.shape[0]):
      for j in range(data_std.shape[1]):
        std_val = data_std[i, j]
        if not np.isnan(std_val):
          ax.scatter(j, i, s=std_val * 1000, facecolors='none',
                     edgecolors='midnightblue', linewidths=2, alpha=0.6)

  return im


def _setup_heatmap_axes(axes: Sequence[Axes], layout='1row'):
  """Helper function to setup common properties for heatmap axes.

  Args:
    axes: List of axes to setup
    layout: '1row' for single row layout, '2row' for 2x3 layout
  """
  for i, ax in enumerate(axes):
    ax.invert_yaxis()
    ax.set_xticks(np.arange(cfg.decay_rate_array.size))
    ax.set_yticks(np.arange(cfg.rewiring_rate_array.size))

    # For 2x3 layout, show y-axis labels on leftmost columns (indices 0 and 3)
    # For 1-row layout, show y-axis labels only on the first axis
    if layout == '2row':
      show_ylabel = (i % 3 == 0)  # First column of each row
      show_xlabel = (i // 3 == 1)  # Only bottom row
    else:
      show_ylabel = (i == 0)  # Only first axis
      show_xlabel = True

    if show_xlabel:
      ax.set_xlabel(r'influence ($\alpha$)')
      ax.set_xticklabels(cfg.decay_rate_array, rotation=90)
    else:
      ax.set_xlabel('')
      ax.set_xticklabels([])

    if show_ylabel:
      ax.set_ylabel(r'rewiring ($q$)')
      ax.set_yticklabels(cfg.rewiring_rate_array)
    else:
      ax.set_ylabel('')
      ax.set_yticklabels([])


def plot_combined_heatmaps(
    metrics: List[Tuple[Callable[[ScenarioStatistics], float], str, dict]],
    with_diff=True,
    with_std=True,
):
  """Plot multiple metrics in a combined figure.

  Args:
    metrics: List of tuples (function, stats_name, plot_kwargs)
    peaks: Peak count filter
    with_diff: Whether to show difference plots
    with_std: Whether to show standard deviation circles
  """
  engine, session = create_db_engine_and_session(
      stats_db_path, ScenarioStatistics.Base)

  # Calculate data for all metrics
  all_data = []
  for f, stats_name, kwargs in metrics:
    peaks = kwargs.get('peaks', None)
    evaluator_st = create_heatmap_evaluator(session, "StructureM9", peaks)
    evaluator_op = create_heatmap_evaluator(session, "OpinionM9", peaks)
    mean_st, std_st = evaluator_st(f)
    mean_op, std_op = evaluator_op(f)
    diff = mean_op - mean_st
    all_data.append(
        (mean_st, std_st, mean_op, std_op, diff, stats_name, kwargs))

  # Setup figure
  n_cols_per_metric = 3 if with_diff else 2
  # Use 2x3 layout instead of 1x6
  fig, axes_orig = plt_figure(
      n_row=2,
      n_col=3,
      hw_ratio=1,
      constrained_layout=False,
  )

  # Flatten axes for easier indexing
  axes = np.array(axes_orig).flatten().tolist()

  for metric_idx, (mean_st, std_st, mean_op, std_op, diff, stats_name, kwargs) in enumerate(all_data):
    base_idx = metric_idx * n_cols_per_metric

    # Get plot parameters
    cmap = kwargs.get('cmap', 'RdYlBu_r')
    heatmap_min = kwargs.get('heatmap_min', 0)
    heatmap_max = kwargs.get('heatmap_max', 1)
    diff_range = kwargs.get('diff_range', 0.6)

    # Generate subplot labels
    label_prefix = chr(ord('a') + metric_idx * n_cols_per_metric)

    # Plot structure heatmap
    ax_st = axes[base_idx]
    im_st = _plot_single_heatmap(
        ax_st, mean_st, std_st,
        f'({chr(ord(label_prefix) + 0)}) {stats_name} (structure)',
        cmap, heatmap_min, heatmap_max, with_std)
    fig.colorbar(im_st, ax=ax_st, fraction=0.046, pad=0.04)

    # Plot opinion heatmap
    ax_op = axes[base_idx + 1]
    im_op = _plot_single_heatmap(
        ax_op, mean_op, std_op,
        f'({chr(ord(label_prefix) + 1)}) {stats_name} (opinion)',
        cmap, heatmap_min, heatmap_max, with_std)
    fig.colorbar(im_op, ax=ax_op, fraction=0.046, pad=0.04)

    # Plot difference heatmap if requested
    if with_diff:
      ax_diff = axes[base_idx + 2]
      cur_chr = chr(ord(label_prefix) + 2)
      sub1_chr = chr(ord(label_prefix) + 0)
      sub2_chr = chr(ord(label_prefix) + 1)
      im_diff = _plot_single_heatmap(
          ax_diff, diff, np.zeros_like(diff),
          f'({cur_chr}) difference ({sub2_chr} - {sub1_chr})',
          'RdBu_r', -diff_range, diff_range, False)
      fig.colorbar(im_diff, ax=ax_diff, fraction=0.046, pad=0.04)

  _setup_heatmap_axes(axes, layout='2row')
  fig.tight_layout()

  session.close()
  engine.dispose()

  return fig


def plot_heatmap_by_retweet(
    f: Callable[[ScenarioStatistics],
                float] = lambda x: x.grad_index,  # type: ignore
    stats_name='$I_w$',
    heatmap_min=0,
    heatmap_max=1,
):
  engine, session = create_db_engine_and_session(
      stats_db_path, ScenarioStatistics.Base)

  evaluators = [
      create_heatmap_evaluator_retweet(session, r, k)
      for r, k in [(0, 'StructureM9'), (0, 'OpinionM9'), (0.5, 'StructureM9'), (0.5, 'OpinionM9'), ]
  ]

  labels = [
      'St., p=0',
      'St., p=0',
      'Op., p=0.5',
      'Op., p=0.5',
  ]

  stats = [e(f) for e in evaluators]

  fig, axes = plt_figure(
      n_row=1, n_col=4, hw_ratio=1,
      constrained_layout=False,
  )

  for i, ax in enumerate(axes):

    grad_index_mean_st, grad_index_std_st = stats[i]

    im1 = ax.imshow(grad_index_mean_st, cmap='YlGnBu',
                    vmin=heatmap_min, vmax=heatmap_max)
    ax.set_title(f'(a) {stats_name} ({labels[i]})', loc='left')
    fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)

  for i, ax in enumerate(axes):
    ax.invert_yaxis()
    ax.set_xticks(np.arange(cfg.decay_rate_array.size))
    ax.set_yticks(np.arange(cfg.rewiring_rate_array.size))

    ax.set_xlabel(r'influence ($\alpha$)')
    ax.set_xticklabels(cfg.decay_rate_array, rotation=90)

    if i == 0:
      ax.set_ylabel(r'rewiring ($q$)')
      ax.set_yticklabels(cfg.rewiring_rate_array)
    else:
      ax.set_ylabel('')
      ax.set_yticklabels([])

  fig.tight_layout()

  session.close()
  engine.dispose()

  return fig


if __name__ == '__main__':
  figure_dir = MAIN_SCAN_OUTPUT.resolve() / "figures"
  figure_dir.mkdir(parents=True, exist_ok=True)

  def f_grad(x: ScenarioStatistics) -> float:
    assert x.grad_index is not None
    return x.grad_index

  p_indices_op = {}
  p_indices_st = {}

  def f_p_backdrop(x: ScenarioStatistics) -> float:
    assert x.p_backdrop is not None and x.p_index is not None
    if x.recsys_type == "OpinionM9":
      p_indices_op[x.name] = x.p_index[-1]
    else:
      p_indices_st[x.name] = x.p_index[-1]
    return x.p_backdrop

  def f_active_step(x: ScenarioStatistics) -> float:
    assert x.active_step is not None
    return np.log10(x.active_step)

  # Create combined plot for gradient index and polarization backdrop index
  metrics = [
      (f_grad, '$I_w$', {
          'cmap': 'RdYlBu_r', 'heatmap_min': 0,
          'heatmap_max': 1, 'diff_range': 0.6
      }),
      (f_p_backdrop, r"$I_p^T$, #P=1", {
          'cmap': 'YlGnBu', 'heatmap_min': 0.5, 'heatmap_max': 2, 'diff_range': 1, 'peaks': 1}),
  ]

  plt_save_and_close(
      plot_combined_heatmaps(
          metrics=metrics,
          with_diff=True,
          with_std=False,  # Disable std for cleaner look as in original f_p_backdrop plot
      ), str(figure_dir / "f_grad_and_pol_trad_index")
  )

  all_indices_op = np.array(list(p_indices_op.values()), dtype=float)
  all_indices_st = np.array(list(p_indices_st.values()), dtype=float)

  stats_op = f"p_index_op stats: mean={np.mean(all_indices_op):.4f}, std={np.std(all_indices_op):.4f}, min={np.min(all_indices_op):.4f}, max={np.max(all_indices_op):.4f}"
  stats_st = f"p_index_st stats: mean={np.mean(all_indices_st):.4f}, std={np.std(all_indices_st):.4f}, min={np.min(all_indices_st):.4f}, max={np.max(all_indices_st):.4f}"

  print(stats_op)
  print(stats_st)

  with (figure_dir / "f_p_backdrop_stats.txt").open("w") as f:
    f.write(stats_op + '\n' + stats_st)

  plt_save_and_close(
      plot_heatmap_by_retweet(
          f=f_active_step,
          stats_name=r'$\log_{10}(t_a)$', heatmap_max=5, heatmap_min=1,
      ), str(figure_dir / "f_p_active_step")
  )
