from typing import Iterable, TypeVar

import os

import numpy as np

from sqlalchemy.orm import Session

from ehk.common.plotting import plt_figure, plt_save_and_close, setup_paper_params
from ehk.common.storage.sqlite import create_db_session
import experiments.paper.scenarios as cfg
from experiments.paper.paths import STAT_INPUT, REPLICATE_OUTPUT
from experiments.paper.analysis.records import ScenarioStatistics


T = TypeVar("T")
T1 = TypeVar("T1")
T2 = TypeVar("T2")

# parameters

plot_path = str(STAT_INPUT.resolve())
stats_db_path = os.path.join(plot_path, 'stats.eps.db')

rs_keys = list(cfg.rs_names)
tw_keys = cfg.retweet_rate_array.tolist()


setup_paper_params()

# utilities

# build scenario
short_progress_bar = "{l_bar}{bar:10}{r_bar}{bar:-10b}"


def get_random_stats(
    session: Session,
    decay: float | None = None,
    rewiring: float | None = None,
    retweet: float | None = None,
):
  full_data_selector: Iterable[ScenarioStatistics] = session.query(ScenarioStatistics).filter(
      ScenarioStatistics.name.startswith('s_rep'),
      *(x for x in (
          (ScenarioStatistics.decay == decay) if decay is not None else None,
          (ScenarioStatistics.rewiring == rewiring) if rewiring is not None else None,
          (ScenarioStatistics.retweet == retweet) if retweet is not None else None,
      ) if x is not None)
  )
  return list(full_data_selector)


def piecewise_linear_integral_trapz(x: np.ndarray, y: np.ndarray, a: float, b: float):
  # Normalize the interval orientation.
  if a > b:
    a, b = b, a

  # Include interval endpoints.
  x_new = np.sort(np.concatenate([x, [a, b]]))
  y_new = np.interp(x_new, x, y, left=0, right=1)

  # Keep points inside the interval.
  mask = (x_new >= a) & (x_new <= b)
  x_new = x_new[mask]
  y_new = y_new[mask]

  # Integrate the clipped curve.
  return np.trapz(y_new, x_new)


if __name__ == '__main__':

  stats_session = create_db_session(stats_db_path, ScenarioStatistics.Base)

  stats = get_random_stats(stats_session)

  all_epsilon_vals = set(x.recsys_type for x in stats)
  vals = {x: [] for x in sorted(list(all_epsilon_vals))}
  for x in stats:
    vals[x.recsys_type].append(
        (x.grad_index, x.last_community_count, x.last_opinion_peak_count, x.triads))

  plt_x = np.array(list(vals.keys()))
  plt_ig_avg, plt_cm_avg, plt_cp_avg, plt_ti_avg, plt_ig_std, plt_cm_std, plt_cp_std, plt_ti_std = np.array(
      [(np.mean(v, axis=0), np.std(v, axis=0)) for v in vals.values()]
  ).reshape(-1, 8).T

  fig, axes = plt_figure(n_row=1, n_col=4, hw_ratio=2/1, total_width=10)

  (ax1, ax2, ax3, ax4) = axes

  ax1.bar(plt_x, plt_ig_avg, yerr=plt_ig_std,
          color='skyblue', edgecolor='black', alpha=0.5)
  ax2.bar(plt_x, plt_cm_avg, yerr=plt_cm_std,
          color='skyblue', edgecolor='black', alpha=0.5)
  ax3.bar(plt_x, plt_cp_avg, yerr=plt_cp_std,
          color='skyblue', edgecolor='black', alpha=0.5)
  ax4.bar(plt_x, plt_ti_avg, yerr=plt_ti_std,
          color='skyblue', edgecolor='black', alpha=0.5)

  ax1.set_title(r'(a) $I_w$', loc='left')
  ax2.set_title(r'(b) #community', loc='left')
  ax3.set_title(r'(c) #peaks', loc='left')
  ax4.set_title(r'(d) #triads', loc='left')

  for ax in axes:
    ax.grid(True)
    ax.set_xticklabels(plt_x, rotation=45)

  fig.tight_layout()
  figure_dir = REPLICATE_OUTPUT.resolve() / "figures"
  figure_dir.mkdir(parents=True, exist_ok=True)
  plt_save_and_close(fig, str(figure_dir / "f_replicate_zignani"))
