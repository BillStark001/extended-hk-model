from typing import Iterable, TypeVar

import os

import numpy as np

from sqlalchemy.orm import Session

from ehk.common.plotting import plt_figure, plt_save_and_close, setup_paper_params
from ehk.common.storage.sqlite import create_db_session
import experiments.paper.scenarios as cfg
from experiments.paper.paths import STAT_INPUT, EPSILON_SCAN_OUTPUT
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


def _():
  pass

def get_random_stats(
    session: Session,
    decay: float | None = None,
    rewiring: float | None = None,
    retweet: float | None = None,
):
  full_data_selector: Iterable[ScenarioStatistics] = session.query(ScenarioStatistics).filter(
      ScenarioStatistics.name.startswith('s_eps'),
      ScenarioStatistics.recsys_type == 'Random',
      *(x for x in (
          (ScenarioStatistics.decay == decay) if decay is not None else None,
          (ScenarioStatistics.rewiring == rewiring) if rewiring is not None else None,
          (ScenarioStatistics.retweet == retweet) if retweet is not None else None,
      ) if x is not None)
  )
  return list(full_data_selector)


if __name__ == '__main__':

  stats_session = create_db_session(stats_db_path, ScenarioStatistics.Base)

  stats = get_random_stats(stats_session)

  all_epsilon_vals = set(x.tolerance for x in stats)
  vals = {x: [] for x in sorted(list(all_epsilon_vals))}
  for x in stats:
    vals[x.tolerance].append(
        (x.grad_index, x.last_community_count, x.last_opinion_peak_count))

  plt_x = np.array(list(vals.keys()))
  plt_ig_avg, plt_cm_avg, plt_cp_avg, \
      plt_ig_std, plt_cm_std, plt_cp_std = np.array(
          [(np.mean(v, axis=0), np.std(v, axis=0)) for v in vals.values()]
      ).reshape(-1, 6).T

  fig, axes = plt_figure(n_row=1, n_col=3)
  (ax1, ax2, ax3) = axes

  ax1.errorbar(plt_x, plt_ig_avg, yerr=plt_ig_std, zorder=1)
  ax2.errorbar(plt_x, plt_cm_avg, yerr=plt_cm_std, zorder=1)
  ax3.errorbar(plt_x, plt_cp_avg, yerr=plt_cp_std, zorder=1)

  ax1.scatter(plt_x, plt_ig_avg, marker='x', c='black', zorder=2)
  ax2.scatter(plt_x, plt_cm_avg, marker='x', c='black', zorder=2)
  ax3.scatter(plt_x, plt_cp_avg, marker='x', c='black', zorder=2)

  for ax in axes:
    ax.grid(True)
    ax.set_xlabel(r'tolerance ($\epsilon$)')

  ax1.set_title(r'(a) $I_w$', loc='left')
  ax2.set_title(r'(b) #community (Leiden)', loc='left')
  ax3.set_title(r'(c) #peaks', loc='left')
  

  figure_dir = EPSILON_SCAN_OUTPUT.resolve() / "figures"
  figure_dir.mkdir(parents=True, exist_ok=True)
  plt_save_and_close(fig, str(figure_dir / "f_eps_select"))
