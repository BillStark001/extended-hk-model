import os
import numpy as np
import matplotlib.pyplot as plt

from base.scenario import Scenario

from w_scenarios import all_scenarios
import w_snapshots as ss

def plt_figure():
  return plt.figure(figsize=(8, 6)) 

def plt_save_and_close(path: str):
  plt.savefig(path + '.eps', dpi=300)
  plt.savefig(path + '.png', dpi=300)
  return plt.close()

def plot_data(S: Scenario, base_dir: str):

  steps, opinion, dn, dr = S.get_opinion_data()  # (t, n)
  _p = lambda x: base_dir + x#os.path.join(base_dir, x)

  plt.plot(opinion, lw=0.5)
  plt.title('Opinion')
  plt_save_and_close(_p('opinion'))

  sn = np.std(dn, axis=1)
  sr = np.std(dr, axis=1)

  plt.plot(steps, sn, lw=1)
  plt.plot(steps, sr, lw=1)
  plt.plot(steps, sn + sr, lw=1)
  plt.legend(['Neighbor', 'Recommended', 'Total'])
  plt.title('Standard Deviation of Contribution')
  plt_save_and_close(_p('std_contrib'))

  stats = S.generate_stats()
  stats_index = stats['step']

  plt.plot(stats_index, stats['triads'])
  plt.title('Count of Closed Triads')
  plt_save_and_close(_p('triads'))

  plt.plot(stats_index, stats['cluster'])
  plt.title('Average Clustering Coefficient')
  plt_save_and_close(_p('c_coeff'))

  plt.plot(stats_index, stats['s-index'])
  plt.title('Segregation Index')
  plt_save_and_close(_p('s_index'))

  plt.plot(stats_index, stats['distance-init-o'])
  plt.plot(stats_index, stats['distance-init-s'])
  plt.plot(stats_index, stats['distance-final-o'])
  plt.plot(stats_index, stats['distance-final-s'])
  plt.legend(['O/Best', 'S/Best', 'O/Worst', 'S/Worst'])
  plt.title('Relative Distance of Opinion Distance')
  plt_save_and_close(_p('rd_od'))

  plt.plot(stats_index, stats['in-degree-alpha'])
  plt.plot(stats_index, np.ones(len(stats_index)) * 2, lw=1, linestyle='dashed')
  plt.plot(stats_index, np.ones(len(stats_index)) * 3, lw=1, linestyle='dashed')
  plt.title('In-degree, \\alpha')
  plt_save_and_close(_p('di_alpha'))

  plt.plot(stats_index, stats['in-degree-p-value'])
  plt.plot(stats_index, np.ones(len(stats_index)) * 0.05, lw=1, linestyle='dashed')
  plt.plot(stats_index, np.ones(len(stats_index)) * 0.01, lw=1, linestyle='dashed')
  plt.title('In-degree, p-value')
  plt_save_and_close(_p('di_p_val'))

  if 'in-degree' in stats:
    si = stats['in-degree'][-1]
    plt.plot(si[0], si[1])
    plt.title('Final In-degree Distribution')
    plt_save_and_close(_p('di_final'))



if __name__ == '__main__':

  BASE_PATH = './fig'

  os.makedirs(BASE_PATH, exist_ok=True)
  
  for scenario_name, scenario_params in all_scenarios.items():

    # load model
    model = Scenario(*scenario_params)
    snapshot, snapshot_name = ss.load_latest(scenario_name)
    if not snapshot:
      continue
    model.load(*snapshot)
    
    plot_data(model, os.path.join(BASE_PATH, scenario_name) + '_')
