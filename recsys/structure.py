from typing import List, Dict
import numpy as np
from model import HKAgent, HKModel, HKModelRecommendationSystem
from tqdm import tqdm
import networkx as nx


def common_neighbors_count(G: nx.DiGraph, u: int, v: int):
  i1 = len([w for w in G._pred[u] if w in G._pred[v] or w in G._succ[v]])
  i2 = len([w for w in G._succ[u] if w in G._pred[v] or w in G._succ[v]])
  return i1 + i2


class Structure(HKModelRecommendationSystem):

  num_nodes = 0

  def __init__(
      self,
      model: HKModel,
      eta: float = 1,
      sigma: float = 0.5
  ):
    super().__init__(model)
    self.eta = eta if eta > 0 else 0
    self.sigma = sigma if sigma > 0 else -sigma
    self.agent_map: Dict[int, HKAgent] = {}

  def post_init(self):
    self.num_nodes = n = self.model.graph.number_of_nodes()
    self.conn_mat = np.zeros((n, n), dtype=int)
    # calculate full connection matrix
    G = self.model.graph
    for u in tqdm(range(0, self.num_nodes)):
      for v in range(u + 1, self.num_nodes):  # v > u
        self.conn_mat[u, v] = common_neighbors_count(G, u, v)
    # build agent map
    for a in self.model.schedule.agents:
      self.agent_map[a.unique_id] = a

  def pre_step(self):
    pass

  def recommend(self, agent: HKAgent, neighbors: List[HKAgent], count: int) -> List[HKAgent]:
    a = agent.unique_id
    vals = self.conn_mat[:, a].flatten() + \
        self.conn_mat[a, :].flatten()
    epsilon = np.random.normal(0, self.sigma, vals.shape)
    ret1 = vals * (1 - 2 * epsilon) + epsilon
    ret1[ret1 < 0] = 0
    if self.eta != 1:
      ret1 = ret1 ** self.eta

    exclude_ids = np.array([x.unique_id for x in neighbors])
    ret = np.setdiff1d(np.argpartition(
        ret1, len(neighbors) + count), exclude_ids)

    return [self.agent_map[i] for i in ret[:count]]

  def post_step(self, changed: List[int]):
    # update connection matrix
    changed = list(set(changed))
    changed.sort()
    G = self.model.graph
    for i in range(len(changed)):
      for j in range(i + 1, len(changed)):
        u = changed[i]
        v = changed[j]
        self.conn_mat[u, v] = common_neighbors_count(G, u, v)
    pass

  def pre_commit(self):
    pass
