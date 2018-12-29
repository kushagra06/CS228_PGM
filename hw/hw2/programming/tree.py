import sys
import numpy as np
from collections import Counter
from builtins import map
try:
  import networkx as nx
except ImportError:
  print("!!!!! Please install networkx using 'pip install --upgrade networkx'")
  sys.exit(1)


def renormalize(cnt):
  '''
  renormalize a Counter()
  '''
  tot = 1. * sum(cnt.values())
  for a_i in cnt:
    cnt[a_i] /= tot
  return cnt


def compute_MI(A, C, i, j):
  '''
  compute conditional information I(X_i, X_j | C)
  '''
  I_ij = 0.0
  M, N = A.shape
  for c in range(2):
    mask = (C == c) & (A[:, i] != -1) & (A[:, j] != -1)
    p_c = 1.*np.sum(C == c) / M
    p_joint = Counter({(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0})
    p_joint.update(zip(A[mask, i], A[mask, j]))
    p_joint = renormalize(p_joint)
    p_cond_i = Counter({0: 0, 1: 0})
    p_cond_i.update(A[mask, i])
    p_cond_i = renormalize(p_cond_i)
    p_cond_j = Counter({0: 0, 1: 0})
    p_cond_j.update(A[mask, j])
    p_cond_j = renormalize(p_cond_j)
    idxs = [(a_i, a_j) for a_i in range(2) for a_j in range(2)]
    idxs = [idx for idx in idxs if p_joint[idx] > 0]
    # print('p_joint', [p_joint[idx] for idx in idxs])
    # print('p_cond_i', [p_cond_i[a_i] for a_i, _ in idxs])
    # print('p_cond_j', [p_cond_j[a_j] for _, a_j in idxs])
    _I_ij = p_c * sum(
        np.array([p_joint[idx] for idx in idxs]) *
        np.log([p_joint[idx] for idx in idxs]) - (
            np.log([p_cond_i[a_i] for a_i, _ in idxs]) +
            np.log([p_cond_j[a_j] for _, a_j in idxs])
        )
    )
    assert _I_ij >= 0
    I_ij += _I_ij
  return I_ij


def get_mst(A, C):
  '''
  obtain the Maximal spanning tree from the complete graph over all
  attributes with edges between nodes weighted by the conditional mutual
  information  I(X_i, X_j|C)
  '''
  M, N = A.shape
  G = nx.Graph()
  G.add_nodes_from(range(N))
  for i in range(N):
    for j in range(i):
      G.add_edge(i, j, weight=-compute_MI(A, C, i, j))

  mst = nx.minimum_spanning_tree(G)
  return mst


def get_tree_root(mst):
  '''
  determine the root of the mst.  choose the highest degree node
  '''
  N = mst.number_of_nodes()
  root, deg = max(dict(mst.degree(range(N))).items(), key=lambda x: x[1])
  return root


def get_tree_edges(mst, root):
  '''
  iterate over tree edges from the mst rooted at the specified node
  each edge is a tuple (a,b), which implies a -> b
  '''
  for edge in nx.dfs_edges(mst, root):
    yield edge
