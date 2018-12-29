import numpy as np
import random

vote_data_path = './data/house-votes-84.complete.data'
incomplete_data_path = './data/house-votes-84.incomplete.data'


def load_vote_data():
  '''
  load voting dataset
  '''
  A = []
  C = []
  with open(vote_data_path) as fin:
    for line in fin:
      entries = line.strip().split(',')
      A_i = list(map(lambda x: 1 if x == 'y' else 0 if x ==
                     'n' else -1, entries[1:]))
      assert -1 not in A_i
      C_i = int(entries[0] == 'democrat')
      A.append(A_i)
      C.append(C_i)
  A = np.vstack(A)
  C = np.array(C)
  M, N = A.shape
  l = range(M)
  A = A[l, :]
  C = C[l]
  return A, C


def load_incomplete_entry():
  '''
  load incomplete entry 1
  '''
  with open(incomplete_data_path) as fin:
    for line in fin:
      entries = line.strip().split(',')
      A_i = list(map(lambda x: 1 if x == 'y' else 0 if x ==
                     'n' else -1, entries[:]))
      return np.array(A_i)
