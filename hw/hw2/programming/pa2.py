import os
import sys
import numpy as np
from scipy.misc import logsumexp
from collections import Counter
import random
import itertools
from itertools import product

# helpers to load data
from data_helper import load_vote_data, load_incomplete_entry
# helpers to learn and traverse the tree over attributes
from tree import get_mst, get_tree_root, get_tree_edges

# pseudocounts for uniform dirichlet prior
alpha = 0.1


def renormalize(cnt):
  '''
  renormalize a Counter()
  '''
  tot = 1. * sum(cnt.values())
  for a_i in cnt:
    cnt[a_i] /= tot
  return cnt

#--------------------------------------------------------------------------
# Naive bayes CPT and classifier
#--------------------------------------------------------------------------


class NBCPT(object):
  '''
  NB Conditional Probability Table (CPT) for a child attribute.  Each child
  has only the class variable as a parent
  '''

  def __init__(self, A_i):
    '''
    TODO create any persistent instance variables you need that hold the
    state of the learned parameters for this CPT
        - A_i: the index of the child variable
    '''
    super(NBCPT, self).__init__()

    self.i = A_i
    # Given each value of c (ie, c = 0 and c = 1)
    self.pseudocounts = [alpha, alpha]
    self.c_count = [2*alpha, 2*alpha]

  def learn(self, A, C):
    '''
    populate any instance variables specified in __init__ to learn
    the parameters for this CPT
        - A: a 2-d numpy array where each row is a sample of assignments
        - C: a 1-d n-element numpy where the elements correspond to the
          class labels of the rows in A
    '''
    for i in range(2):
      self.c_count[i] += len(C[C == i])
      self.pseudocounts[i] += len(C[(A[:, self.i] == 1) & (C == i)])

  def get_cond_prob(self, entry, c):
    '''
    return the conditional probability P(X|Pa(X)) for the values
    specified in the example entry and class label c
        - entry: full assignment of variables
            e.g. entry = np.array([0,1,1]) means A_0 = 0, A_1 = 1, A_2 = 1
        - c: the class
    '''
    entry_is_one_prob = self.pseudocounts[c] / float(self.c_count[c])
    return entry_is_one_prob if entry[self.i] == 1 else (1 - entry_is_one_prob)


class NBClassifier(object):
  '''
  NB classifier class specification
  '''

  def __init__(self, A_train, C_train):
    '''
    TODO create any persistent instance variables you need that hold the
    state of the trained classifier and populate them with a call to
    Suggestions for the attributes in the classifier:
        - P_c: the probabilities for the class variable C
        - cpts: a list of NBCPT objects
    '''
    super(NBClassifier, self).__init__()
    assert(len(np.unique(C_train))) == 2
    n, m = A_train.shape
    self.cpts = [NBCPT(i) for i in range(m)]
    self.P_c = 0.0
    self._train(A_train, C_train)

  def _train(self, A_train, C_train):
    '''
    train your NB classifier with the specified data and class labels
    hint: learn the parameters for the required CPTs
        - A_train: a 2-d numpy array where each row is a sample of assignments
        - C_train: a 1-d n-element numpy where the elements correspond to
          the class labels of the rows in A
    '''
    self.P_c = len(C_train[C_train == 1]) / float(len(C_train))
    for cpt in self.cpts:
      cpt.learn(A_train, C_train)

  def classify(self, entry):
    '''
    return the log probabilites for class == 0 and class == 1 as a
    tuple for the given entry
    - entry: full assignment of variables
    e.g. entry = np.array([0,1,1]) means variable A_0 = 0, A_1 = 1, A_2 = 1

    NOTE this must return both the predicated label {0,1} for the class
    variable and also the log of the conditional probability of this
    assignment in a tuple, e.g. return (c_pred, logP_c_pred)

    '''
    # Calculate the log probability to avoid underflow issues.
    # We DO NOT normalize these results.
    P_c_pred = [0, 0]
    # Find all unobserved so we can try all settings of them.
    unobserved_idx = [i for i, e in enumerate(entry) if e == -1]
    # Add empty set so loop is always executed even when we have full
    # assignment.
    unobserved_assigments = list(itertools.product(
        (0, 1), repeat=len(unobserved_idx))) + [[]]
    for unobserved_assigment in unobserved_assigments:
      probability = 0.0
      for i, value in enumerate(unobserved_assigment):
        entry[unobserved_idx[i]] = value

      # Calculate joint given the above
      full_P_c_pred = [1 - self.P_c, self.P_c]
      for cpt in self.cpts:
        for i in range(2):
          full_P_c_pred[i] *= cpt.get_cond_prob(entry, i)

      for i in range(2):
        P_c_pred[i] += full_P_c_pred[i]

    # Normalize the distributions.
    P_c_pred /= np.sum(P_c_pred)
    c_pred = np.argmax(P_c_pred)
    return (c_pred, np.log(P_c_pred[c_pred]))

  def predict_unobserved(self, entry, index):
    '''
    Predicts P(A_index = 1 \mid entry)
    '''
    if entry[index] == 1 or entry[index] == 0:
      return [1-entry[index], entry[index]]

    # Not observed, so use model to predict.
    P_index_pred = [0.0, 0.0]
    # Find all unobserved so we can try all settings of them except the one
    # we wish to predict.
    unobserved_idx = [i for i, e in enumerate(entry) if e == -1 and i != index]
    # Add empty set so loop is always executed even when we have full
    # assignment.
    unobserved_assigments = list(itertools.product(
        (0, 1), repeat=len(unobserved_idx))) + [[]]
    for p_value in range(2):
      entry[index] = p_value
      for unobserved_assigment in unobserved_assigments:
        probability = 0.0
        for i, value in enumerate(unobserved_assigment):
          entry[unobserved_idx[i]] = value

        # Calculate joint given the above
        full_P_c_pred = [1 - self.P_c, self.P_c]
        for cpt in self.cpts:
          for i in range(2):
            full_P_c_pred[i] *= cpt.get_cond_prob(entry, i)
        # Sum over c.
        P_index_pred[p_value] += np.sum(full_P_c_pred)

      # Normalize the distributions.
    P_index_pred /= np.sum(P_index_pred)
    return P_index_pred

#--------------------------------------------------------------------------
# TANB CPT and classifier
#--------------------------------------------------------------------------


class TANBCPT(object):
  '''
  TANB CPT for a child attribute.  Each child can have one other attribute
  parent (or none in the case of the root), and the class variable as a
  parent
  '''

  def __init__(self, A_i, A_p):
    '''
    TODO create any persistent instance variables you need that hold the
    state of the learned parameters for this CPT
     - A_i: the index of the child variable
     - A_p: the index of its parent variable (in the Chow-Liu algorithm,
       the learned structure will have a single parent for each child)
    '''
    super(TANBCPT, self).__init__()
    self.i = A_i
    self.p = A_p
    # Given each value of c (ie, c = 0 and c = 1) we have access to the pseudo
    # counts for different settings of A_i and A_p.
    # eg. self.pseudocounts[a_i][a_p][c] is the pseudo count of A_i = a_i,
    # A_p = a_p | C = c.
    self.pseudocounts = [[[alpha, alpha], [alpha, alpha]],
                         [[alpha, alpha], [alpha, alpha]]]
    # To make sure the joint probabilites are normalized for each c.
    self.c_count = [4*alpha, 4*alpha]

  def learn(self, A, C):
    '''
    TODO populate any instance variables specified in __init__ to learn
    the parameters for this CPT
     - A: a 2-d numpy array where each row is a sample of assignments
     - C: a 1-d n-element numpy where the elements correspond to the class
       labels of the rows in A
    '''
    for c in range(2):
      self.c_count[c] += len(C[C == c])
      for ai in range(2):
        for ap in range(2):
          self.pseudocounts[ai][ap][c] += len(
              C[(A[:, self.i] == ai) & (A[:, self.p] == ap) & (C == c)])

  def get_cond_prob(self, entry, c):
    '''
    TODO return the conditional probability P(X|Pa(X)) for the values
    specified in the example entry and class label c
        - entry: full assignment of variables
                e.g. entry = np.array([0,1,1]) means A_0 = 0, A_1 = 1, A_2 = 1
        - c: the class
    '''
    ai = entry[self.i]
    ap = entry[self.p]
    return self.pseudocounts[ai][ap][c] / float(self.c_count[c])


class TANBClassifier(NBClassifier):
  '''
  TANB classifier class specification
  '''

  def __init__(self, A_train, C_train):
    '''
    TODO create any persistent instance variables you need that hold the
    state of the trained classifier and populate them with a call to
    _train()
        - A_train: a 2-d numpy array where each row is a sample of
          assignments
        - C_train: a 1-d n-element numpy where the elements correspond to
          the class labels of the rows in A

    '''
    assert(len(np.unique(C_train))) == 2

    mst = get_mst(A_train, C_train)
    root = get_tree_root(mst)
    self.cpts = []
    for (A_p, A_i) in get_tree_edges(mst, root):
      self.cpts.append(TANBCPT(A_i, A_p))

    self.P_c = 0.0

    self._train(A_train, C_train)

# load all data
A_base, C_base = load_vote_data()


def evaluate(classifier_cls, train_subset=False):
  '''
  evaluate the classifier specified by classifier_cls using 10-fold cross
  validation
  - classifier_cls: either NBClassifier or TANBClassifier
  - train_subset: train the classifier on a smaller subset of the training
    data
  NOTE you do *not* need to modify this function

  '''
  global A_base, C_base

  A, C = A_base, C_base

  # score classifier on specified attributes, A, against provided labels,
  # C
  def get_classification_results(classifier, A, C):
    results = []
    pp = []
    for entry, c in zip(A, C):
      c_pred, unused = classifier.classify(entry)
      results.append((c_pred == c))
      pp.append(unused)
    # print('logprobs', np.array(pp))
    return results
  # partition train and test set for 10 rounds
  M, N = A.shape
  tot_correct = 0
  tot_test = 0
  step = int(M / 10 + 1)
  for holdout_round, i in enumerate(range(0, M, step)):
    # print("Holdout round: %s." % (holdout_round + 1))
    A_train = np.vstack([A[0:i, :], A[i+step:, :]])
    C_train = np.hstack([C[0:i], C[i+step:]])
    A_test = A[i: i+step, :]
    C_test = C[i: i+step]
    if train_subset:
      A_train = A_train[: 16, :]
      C_train = C_train[: 16]

    # train the classifiers
    classifier = classifier_cls(A_train, C_train)

    train_results = get_classification_results(classifier, A_train, C_train)
    # print(
    #    '  train correct {}/{}'.format(np.sum(train_results), A_train.shape[0]))
    test_results = get_classification_results(classifier, A_test, C_test)
    tot_correct += sum(test_results)
    tot_test += len(test_results)

  return 1.*tot_correct/tot_test, tot_test


def evaluate_incomplete_entry(classifier_cls):

  global A_base, C_base

  # train a TANB classifier on the full dataset
  classifier = classifier_cls(A_base, C_base)

  # load incomplete entry 1
  entry = load_incomplete_entry()

  c_pred, logP_c_pred = classifier.classify(entry)
  print("  P(C={}|A_observed) = {:2.4f}".format(c_pred, np.exp(logP_c_pred)))

  return


def predict_unobserved(classifier_cls, index=11):
  global A_base, C_base

  # train a TANB classifier on the full dataset
  classifier = classifier_cls(A_base, C_base)

  # load incomplete entry 1
  entry = load_incomplete_entry()

  a_pred = classifier.predict_unobserved(entry, index)
  print("  P(A{}=1|A_observed) = {:2.4f}".format(index+1, a_pred[1]))

  return


def main():
  '''
  TODO modify or add calls to evaluate() to evaluate your implemented
  classifiers
  '''
  print('Naive Bayes')
  accuracy, num_examples = evaluate(NBClassifier, train_subset=False)
  print('  10-fold cross validation total test accuracy {:2.4f} on {} '
        'examples'.format(accuracy, num_examples))

  print('TANB Classifier')
  accuracy, num_examples = evaluate(TANBClassifier, train_subset=False)
  print('  10-fold cross validation total test accuracy {:2.4f} on {} '
        'examples'.format(accuracy, num_examples))

  print('Naive Bayes Classifier on missing data')
  evaluate_incomplete_entry(NBClassifier)

  print('TANB Classifier on missing data')
  evaluate_incomplete_entry(TANBClassifier)

  index = 11
  print('Prediting vote of A%s using NBClassifier on missing data' % (
      index + 1))
  predict_unobserved(NBClassifier, index)
  print('Prediting vote of A%s using TANBClassifier on missing data' % (
      index + 1))
  predict_unobserved(TANBClassifier, index)

  print('Naive Bayes (Small Data)')
  accuracy, num_examples = evaluate(NBClassifier, train_subset=True)
  print('  10-fold cross validation total test accuracy {:2.4f} on {} '
        'examples'.format(accuracy, num_examples))

  print('TANB Classifier (Small Data)')
  accuracy, num_examples = evaluate(TANBClassifier, train_subset=True)
  print('  10-fold cross validation total test accuracy {:2.4f} on {} '
        'examples'.format(accuracy, num_examples))

if __name__ == '__main__':
  main()
