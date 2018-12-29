###############################################################################
# PA2 reference implementation
# Ignore the divide-by-zero warning by numpy. This happens due to np.log(0.0).
# Run python do_homework2.py to see the answers.
# author: Billy Jun, Isaac Caswell
# date: Jan 20, 2015; Jan 14, 2016
###############################################################################
# Your task:
# implement the code block in build_clique_tree and use the resulting 
# datastructure to answer the homework questions. To do this, you'll also need
# to fill in code blocks the following functions:
# 
# -clique_tree.populate_sepsets
# -clique_tree.query_marginal
# -factors.compose_factors
# -factors.multiply
# -factors.marginalize_all_but
# -solution.build_clique_tree (this file)
#
# The code required for each of these blocks should not be too many lines (in 
# general, 5-15)
# 
# NOTE that you are free to modify the rest of the code as you see fit, or
# indeed completely rewrite it.  Building PGMs is a fairly complex task, and 
# depending on how you think of things, certain representations might be much 
# easier for you personally.
#
# The first thing that occurs to me along these lines is the representation of 
# table factors.  We use a numpy ndarray, and alternate between indexing into 
# the assignment with a tuple of values, and flattening the array and indexing 
# into it with a scalar index. (We provide functions to navigate between these
# two in factors.py).  If there's a way of implementing this that's easier for 
# you to think about, we encourage you to change the factor.val field!
#
# Good luck!
###############################################################################
# Learning goals:
# -basics of implementing and performing inference on a discrete PGM
# -more useful and fun Python
#   -Python classes
#   -lambda functions
#   -fun extra: overriding the __repr__ and __str__ functions
#   -some annoying numpy things

import numpy as np
import sys
from factors import *
from binary_tree import BinaryTreeNode
from mrf_tree import MRFTreeModel

###############################################################################
# Used for parsing and storing the input data.  No need to look at these
###############################################################################


def read_nh_file_into_single_line(filename):
  with open(filename) as f:
    str_to_parse = f.read().replace('\n', '').replace(' ', '').strip(' \t\n\r;')
  return str_to_parse


def create_animal_to_idx(node):
  if node.is_leaf():
    return {node.data : node.idx}
  animal_to_idx = create_animal_to_idx(node.left)
  animal_to_idx.update( create_animal_to_idx(node.right) )
  return animal_to_idx

def read_sequences(filename, animal_to_idx):
  observations = {}
  with open(filename) as f:
    line = f.read()
    raw_data = line.split('>')
    for i in xrange(1, len(raw_data)):
      data = raw_data[i].strip().split('\n')
      observations[animal_to_idx[data[0]]] = ''.join(data[1:])
  return observations

def produce_evidence_sets(sequences):
  assert len(set(map(len, sequences.values()))) == 1
  set_size = len(sequences.values()[0])
  evidence_sets = [{} for i in xrange(set_size)]
  for i in xrange(set_size):
    for key in sequences.keys():
      evidence_sets[i][key] = 'ACGT'.index(sequences[key][i])
  return evidence_sets

def build_binary_tree(treeFile):
  tree = BinaryTreeNode()
  tree.parse_from_string(read_nh_file_into_single_line(treeFile))
  interior_nodes = []
  tree.assign_idx(interior_nodes, [1])
  return tree

def naive_BN_inference(node, cpds, values):
  if node.is_leaf():
    return 1.0
  return cpds[values[node.left.idx]][values[node.idx]] * naive_BN_inference(node.left, cpds, values) * \
         cpds[values[node.right.idx]][values[node.idx]] * naive_BN_inference(node.right, cpds, values)

if __name__ == '__main__':
  USE_LOGSPACE = True

  print "-" * 80
  print "TESTING CODE (OUTPUT CAN BE IGNORED)"

  domain = ['A', 'C', 'G', 'T']
  prior = np.array([0.295, 0.205, 0.205, 0.295])
  # first index = child, second = parent
  cpds = np.array( \
          [[0.831, 0.046, 0.122, 0.053], \
           [0.032, 0.816, 0.028, 0.076], \
           [0.085, 0.028, 0.808, 0.029], \
           [0.052, 0.110, 0.042, 0.842]])

  tree_root = build_binary_tree('tree.nh')
  print 'you now have a tree that looks like this: '
  print repr(tree_root)

  model = MRFTreeModel(tree_root, prior, cpds, use_logspace=USE_LOGSPACE)

  print
  model.find_marginal_distributions()
  model.print_p_evidence_and_marginals()
  likelihood, joint = model.find_joint_most_likely()
  print "(Log?)Likelihood and corresponding vector:", likelihood, joint
  # sanity check
  print "Likelihood of that vector estimated naively:", prior[joint[tree_root.idx]] * naive_BN_inference(tree_root, cpds, joint)

  print "-" * 80
  print "SOLUTION TO PART _B_ AND _C_"
  animal_to_idx = create_animal_to_idx(tree_root)
  sequences = read_sequences("column.fa", animal_to_idx)
  evidence = produce_evidence_sets(sequences)[0]
  model.find_marginal_distributions(evidence, print_messages=True)
  model.print_p_evidence_and_marginals()

  print "-" * 80
  print "SOLUTION TO PART _D_"
  likelihood, joint = model.find_joint_most_likely(evidence)
  print ("Log" if USE_LOGSPACE else "") + "Likelihood and corresponding vector:", likelihood, joint
  # sanity check
  joint_with_evidence = joint
  joint_with_evidence.update(evidence)
  print "Likelihood of that vector estimated naively:", prior[joint[tree_root.idx]] * naive_BN_inference(tree_root, cpds, joint_with_evidence)

  print "-" * 80
  print "SOLUTION TO PART _E_"
  sequences = read_sequences("multicolumn.fa", animal_to_idx)
  evidence_sets = produce_evidence_sets(sequences)
  assert USE_LOGSPACE
  total_loglikelihood = 0.0
  for evidence in evidence_sets:
    model.find_marginal_distributions(evidence)
    total_loglikelihood += model.p_evidence
  print "Total loglikelihood:", total_loglikelihood

  print "-" * 80
  print "SOLUTION TO PART _F_"
  tree_root = build_binary_tree('treealt.nh')
  #print 'you now have a tree that looks like this: '
  #print repr(tree_root)
  model = MRFTreeModel(tree_root, prior, cpds, use_logspace=USE_LOGSPACE)
  animal_to_idx = create_animal_to_idx(tree_root)
  sequences = read_sequences("multicolumn.fa", animal_to_idx)
  evidence_sets = produce_evidence_sets(sequences) 
  total_loglikelihood = 0.0
  for evidence in evidence_sets:
    model.find_marginal_distributions(evidence)
    total_loglikelihood += model.p_evidence
  print "Alternative tree total loglikelihood:", total_loglikelihood
