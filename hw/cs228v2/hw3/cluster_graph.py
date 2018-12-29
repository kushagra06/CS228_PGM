###############################################################################
# cluster graph data structure implementation (similar as the CliqueTree
# implementation in PA2)
# author: Billy Jun, Xiaocheng Li
# date: Jan 31st, 2016
###############################################################################

from factors import *
import numpy as np
import math
import random
import multiprocessing as mp

def divide_into_batches(schedule, num_parts):
  schedule_batches = [[] for i in xrange(num_parts)]
  for i in xrange(len(schedule)):
    schedule_batches[i % num_parts].append(schedule[i])
  return schedule_batches


class ClusterGraph:

  def __init__(self, parity_matrix, observed_codeword, p_error):
    '''
        variables - list: list of variable indices
        domains - list: the i-th element represents the domain of the i-th
                     variable;
                     for this programming assignments, all the domains are [0,1]
        var2clique - list of lists: the i-th element is a list with the indices
                     of cliques/factors that contain the i-th variable
        nbr - list of lists: it has the same length with the number of
        cliques/factors,
                    if factor[i] and factor[j] shares variable(s), then j is in
                    nbr[i]
                    and i is in nbr[j]
        factors: a list of Factors
        sepsets: two dimensional array, sepset[i][j] is a list of variables
                shared by factor[i] and factor[j]
        messages: a list of lists to store the messages, keys are (src, dst)
        pairs,
                values are the Factors of sepset[src][dst].
                Here src and dst are the indices for factors.
    '''
    assert (0.0 <= p_error and p_error < 1.0)
    # parity check matrix and dimensions
    self.parity_checks = parity_matrix.shape[0]
    self.codeword_bits = parity_matrix.shape[1]  # same as number of variables
    assert (self.codeword_bits == len(observed_codeword))
    # variables & their domain
    self.variables = range(self.codeword_bits)
    self.domains = [[0, 1] for variable in self.variables]
    # var -> clique map
    self.var2clique = [[clique
                        for clique in xrange(self.parity_checks)
                        if parity_matrix[clique][variable]]
                       for variable in self.variables]
    for i in xrange(self.codeword_bits):
      self.var2clique[i].append(self.parity_checks + i)
    # factors
    self.factors = []
    for clique in xrange(self.parity_checks):
      factor_scope = [variable
                      for variable in self.variables
                      if parity_matrix[clique][variable]]
      factor_card = [len(self.domains[variable]) for variable in factor_scope]
      factor_val = np.tile(1.0, factor_card)
      indices = range(np.prod(factor_card))
      assignments = indices_to_assignments(indices, factor_card)
      for index in indices:
        factor_val.flat[index] = 1.0 - np.mod(np.sum(assignments[index]), 2)
      assert np.sum(factor_val) * 2 == np.prod(factor_card)
      self.factors.append(Factor(scope=factor_scope,
                                 card=factor_card,
                                 val=factor_val,
                                 name="parity_{0}".format(clique)))
    for i in xrange(self.codeword_bits):
      factor_val = [p_error, 1.0 - p_error
                   ] if observed_codeword[i] else [1.0 - p_error, p_error]
      self.factors.append(Factor(scope=[i],
                                 card=[2],
                                 val=factor_val,
                                 name="observed_{0}".format(i)))
    # edges between factors & sepsets
    self.nbr = [[] for i in xrange(len(self.factors))]
    self.sepsets = [[[] for j in xrange(len(self.factors))]
                    for i in xrange(len(self.factors))]
    for i in xrange(self.parity_checks):
      #for j in xrange(i + 1, self.parity_checks):
      #  if np.sum(np.dot(parity_matrix[i], parity_matrix[j])):
      #    self.nbr[i].append(j)
      #    self.nbr[j].append(i)
      #    sepset = [variable
      #              for variable in xrange(self.codeword_bits)
      #              if parity_matrix[i][variable] and parity_matrix[j][variable]
      #             ]
      #    self.sepsets[i][j] = sepset
      #    self.sepsets[j][i] = sepset
      for j in xrange(self.codeword_bits):
        if parity_matrix[i][j]:
          self.nbr[i].append(self.parity_checks + j)
          self.nbr[self.parity_checks + j].append(i)
          self.sepsets[i][self.parity_checks + j].append(j)
          self.sepsets[self.parity_checks + j][i].append(j)
    # initialize messages dict
    self.messages = [[None for j in xrange(len(self.factors))]
                     for i in xrange(len(self.factors))]
    for src in xrange(len(self.factors)):
      for dst in self.nbr[src]:
        message_scope = self.sepsets[src][dst]
        message_card = [len(self.domains[variable])
                        for variable in message_scope]
        self.messages[src][dst] = Factor(scope=message_scope,
                                     card=message_card,
                                     val=np.tile(1.0, message_card),
                                     name="message {0}->{1}".format(src, dst))
    pass

  def evaluateWeight(self, assignment):
    '''
        param - assignment: the full assignment of all the variables
        return: the multiplication of all the factors' values for this
        assigments
        '''
    a = np.array(assignment, copy=True)
    output = 1.0
    for f in self.factors:
      output *= f.val.flat[assignments_to_indices([a[f.scope]], f.card)]
    return output

  def calculateNewMessage(self, src, dst):
    product = self.factors[src]
    for neighbor in set(self.nbr[src]) - set([dst]):
      product = product.multiply(self.messages[neighbor][src])
      product = product.normalize()
    product = product.marginalize_all_but(self.sepsets[src][dst])
    #print 'updating message for', src, dst
    self.new_messages[src][dst] = product

  def processMessageUpdatesBatch(self, pairs_batch):
    for (src, dst) in pairs_batch:
      self.calculateNewMessage(src, dst)

  def runParallelLoopyBP(self, iterations, num_threads):
    '''
        param - iterations: the number of iterations you do loopy BP

        In this method, you need to implement the loopy BP algorithm. The only
        values
        you should update in this function is self.messages.

        Warning: Don't forget to normalize the message at each time. You may
        find the normalize
        method in Factor useful.
        '''
    schedule = []
    for src in xrange(len(self.factors)):
      for dst in self.nbr[src]:
        schedule.append((src, dst))
    self.new_messages = [[[] for j in xrange(len(self.factors))] for i in xrange(len(self.factors))]
    for iter in range(iterations):
      random.shuffle(schedule)
      if num_threads == 1:
        for (src, dst) in schedule:
          self.calculateNewMessage(src, dst)
      else:
        schedule_batches = divide_into_batches(schedule, num_threads)
        processes = [mp.Process(target=self.processMessageUpdatesBatch, args=(batch,)) for batch in schedule_batches]
        for p in processes:
          p.start()
        for p in processes:
          p.join()
      self.messages = self.new_messages

  def print_all_messages(self):
    for src in xrange(len(self.factors)):
      for dst in self.nbr[src]:
        print "Message from {0} to {1}".format(src, dst)
        print repr(self.messages[src][dst])

  def estimateMarginalProbability(self, variable):
    '''
        param - variable: a single variable index
        return: the marginal probability of the variable

        example:
        >>> cluster_graph.estimateMarginalProbability(0)
        >>> [0.2, 0.8]

        Since in this assignment, we only care about the marginal
        probability of a single variable, you only need to implement the
        marginal
        query of a single variable.
    '''
    clique = self.var2clique[variable][-1]
    belief = self.factors[clique]
    for neighbor in self.nbr[clique]:
      belief = belief.multiply(self.messages[neighbor][clique])
      belief = belief.normalize()
    belief = belief.marginalize_all_but([variable])
    return np.array(belief.val, copy=True)

  def getMarginalMAP(self):
    '''
        In this method, the return value output should be the marginal MAP
        assignments for the variables. You may utilize the method
        estimateMarginalProbability.

        example: (N=2, 2*N=4)
        >>> cluster_graph.getMarginalMAP()
        >>> [0, 1, 0, 0]
        '''
    output = np.zeros(len(self.variables))
    for variable in self.variables:
      val = self.estimateMarginalProbability(variable)
      if val[1] > val[0]:
        output[variable] = 1.0
    return output
