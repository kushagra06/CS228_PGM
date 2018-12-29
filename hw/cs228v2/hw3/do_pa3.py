###############################################################################
# Finishes PA 3
# author: Billy Jun, Xiaocheng Li
# date: Jan 31, 2016
###############################################################################

## Utility code for PA3
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from cluster_graph import *
from factors import *
import random

def loadLDPC(name):
  """
    :param - name: the name of the file containing LDPC matrices

    return values:
    G: generator matrix
    H: parity check matrix
    """
  A = sio.loadmat(name)
  G = A['G']
  H = A['H']
  return G, H


def loadImage(fname, iname):
  '''
    :param - fname: the file name containing the image
    :param - iname: the name of the image
    (We will provide the code using this function, so you don't need to worry
    too much about it)

    return: image data in matrix form
    '''
  img = sio.loadmat(fname)
  return img[iname]


def applyChannelNoise(y, p_error):
  '''
    :param y - codeword with 2N entries
    :param p_error channel noise probability

    return corrupt message yhat
    yhat_i is obtained by flipping y_i with probability p
    '''
  yhat = np.array(y, copy=True)
  for i in xrange(len(yhat)):
    if random.random() < p_error:
      yhat[i] = 1 - yhat[i]
  return yhat

def encodeMessage(x, G):
  '''
    :param - x orginal message
    :param[in] G generator matrix
    :return codeword y=Gx mod 2
    '''
  encoded = np.zeros(len(G))
  for i in xrange(len(G)):
    assert len(G[i]) == len(x)
    for j in xrange(len(x)):
      encoded[i] += G[i][j] * x[j]
  return np.mod(encoded, 2)


def constructClusterGraph(yhat, H, p):
  '''
    :param - yhat: observed codeword
    :param - H parity check matrix
    :param - p channel noise probability

    return G clusterGraph

    You should consider two kinds of factors:
    - M unary factors
    - N each parity check factors
    '''
  G = ClusterGraph(H, yhat, p)
  return G


def do_part_a():
  yhat = [1, 1, 1, 1, 1]
  H = np.array([ \
      [0, 1, 1, 0, 1], \
      [0, 1, 0, 1, 1], \
      [1, 1, 0, 1, 0], \
      [1, 0, 1, 1, 0], \
      [1, 0, 1, 0, 1]])
  p = 0.05
  G = ClusterGraph(H, yhat, p)
  ##############################################################
  # To do: your code starts here 
  # Design two invalid codewords ytest1, ytest2 and one valid codewords ytest3.
  # Report their weights respectively.
  ytest1 = [1, 1, 1, 1, 1]
  ytest2 = [0, 1, 1, 0, 1]
  ytest3 = [0, 0, 0, 0, 0]
  ##############################################################
  print(
      G.evaluateWeight(ytest1), \
      G.evaluateWeight(ytest2), \
      G.evaluateWeight(ytest3))

def test_part_c():
  H = np.array([ \
      [1, 1, 1, 1, 0], \
      [0, 1, 1, 0, 1]])
  G = np.array([
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
      [1, 1, 1],
      [0, 1, 1]])
  p_error = 0.05
  for iter in xrange(1):
    x = np.array([0, 0, 0])
    y = encodeMessage(x, G)
    print 'message:', x, 'encoded:', y
    y_transmitted = applyChannelNoise(y, p_error)
    print 'with noise:', y_transmitted
    graph = ClusterGraph(H, y_transmitted, p_error)
    graph.runParallelLoopyBP(50, 1)
    graph.print_all_messages()
    for bit in xrange(len(y)):
      marginal = graph.estimateMarginalProbability(bit)
      print 'marginal prob for bit', bit, marginal

def do_part_c():
  '''
    In part b, we provide you an all-zero initialization of message x, you
    should
    apply noise on y to get yhat, and then do loopy BP to obatin the
    marginal probabilities of the unobserved y_i's.
    '''
  G, H = loadLDPC('ldpc36-128.mat')
  p_error = 0.05
  N = G.shape[1]
  x = np.zeros(N, dtype='int32')
  y = encodeMessage(x, G)
  y_transmitted = applyChannelNoise(y, p_error)
  print "Transmitted message:", y_transmitted
  graph = ClusterGraph(H, y_transmitted, p_error)
  graph.runParallelLoopyBP(50, 1)
  #graph.print_all_messages()
  probabilities = []
  for bit in xrange(len(y)):
    probabilities.append(graph.estimateMarginalProbability(bit)[1])
  print probabilities

def do_part_de(numTrials, p_error, iterations=50):
  '''
    param - numTrials: how many trials we repreat the experiments
    param - error: the transmission error probability
    param - iterations: number of Loopy BP iterations we run for each trial
    '''
  G, H = loadLDPC('ldpc36-128.mat')
  N = G.shape[1]
  errors = [[] for i in xrange(numTrials)]
  for trial in xrange(numTrials):
    print "Trial", trial
    x = np.zeros(N, dtype='int32')
    y = encodeMessage(x, G)
    y_transmitted = applyChannelNoise(y, p_error)
    print "Transmitted message:", y_transmitted
    graph = ClusterGraph(H, y_transmitted, p_error)
    for iteration in xrange(iterations):
      graph.runParallelLoopyBP(1, 1)
      errors[trial].append(0)
      for bit in xrange(len(y)):
        val = graph.estimateMarginalProbability(bit)
        if val[1] > val[0]:
          errors[trial][-1] += 1
  print "Errors per trial per iteration"
  print errors


def print_image(data):
  assert len(data) == 40 * 40
  for line in xrange(40):
    for column in xrange(40):
      print ('x' if data[40 * line + column] > 0.5 else ' '),
    print
  print
  print '-' * 40

def do_part_fg(p_error):
  '''
    param - error: the transmission error probability
    '''
  G, H = loadLDPC('ldpc36-1600.mat')
  img = loadImage('images.mat', 'cs242')
  data = img.flatten()
  print "Source image:"
  print_image(data)
  codeword = encodeMessage(data, G)
  transmitted = applyChannelNoise(codeword, p_error)
  graph = ClusterGraph(H, transmitted, p_error)
  iterations = [0, 0, 1, 2, 3, 5, 10, 20, 30]
  for step in xrange(1, len(iterations)):
    delta = iterations[step] - iterations[step - 1]
    graph.runParallelLoopyBP(delta, 1)
    decoded = graph.getMarginalMAP()
    print "Image after {0} iterations of LBP".format(iterations[step])
    print_image(decoded[:1600])

random.seed(0x31337)
print('Doing part (a): Should see 0.0, 0.0, >0.0')
#do_part_a()
print('Doing part (c)')
#test_part_c()
do_part_c()
print('Doing part (d)')
do_part_de(10, 0.06)
print('Doing part (e)')
print 'Error probability 0.08'
do_part_de(10, 0.08)
print 'Error probability 0.10'
do_part_de(10, 0.10)
print('Doing part (f)')
#do_part_fg(0.06)
print('Doing part (g)')
#do_part_fg(0.10)
