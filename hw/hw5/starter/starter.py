# Learning: GMM and EM
# Authors: Gunaa A V, Isaac C, Volodymyr K, Haque I, Aditya G
# Last updated: February 27, 2018

import numpy as np
from pprint import pprint
import copy
import math
import time
import matplotlib.pyplot as plt
from collections import defaultdict
from pprint import pprint

LABELED_FILE = "surveylabeled.dat"
UNLABELED_FILE = "surveyunlabeled.dat"


#===============================================================================
# General helper functions

def colorprint(message, color="rand"):
  """Prints your message in pretty colors!

  So far, only the colors below are available.
  """
  if color == 'none':
    print(message)
    return
  if color == 'demo':
    for i in range(99):
      print('%i-' % i + '\033[%sm' % i + message + '\033[0m\t'),
  print('\033[%sm' % {
      'neutral': 99,
      'flashing': 5,
      'underline': 4,
      'magenta_highlight': 45,
      'red_highlight': 41,
      'pink': 35,
      'yellow': 93,
      'teal': 96,
      'rand': np.random.randint(1, 99),
      'green?': 92,
      'red': 91,
      'bold': 1
  }.get(color, 1) + message + '\033[0m')


def read_labeled_matrix(filename):
  """Read and parse the labeled dataset.

  Output:
      Xij: dictionary of measured statistics
          Dictionary is indexed by tuples (i,j).
          The value assigned to each key is a (1,2) numpy.matrix encoding X_ij.
      Zij: dictionary of party choices.
          Dictionary is indexed by tuples (i,j).
          The value assigned to each key is a float.
      N, M: Counts of precincts and voters.
  """
  Zij = {}
  Xij = {}
  M = 0.0
  N = 0.0
  with open(filename, 'r') as f:
    lines = f.readlines()
    for line in lines[1:]:
      i, j, Z, X1, X2 = line.split()
      i, j = int(i), int(j)
      if i > N:
        N = i
      if j > M:
        M = j

      Zij[i-1, j-1] = float(Z)
      Xij[i-1, j-1] = np.matrix([float(X1), float(X2)])
  return Xij, Zij, N, M


def read_unlabeled_matrix(filename):
  """Read and parse the unlabeled dataset.

  Output:
      Xij: dictionary of measured statistics
          Dictionary is indexed by tuples (i,j).
          The value assigned to each key is a (1,2) numpy.matrix encoding X_ij.
      N, M: Counts of precincts and voters.
  """
  Xij = {}
  M = 0.0
  N = 0.0
  with open(filename, 'r') as f:
    lines = f.readlines()
    for line in lines[1:]:
      i, j, X1, X2 = line.split()
      i, j = int(i), int(j)
      if i > N:
        N = i
      if j > M:
        M = j

      Xij[i-1, j-1] = np.matrix([float(X1), float(X2)])
  return Xij, N, M


#===============================================================================
# Functions that define the probability distribution
#
# There are helper functions that you may find useful for solving the problem.
# You do not neet to use them, but we have found them to be helpful.
# Not all of them are implemented. We give a few examples, and you should
# implement the rest. We suggest you first start by implementing the rest of the
# homework and fill-in the missing functions here when you realize that you need
# them.

def p_yi(y_i, phi):
  """Probability of y_i.

  Bernouilli distribution with parameter phi.
  """
  return (phi**y_i) * ((1-phi)**(1-y_i))


def p_zij(z_ij, pi):
  """Probability of z_ij.

  Bernouilli distribution with parameter pi.
  """
  return (pi**z_ij) * ((1-pi)**(1-z_ij))


def p_zij_given_yi(z_ij, y_i, lambd):
  """Probability of z_ij given yi.

  Bernouilli distribution with parameter lambd that targets
  the variable (z_ij == y_i).
  """
  if z_ij == y_i:
    return lambd
  return 1-lambd


def z_marginal(z_ij, lambd, phi):
  """Marginal probability of z_ij with yi marginalized out."""
  return p_zij_given_yi(z_ij, 1, lambd) * p_yi(1, phi) \
      + p_zij_given_yi(z_ij, 0, lambd) * p_yi(0, phi)


def p_xij_given_zij(x_ij, mu_zij, sigma_zij):
  """Probability of x_ij.

  Given by multivariate normal distribution with params mu_zij and sigma_zij.

  Input:
      x_ij: (1,2) array of continuous variables
      mu_zij: (1,2) array representing the mean of class z_ij
      sigma_zij: (2,2) array representing the covariance matrix of class z_ij

  All arrays must be instances of numpy.matrix class.
  """
  assert isinstance(x_ij, np.matrix)
  k = x_ij.shape[1]
  assert(k == 2)

  det_sigma_zij = sigma_zij[0, 0] * \
      sigma_zij[1, 1] - sigma_zij[1, 0]*sigma_zij[0, 1]
  assert det_sigma_zij > 0

  sigma_zij_inv = -copy.copy(sigma_zij)
  sigma_zij_inv[0, 0] = sigma_zij[1, 1]
  sigma_zij_inv[1, 1] = sigma_zij[0, 0]
  sigma_zij_inv /= det_sigma_zij

  # print "better be identity matrix:\n", sigma_zij.dot(sigma_zij_inv)

  multiplicand = (((2*math.pi)**k)*det_sigma_zij)**(-0.5)
  exponentiand = -.5 * (x_ij-mu_zij).dot(sigma_zij_inv).dot((x_ij-mu_zij).T)
  exponentiand = exponentiand[0, 0]
  return multiplicand * np.exp(exponentiand)


def p_zij_given_xij_unnorm(z_ij, x_ij, lambd, phi, mu_zij, sigma_zij):
  """Unnormalized posterior probability of z_ij given x_ij."""

  # -------------------------------------------------------------------------
  # TODO (Optional): Put your code here

  pass

  # END_YOUR_CODE


def p_xij_given_yi(x_ij, y_i, mu_0, sigma_0, mu_1, sigma_1, lambd):
  """Probability of x_ij given y_i.

  To compute this, marginalize (i.e. sum out) over z_ij.
  """

  # -------------------------------------------------------------------------
  # TODO (Optional): Put your code here

  return (p_xij_given_zij(x_ij, mu_0, sigma_0) * p_zij_given_yi(0, y_i, lambd) +
          p_xij_given_zij(x_ij, mu_1, sigma_1) * p_zij_given_yi(1, y_i, lambd))

  # END_YOUR_CODE


def p_yi_given_xij_unnorm(x_ij, y_i, mu_0, sigma_0, mu_1, sigma_1, phi, lambd):
  """Unnormalized posterior probability of y_ij given x_ij.

  Hint: use Bayes' rule!
  """

  # -------------------------------------------------------------------------
  return p_xij_given_yi(x_ij, y_i, mu_0, sigma_0, mu_1, sigma_1, lambd) * p_yi(y_i, phi)

  # END_YOUR_CODE


def MLE_EstimationInternal(Xij, Zij):

  pi = 0.0
  # -------------------------------------------------------------------------
  # TODO: Code to compute pi
  for (i, j), _ in Xij.items():
    pi += Zij[(i, j)]
  z_1_count = pi
  z_0_count = len(Xij) - z_1_count

  pi /= len(Xij)
  # END_YOUR_CODE

  mu0 = np.matrix([0.0, 0.0])
  mu1 = np.matrix([0.0, 0.0])
  # -------------------------------------------------------------------------
  # TODO: Code to compute mu0, mu1

  for (i, j), xij in Xij.items():
    p = Zij[(i, j)]
    mu0 += (1-p) * xij
    mu1 += p * xij

  mu0 /= z_0_count
  mu1 /= z_1_count
  # END_YOUR_CODE

  sigma0 = np.matrix([[0.0, 0.0], [0.0, 0.0]])
  sigma1 = np.matrix([[0.0, 0.0], [0.0, 0.0]])
  # -------------------------------------------------------------------------
  # TODO: Code to compute sigma0, sigma1

  for (i, j), xij in Xij.items():
    p = Zij[(i, j)]
    t0 = (xij - mu0)
    t1 = (xij - mu1)
    sigma0 += (1-p) * np.dot(t0.T, t0)
    sigma1 += p * np.dot(t1.T, t1)

  sigma0 /= z_0_count
  sigma1 /= z_1_count

  # END_YOUR_CODE

  return pi, mu0, mu1, sigma0, sigma1


def MLE_Estimation():
  """Perform MLE estimation of Model A parameters.

  Output:
      pi: (float), estimate of party proportions
      mu0: (1,2) numpy.matrix encoding the estimate of the mean of class 0
      mu1: (1,2) numpy.matrix encoding the estimate of the mean of class 1
      sigma0: (2,2) numpy.matrix encoding the estimate of the covariance of class 0
      sigma1: (2,2) numpy.matrix encoding the estimate of the covariance of class 1
  """

  Xij, Zij, N, M = read_labeled_matrix(LABELED_FILE)
  return MLE_EstimationInternal(Xij, Zij)


def compute_log_likelihoodA(X, mu_0, mu_1, sigma_0, sigma_1, pi):
  """Compute the log-likelihood of the data given our parameters

  Input:
  X: dictionary of measured statistics. Dictionary is indexed by tuples (i,j).
  The value assigned to each key is a (1,2) numpy.matrix encoding X_ij.
  pi: (float), estimate of party proportions
  mu_0: (1,2) numpy.matrix encoding the estimate of the mean of class 0
  mu_1: (1,2) numpy.matrix encoding the estimate of the mean of class 1
  sigma_0: (2,2) numpy.matrix encoding the estimate of the covariance of class 0
  sigma_1: (2,2) numpy.matrix encoding the estimate of the covariance of class 1

  Output:
  ll: (float), value of the log-likelihood
  """
  ll = 0.0
  # -------------------------------------------------------------------------
  # TODO: Code to compute ll
  z_ij_1 = p_zij(1, pi)
  z_ij_0 = p_zij(0, pi)
  for (i, j), xij in X.items():
    ll += np.log(z_ij_0 * p_xij_given_zij(xij, mu_0, sigma_0) +
                 z_ij_1 * p_xij_given_zij(xij, mu_1, sigma_1))

  # END_YOUR_CODE
  return ll


def perform_em_modelA(X, N, M, init_params, max_iters=50, eps=1e-2):
  """Estimate Model A paramters using EM

  Input:
      X: dictionary of measured statistics
          Dictionary is indexed by tuples (i,j).
          The value assigned to each key is a (1,2) numpy.matrix encoding X_ij.
      N, M: Counts of precincts and voters
      init_params: parameters of the model given as a dictionary
          Dictionary shoudl contain params['pi'], params['mu_0'],
          params['mu_1'], params['sigma_0'], params['sigma_1'],

  Output:
      params: parameters of the trained model encoded as a dictionary
          Dictionary shoudl contain params['pi'], params['mu_0'],
          params['mu_1'], params['sigma_0'], params['sigma_1'],
          params['phi'], params['lambda']
      log_likelihood: array of log-likelihood values across iterations
  """

  # unpack the parameters

  mu_0 = init_params['mu_0']
  mu_1 = init_params['mu_1']
  sigma_0 = init_params['sigma_0']
  sigma_1 = init_params['sigma_1']
  pi = init_params['pi']

  # set up list that will hold log-likelihood over time
  log_likelihood = [compute_log_likelihoodA(
      X, mu_0, mu_1, sigma_0, sigma_1, pi)]

  for iter in range(max_iters):
    # -------------------------------------------------------------------------
    # TODO: Code for the E step

    Z = {}
    p_z_1 = p_zij(1, pi)
    p_z_0 = p_zij(0, pi)
    for (i, j), xij in X.items():
      p_z = [p_xij_given_zij(xij, mu_0, sigma0) * p_z_0,
             p_xij_given_zij(xij, mu_1, sigma1) * p_z_1]
      p_z /= np.sum(p_z)
      Z[(i, j)] = p_z[1]

    # END_YOUR_CODE

    # -------------------------------------------------------------------------
    # TODO: Code for the M step
    # You should fill the values of pi, mu_0, mu_1, sigma_0, sigma_1

    pi, mu_0, mu_1, sigma_0, sigma_1 = MLE_EstimationInternal(X, Z)

    # END_YOUR_CODE

    # Check for convergence
    this_ll = compute_log_likelihoodA(X, mu_0, mu_1, sigma_0, sigma_1, pi)
    log_likelihood.append(this_ll)
    if np.abs((this_ll - log_likelihood[-2]) / log_likelihood[-2]) < eps:
      break

  # pack the parameters and return
  params = {}
  params['mu_0'] = mu_0
  params['mu_1'] = mu_1
  params['sigma_0'] = sigma_0
  params['sigma_1'] = sigma_1
  params['pi'] = pi

  return params, log_likelihood


def MLE_of_phi_and_lambda_internal(Y, Z, N, M):
  MLE_phi, MLE_lambda = 0.0, 0.0

  for _, yi in Y.items():
    MLE_phi += yi

  MLE_phi /= N

  for (i, j), zij in Z.items():
    MLE_lambda += (1 - abs(zij - Y[i]))
  MLE_lambda /= (N*M)

  # END_YOUR_CODE

  return MLE_phi, MLE_lambda


def MLE_of_phi_and_lamdba():
  """Perform MLE estimation of Model B parameters.

  Assumes that Y variables have been estimated using heuristic proposed in
  the question.

  Output:
      MLE_phi: estimate of phi
      MLE_lambda: estimate of lambda
  """
  X, Z, N, M = read_labeled_matrix(LABELED_FILE)
  assert(len(Z.items()) == M*N)
  sums = defaultdict(float)
  for (i, j), zij in Z.items():
    sums[i] += zij
  Y = {}
  for i, s in sums.items():
    Y[i] = 1 if (s >= M/2) else 0

  MLE_phi, MLE_lambda = 0.0, 0.0

  # -------------------------------------------------------------------------
  # TODO: Code to compute MLE_phi, MLE_lambda
  MLE_phi, MLE_lambda = MLE_of_phi_and_lambda_internal(Y, Z, N, M)
  # END_YOUR_CODE

  return MLE_phi, MLE_lambda


def compute_posterior_y(X, N, M, params):
  mu_0 = params['mu_0']
  mu_1 = params['mu_1']
  sigma_0 = params['sigma_0']
  sigma_1 = params['sigma_1']
  MLE_phi = params['phi']
  MLE_lambda = params['lambda']

  posterior_y = [None for i in range(N)]

  p_z0_given_y0 = p_zij_given_yi(0, 0, MLE_lambda)
  p_z0_given_y1 = p_zij_given_yi(0, 1, MLE_lambda)
  p_z1_given_y0 = p_zij_given_yi(1, 0, MLE_lambda)
  p_z1_given_y1 = p_zij_given_yi(1, 1, MLE_lambda)
  for i in range(N):
    log_p_y_0 = np.log(p_yi(0, MLE_phi))
    log_p_y_1 = np.log(p_yi(1, MLE_phi))
    for j in range(M):
      log_p_y_0 += np.log(
          p_xij_given_zij(X[(i, j)], mu_0, sigma_0) * p_z0_given_y0 +
          p_xij_given_zij(X[(i, j)], mu_1, sigma_1) * p_z1_given_y0)
      log_p_y_1 += np.log(
          p_xij_given_zij(X[(i, j)], mu_0, sigma_0) * p_z0_given_y1 +
          p_xij_given_zij(X[(i, j)], mu_1, sigma_1) * p_z1_given_y1)
    posterior_y[i] = np.exp(log_p_y_1) / (np.exp(log_p_y_1) + np.exp(log_p_y_0))

  return posterior_y


def estimate_leanings_of_precincts(X, N, M, params=None):
  """Estimate the leanings y_i given data X.

  Input:
      X: dictionary of measured statistics
          Dictionary is indexed by tuples (i,j).
          The value assigned to each key is a (1,2) numpy.matrix encoding X_ij.
      N, M: Counts of precincts and voters
      params: parameters of the model given as a dictionary
          Dictionary shoudl contain params['pi'], params['mu_0'],
          params['mu_1'], params['sigma_0'], params['sigma_1'],
          params['phi'], params['lambda']

  Output:
      Summary: length-N list summarizing the leanings
          Format is: [(i, prob_i, y_i) for i in range(N)]
  """
  if params == None:
    pi, mu_0, mu_1, sigma_0, sigma_1 = MLE_Estimation()
    MLE_phi, MLE_lambda = MLE_of_phi_and_lamdba()
    params = {
        'pi': pi, 'mu_0': mu_0, 'mu_1': mu_1, 'sigma_0': sigma_0,
        'sigma_1': sigma_1, 'phi': MLE_phi, 'lambda': MLE_lambda
    }
  else:
    pi = params['pi']
    mu_0 = params['mu_0']
    mu_1 = params['mu_1']
    sigma_0 = params['sigma_0']
    sigma_1 = params['sigma_1']
    MLE_phi = params['phi']
    MLE_lambda = params['lambda']

  # -------------------------------------------------------------------------
  # TODO: Code to compute posterior_y

  posterior_y = compute_posterior_y(X, N, M, params)

  # END_YOUR_CODE

  summary = [(i, p, 1 if p >= .5 else 0) for i, p in enumerate(posterior_y)]
  return summary


def compute_posterior_z_ij(posterior_y_i, lambd):
  posterior_z = [0, 0]
  posterior_z[0] = p_zij_given_yi(
      0, 0, lambd) * (1 - posterior_y_i) + p_zij_given_yi(
      0, 1, lambd) * posterior_y_i
  posterior_z[1] = p_zij_given_yi(
      1, 0, lambd) * (1 - posterior_y_i) + p_zij_given_yi(
      1, 1, lambd) * posterior_y_i
  assert(abs(1 - np.sum(posterior_z)) < 1e-8)
  return posterior_z


def plot_individual_inclinations(X, N, M, outfile, params=None):
  """Generate 2d plot of inidivudal statistics in each class.

  Input:
      X: dictionary of measured statistics
          Dictionary is indexed by tuples (i,j).
          The value assigned to each key is a (1,2) numpy.matrix encoding X_ij.
      N, M: Counts of precincts and voters
      params: parameters of the model given as a dictionary
          Dictionary shoudl contain params['pi'], params['mu_0'],
          params['mu_1'], params['sigma_0'], params['sigma_1'],
          params['phi'], params['lambda']
  """

  if params == None:
    pi, mu_0, mu_1, sigma_0, sigma_1 = MLE_Estimation()
    MLE_phi, MLE_lambda = MLE_of_phi_and_lamdba()
    params = {
        'pi': pi, 'mu_0': mu_0, 'mu_1': mu_1, 'sigma_0': sigma_0,
        'sigma_1': sigma_1, 'phi': MLE_phi, 'lambda': MLE_lambda
    }
  else:
    pi = params['pi']
    mu_0 = params['mu_0']
    mu_1 = params['mu_1']
    sigma_0 = params['sigma_0']
    sigma_1 = params['sigma_1']
    MLE_phi = params['phi']
    MLE_lambda = params['lambda']

  domain0 = []
  range0 = []
  domain1 = []
  range1 = []

  # p(y_i = 1 | D) = posterior_y[i]
  posterior_y = compute_posterior_y(X, N, M, params)

  for (i, j), x_ij in X.items():
    posterior_z = [0.0, 0.0]

    # -------------------------------------------------------------------------
    # TODO: Code to compute posterior_z
    posterior_z = compute_posterior_z_ij(posterior_y[i], MLE_lambda)
    # END_YOUR_CODE

    if posterior_z[1] >= posterior_z[0]:
      domain0.append(x_ij[0, 0])
      range0.append(x_ij[0, 1])
    else:
      domain1.append(x_ij[0, 0])
      range1.append(x_ij[0, 1])

  plt.plot(domain1, range1, 'r+')
  plt.plot(domain0, range0, 'b+')
  p1,  = plt.plot(mu_0[0, 0], mu_0[0, 1], 'kd')
  p2,  = plt.plot(mu_1[0, 0], mu_1[0, 1], 'kd')
  plt.savefig(outfile)
  plt.show()


def compute_log_likelihoodB(X, mu_0, mu_1, sigma_0, sigma_1, phi, lambd):
  """Compute the log-likelihood of the data given our parameters

  Input:
      mu_0: (1,2) numpy.matrix encoding the estimate of the mean of class 0
      mu_1: (1,2) numpy.matrix encoding the estimate of the mean of class 1
      sigma_0: (2,2) numpy.matrix encoding the estimate of the covariance of class 0
      sigma_1: (2,2) numpy.matrix encoding the estimate of the covariance of class 1
      phi: hyperparameter for precinct preferences
      lambd: hyperparameter for precinct preferences

  Output:
      ll: (float), value of the log-likelihood
  """
  ll = 0.0

  # -------------------------------------------------------------------------
  # TODO: Code to compute ll
  z_ij_1 = z_marginal(1, lambd, phi)
  z_ij_0 = z_marginal(0, lambd, phi)
  for (i, j), xij in X.items():
    ll += np.log(z_ij_0 * p_xij_given_zij(xij, mu_0, sigma_0) +
                 z_ij_1 * p_xij_given_zij(xij, mu_1, sigma_1))

  # END_YOUR_CODE

  return ll


def perform_em(X, N, M, init_params, max_iters=50, eps=1e-2):
  """Estimate Model B paramters using EM

  Input:
      X: dictionary of measured statistics
          Dictionary is indexed by tuples (i,j).
          The value assigned to each key is a (1,2) numpy.matrix encoding X_ij.
      N, M: Counts of precincts and voters
      init_params: parameters of the model given as a dictionary
          Dictionary shoudl contain: params['mu_0'],
          params['mu_1'], params['sigma_0'], params['sigma_1'],
          params['phi'], params['lambda']

  Output:
      params: parameters of the trained model encoded as a dictionary
          Dictionary shoudl contain params['pi'], params['mu_0'],
          params['mu_1'], params['sigma_0'], params['sigma_1'],
          params['phi'], params['lambda']
      log_likelihood: array of log-likelihood values across iterations
  """

  mu_0 = init_params['mu_0']
  mu_1 = init_params['mu_1']
  sigma_0 = init_params['sigma_0']
  sigma_1 = init_params['sigma_1']
  phi = init_params['phi']
  lambd = init_params['lambda']

  log_likelihood = [compute_log_likelihoodB(
      X, mu_0, mu_1, sigma_0, sigma_1, phi, lambd)]

  for iter in range(max_iters):

    # -------------------------------------------------------------------------
    # TODO: Code for the E step

    params = {'mu_0': mu_0, 'mu_1': mu_1, 'sigma_0': sigma_0,
              'sigma_1': sigma_1, 'phi': phi, 'lambda': lambd}
    Y = {i: p_yi for i, p_yi in enumerate(compute_posterior_y(X, N, M, params))}
    Z = {}
    for (i, j), xij in X.items():
      Z[(i, j)] = compute_posterior_z_ij(Y[i], lambd)[1]

    # END_YOUR_CODE

    phi, lambd = 0.0, 0.0
    mu_0 = np.matrix([0.0, 0.0])
    mu_1 = np.matrix([0.0, 0.0])
    sigma_0 = np.matrix([[0.0, 0.0], [0.0, 0.0]])
    sigma_1 = np.matrix([[0.0, 0.0], [0.0, 0.0]])

    # -------------------------------------------------------------------------
    # TODO: Code for the M step
    # You need to compute the above parameters

    _, mu_0, mu_1, sigma_0, sigma_1 = MLE_EstimationInternal(X, Z)
    phi, lambd = MLE_of_phi_and_lambda_internal(Y, Z, N, M)

    # END_YOUR_CODE

    # Check for convergence
    this_ll = compute_log_likelihoodB(
        X, mu_0, mu_1, sigma_0, sigma_1, phi, lambd)
    log_likelihood.append(this_ll)
    if np.abs((this_ll - log_likelihood[-2]) / log_likelihood[-2]) < eps:
      break

  # pack the parameters and return
  params = {}
  params['pi'] = init_params['pi']
  params['mu_0'] = mu_0
  params['mu_1'] = mu_1
  params['sigma_0'] = sigma_0
  params['sigma_1'] = sigma_1
  params['lambda'] = lambd
  params['phi'] = phi

  return params, log_likelihood

#===============================================================================
# This runs the functions that you have defined to produce the answers to the
# assignment problems

#===============================================================================
# pt A.i


pi, mu0, mu1, sigma0, sigma1 = MLE_Estimation()

colorprint("MLE estimates for PA part A.i:", "teal")
colorprint("\tpi: %s\n\tmu_0: %s\n\tmu_1: %s\n\tsigma_0: %s\n\tsigma_1: %s"
           % (pi, mu0, mu1, sigma0, sigma1), "red")

#==============================================================================
# pt A.ii


def random_covariance():
  P = np.matrix(np.random.randn(2, 2))
  D = np.matrix(np.diag(np.random.rand(2) * 0.5 + 1.0))
  return P*D*P.T


X, N, M = read_unlabeled_matrix(UNLABELED_FILE)

# initialization strategy 1
params = {}
pi, mu_0, mu_1, sigma_0, sigma_1 = MLE_Estimation()
params['pi'] = pi
params['mu_0'] = mu_0
params['mu_1'] = mu_1
params['sigma_0'] = sigma_0
params['sigma_1'] = sigma_1
params, log_likelihood = perform_em_modelA(X, N, M, params)
print(log_likelihood)
params_list = [params]
log_likelihood_list = [log_likelihood]

for _ in range(2):
  params = {}
  params['pi'] = np.random.rand()
  params['mu_0'] = np.random.randn(1, 2)
  params['mu_1'] = np.random.randn(1, 2)
  params['sigma_0'] = random_covariance()
  params['sigma_1'] = random_covariance()
  params, log_likelihood = perform_em_modelA(X, N, M, params)
  params_list.append(params)
  log_likelihood_list.append(log_likelihood)

plt.figure()
for i, params in enumerate(params_list):
  print(params)
  plt.plot(log_likelihood_list[i])
plt.legend(['MLE initialization', 'Random initialization',
            'Random initialization'], loc=4)
plt.xlabel('Iteration')
plt.ylabel('Log likelihood')
plt.savefig("2Aii")
plt.show()

#===============================================================================
# pt. B.i

MLE_phi, MLE_lambda = MLE_of_phi_and_lamdba()

colorprint("MLE estimates for PA part B.i:", "teal")
colorprint("\tMLE phi: %s\n\tMLE lambda: %s\n" % (MLE_phi, MLE_lambda), 'red')

#===============================================================================
# pt B.ii

X, N, M = read_unlabeled_matrix(UNLABELED_FILE)
summary = estimate_leanings_of_precincts(X, N, M, params=None)
# TODO: print out the summary just calculated so you can report it in a table as
# required for  this question
pprint(summary)

# END_YOUR_CODE
plot_individual_inclinations(X, N, M,  "2Bii", params=None)

#===============================================================================
# pt B.iv


def random_covariance():
  P = np.matrix(np.random.randn(2, 2))
  D = np.matrix(np.diag(np.random.rand(2) * 0.5 + 1.0))
  return P*D*P.T


X, N, M = read_unlabeled_matrix(UNLABELED_FILE)
# initialization strategy 1
params = {}
pi, mu_0, mu_1, sigma_0, sigma_1 = MLE_Estimation()
MLE_phi, MLE_lambda = MLE_of_phi_and_lamdba()
params['pi'] = pi
params['mu_0'] = mu_0
params['mu_1'] = mu_1
params['sigma_0'] = sigma_0
params['sigma_1'] = sigma_1
params['phi'] = MLE_phi
params['lambda'] = MLE_lambda
params, log_likelihood = perform_em(X, N, M, params)
print(log_likelihood)
params_list = [params]
log_likelihood_list = [log_likelihood]

for _ in range(2):
  params = {}
  params['pi'] = np.random.rand()
  params['mu_0'] = np.random.randn(1, 2)
  params['mu_1'] = np.random.randn(1, 2)
  params['sigma_0'] = random_covariance()
  params['sigma_1'] = random_covariance()
  params['phi'] = np.random.rand()
  params['lambda'] = np.random.rand()
  params, log_likelihood = perform_em(X, N, M, params)
  params_list.append(params)
  log_likelihood_list.append(log_likelihood)

plt.figure()
for i, params in enumerate(params_list):
  print(params)
  plt.plot(log_likelihood_list[i])
plt.legend(['MLE initialization', 'Random initialization',
            'Random initialization'], loc=4)
plt.xlabel('Iteration')
plt.ylabel('Log likelihood')
plt.savefig("2Biv")
plt.show()

#===============================================================================
# pt B.v

X, N, M = read_unlabeled_matrix(UNLABELED_FILE)
# initialization strategy 1
params = {}
pi, mu_0, mu_1, sigma_0, sigma_1 = MLE_Estimation()
MLE_phi, MLE_lambda = MLE_of_phi_and_lamdba()
params['pi'] = pi
params['mu_0'] = mu_0
params['mu_1'] = mu_1
params['sigma_0'] = sigma_0
params['sigma_1'] = sigma_1
params['phi'] = MLE_phi
params['lambda'] = MLE_lambda
params, log_likelihood = perform_em(X, N, M, params)
for k, v in params.items():
  print(k, '=', v)
summary = estimate_leanings_of_precincts(X, N, M, params=params)
plot_individual_inclinations(X, N, M, "2Bv", params=params)
# TODO: print out the summary just calculated so you can report it in a table as
# required for  this question
pprint(summary)

# END_YOUR_CODE
