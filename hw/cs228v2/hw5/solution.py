import copy
import math
import numpy as np
from scipy.special import comb, expit
from scipy.stats import multivariate_normal

from util import *

def estimate_model1_mle(x, z):
  n, m = len(x), len(x[0])
  sum1 = 1.0 * np.sum(z)
  sum0 = n * m - sum1
  pi = 1.0 * sum1 / (sum0 + sum1)
  mean0 = np.array([0.0, 0.0])
  mean1 = np.array([0.0, 0.0])
  for i in xrange(n):
    for j in xrange(m):
      mean1 += z[i][j] / sum1 * x[i][j]
      mean0 += (1.0 - z[i][j]) / sum0 * x[i][j]
  cov0 = np.matrix([[0.0, 0.0], [0.0, 0.0]])
  cov1 = np.matrix([[0.0, 0.0], [0.0, 0.0]])
  for i in xrange(n):
    for j in xrange(m):
      cov1 += z[i][j] / sum1 * np.matrix(x[i][j] - mean1).T * (x[i][j] - mean1)
      cov0 += (1.0 - z[i][j]) / sum0 * np.matrix(x[i][j] - mean0).T * (x[i][j] - mean0)
  return pi, mean0, mean1, cov0, cov1

def estimate_y_by_consensus(z):
  n, m = len(z), len(z[0])
  y = np.array([0] * n)
  for i in xrange(n):
    if sum(z[i]) > 0.5 * m:
      y[i] = 1
  return y

def estimate_model2_phi_lambd(x, y, z):
  n, m = len(x), len(x[0])
  phi = 1.0 * sum(y) / n
  matches, mismatches = 0, 0
  for i in xrange(n):
    for j in xrange(m):
      matches += z[i][j] * y[i] + (1 - z[i][j]) * (1 - y[i])
      mismatches += z[i][j] * (1 - y[i]) + (1 - z[i][j]) * y[i]
  lambd = 1.0 * matches / (matches + mismatches)
  return phi, lambd

def estimate_posterior_y_by_x(x_row, phi, lambd, mean0, mean1, cov0, cov1):
  m = len(x_row)
  p_y_1_running = math.log(phi)
  p_y_0_running = math.log(1.0 - phi)
  for j in xrange(m):
    # sending messages from Z_j to Y and marginalizing all-in-one
    p_y_1_running += math.log(lambd * multivariate_normal.pdf(x_row[j], mean=mean1, cov=cov1) + (1 - lambd) * multivariate_normal.pdf(x_row[j], mean=mean0, cov=cov0))
    p_y_0_running += math.log((1 - lambd) * multivariate_normal.pdf(x_row[j], mean=mean1, cov=cov1) + lambd * multivariate_normal.pdf(x_row[j], mean=mean0, cov=cov0))
  return expit(p_y_1_running - p_y_0_running)

def estimate_posterior_z_by_x(x_row, phi, lambd, mean0, mean1, cov0, cov1):
  m = len(x_row)
  posterior_y = estimate_posterior_y_by_x(x_row, phi, lambd, mean0, mean1, cov0, cov1)
  posterior_z = [None] * m
  for j in xrange(m):
    p_z_0 = (posterior_y * (1 - lambd) + (1 - posterior_y) * lambd) * multivariate_normal.pdf(x_row[j], mean=mean0, cov=cov0)
    p_z_1 = (posterior_y * lambd + (1 - posterior_y) * (1 - lambd)) * multivariate_normal.pdf(x_row[j], mean=mean1, cov=cov1)
    posterior_z[j] = p_z_1 / (p_z_0 + p_z_1)
  return posterior_z

def compute_model1_loglikelihood(x, pi, mean0, mean1, cov0, cov1):
  n, m = len(x), len(x[0])
  loglikelihood = 0.0
  for i in xrange(n):
    for j in xrange(m):
      p_x_ij = pi * multivariate_normal.pdf(x[i][j], mean=mean1, cov=cov1) + \
             (1 - pi) * multivariate_normal.pdf(x[i][j], mean=mean0, cov=cov0)
      loglikelihood += math.log(p_x_ij)
  return loglikelihood

def estimate_model1_params_by_em(x, pi, mean0, mean1, cov0, cov1):
  n, m = len(x), len(x[0])
  loglikelihoods = []
  for iteration in xrange(100):
    loglikelihoods.append(compute_model1_loglikelihood(x, pi, mean0, mean1, cov0, cov1))
    # E step
    posterior_z = [[None for j in xrange(m)] for i in xrange(n)]
    for i in xrange(n):
      for j in xrange(m):
        p_z_0 = (1 - pi) * multivariate_normal.pdf(x[i][j], mean=mean0, cov=cov0)
        p_z_1 = pi * multivariate_normal.pdf(x[i][j], mean=mean1, cov=cov1)
        posterior_z[i][j] = p_z_1 / (p_z_0 + p_z_1)
    # M step
    # magically works since Zs can also be weights, not just 0-1 integers
    pi, mean0, mean1, cov0, cov1 = estimate_model1_mle(x, posterior_z)
  print loglikelihoods
  return pi, mean0, mean1, cov0, cov1

def compute_model2_loglikelihood(x, phi, lambd, mean0, mean1, cov0, cov1):
  n, m = len(x), len(x[0])
  loglikelihood = 0.0
  for i in xrange(n):
    for j in xrange(m):
      p_x_ij = (phi * lambd + (1 - phi) * (1 - lambd)) * multivariate_normal.pdf(x[i][j], mean=mean1, cov=cov1) + \
               ((1 - phi) * lambd + phi * (1 - lambd)) * multivariate_normal.pdf(x[i][j], mean=mean0, cov=cov0)
      try:
        loglikelihood += math.log(p_x_ij)
      except Exception:
        print 'ERROR!'
        print p_x_ij
  return loglikelihood

def estimate_model2_params_by_em(x, phi, lambd, mean0, mean1, cov0, cov1):
  n, m = len(x), len(x[0])
  loglikelihoods = []
  for iteration in xrange(100):
    loglikelihoods.append(compute_model2_loglikelihood(x, phi, lambd, mean0, mean1, cov0, cov1))
    #print 'Iteration {0} log-likelihood {1}'.format(iteration, loglikelihoods[-1])
    #print phi, lambd, mean0, mean1, cov0, cov1
    # E step
    posterior_y = [estimate_posterior_y_by_x(x[i], phi, lambd, mean0, mean1, cov0, cov1) for i in xrange(n)]
    posterior_z = [estimate_posterior_z_by_x(x[i], phi, lambd, mean0, mean1, cov0, cov1) for i in xrange(n)]
    # M step
    # magically works since Zs can also be weights, not just 0-1 integers
    _, mean0, mean1, cov0, cov1 = estimate_model1_mle(x, posterior_z)
    phi, lambd = estimate_model2_phi_lambd(x, posterior_y, posterior_z)
  print loglikelihoods
  return phi, lambd, mean0, mean1, cov0, cov1

def estimate_params_from_labeled_data():
  x, z, n, m = read_data('data/labeled.txt', labeled=True)
  pi, mean0, mean1, cov0, cov1 = estimate_model1_mle(x, z)
  phi, lambd = estimate_model2_phi_lambd(x, estimate_y_by_consensus(z), z)
  return pi, phi, lambd, mean0, mean1, cov0, cov1

def do_part_a_I():
  section('Part (a) - I - estimating model 1')
  pi, phi, lambd, mean0, mean1, cov0, cov1 = estimate_params_from_labeled_data()
  print pi
  print mean0, mean1
  print cov0
  print cov1

def do_part_a_II():
  section('Part (a) - II - estimating model 2')
  pi, phi, lambd, mean0, mean1, cov0, cov1 = estimate_params_from_labeled_data()
  print phi, lambd
  print mean0, mean1
  print cov0
  print cov1

def do_part_a_III():
  section('Part (a) - III - estimating precinct leanings')
  pi, phi, lambd, mean0, mean1, cov0, cov1 = estimate_params_from_labeled_data()
  x, _, n, m = read_data('data/unlabeled.txt', labeled=False)
  posterior_y = [estimate_posterior_y_by_x(x[i], phi, lambd, mean0, mean1, cov0, cov1) for i in xrange(n)]
  posterior_z = [estimate_posterior_z_by_x(x[i], phi, lambd, mean0, mean1, cov0, cov1) for i in xrange(n)]
  print_precinct_leanings(posterior_y)
  plot_z_distribution(posterior_z, 'part_a_iii_z_table.png')
  plot_individual_inclinations(x, posterior_z, mean0, mean1, 'part_a_iii_z_starter.png')

def do_part_b_I():
  section('Part (b) - I - EM model1 estimation from unlabeled data')
  x, _, n, m = read_data('data/unlabeled.txt', labeled=False)
  pi, phi, lambd, mean0, mean1, cov0, cov1 = estimate_params_from_labeled_data()
  print 'Initial estimates from labeled dataset'
  print estimate_model1_params_by_em(x, pi, mean0, mean1, cov0, cov1)
  print 'Initial estimates random'
  pi = np.random.random()
  mean0 = np.random.randn(2)
  mean1 = np.random.randn(2)
  cov0 = np.matrix(np.random.randn(2, 2))
  cov0 = cov0.T * cov0
  cov1 = np.matrix(np.random.randn(2, 2))
  cov1 = cov1.T * cov1
  print estimate_model1_params_by_em(x, pi, mean0, mean1, cov0, cov1)
  print 'Initial estimates random (singular case)'
  pi = np.random.random()
  mean0 = np.random.randn(2)
  mean1 = 1.001 * mean0
  cov0 = np.matrix(np.random.randn(2, 2))
  cov0 = cov0.T * cov0
  cov1 = 1.001 * cov0
  print estimate_model1_params_by_em(x, pi, mean0, mean1, cov0, cov1)

def do_part_b_III():
  section('Part (b) - III - EM model2 estimation from unlabeled data')
  x, _, n, m = read_data('data/unlabeled.txt', labeled=False)
  pi, phi, lambd, mean0, mean1, cov0, cov1 = estimate_params_from_labeled_data()
  print 'Initial estimates from labeled dataset'
  print estimate_model2_params_by_em(x, phi, lambd, mean0, mean1, cov0, cov1)
  print 'Initial estimates random'
  phi = np.random.random()
  lambd = np.random.random()
  mean0 = np.random.randn(2)
  mean1 = np.random.randn(2)
  cov0 = np.matrix(np.random.randn(2, 2))
  cov0 = cov0.T * cov0
  cov1 = np.matrix(np.random.randn(2, 2))
  cov1 = cov1.T * cov1
  print estimate_model2_params_by_em(x, phi, lambd, mean0, mean1, cov0, cov1)
  print 'Initial estimates random (singular case)'
  phi = np.random.random()
  lambd = np.random.random()
  mean0 = np.random.randn(2)
  mean1 = 1.001 * mean0
  cov0 = np.matrix(np.random.randn(2, 2))
  cov0 = cov0.T * cov0
  cov1 = 1.001 * cov0
  print estimate_model2_params_by_em(x, phi, lambd, mean0, mean1, cov0, cov1)

def do_part_b_IV():
  section('Part (b) - IV - EM precinct leanings')
  x, _, n, m = read_data('data/unlabeled.txt', labeled=False)
  pi, phi, lambd, mean0, mean1, cov0, cov1 = estimate_params_from_labeled_data()
  phi, lambd, mean0, mean1, cov0, cov1 = estimate_model2_params_by_em(x, phi, lambd, mean0, mean1, cov0, cov1)
  posterior_y = [estimate_posterior_y_by_x(x[i], phi, lambd, mean0, mean1, cov0, cov1) for i in xrange(n)]
  posterior_z = [estimate_posterior_z_by_x(x[i], phi, lambd, mean0, mean1, cov0, cov1) for i in xrange(n)]
  print_precinct_leanings(posterior_y)
  plot_z_distribution(posterior_z, '/usr/local/google/home/bogatyy/Desktop/part_b_iv_z_table.png')
  plot_individual_inclinations(x, posterior_z, mean0, mean1, '/usr/local/google/home/bogatyy/Desktop/part_b_iv_z_starter.png')


if __name__ == '__main__':
  #do_part_a_I()
  #do_part_a_II()
  #do_part_a_III()
  #do_part_b_I()
  #do_part_b_III()
  do_part_b_IV()

