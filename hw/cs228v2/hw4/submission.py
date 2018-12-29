# Gibbs sampling algorithm to denoise an image
# Author : Gunaa AV, Isaac Caswell
# Date : 2/18/2015

import math
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

BETA = 1.0
ETA = 1.0
MAX_BURNS = 100
MAX_SAMPLES = 1000


def find_energy(x, y):
  assert len(x) == len(y) and len(x[0]) == len(y[0])
  energy = 0.0
  for i in xrange(len(x)):
    for j in xrange(len(y)):
      energy += ETA * x[i][j] * y[i][j]
      if i + 1 < len(x):
        energy += BETA * y[i][j] * y[i + 1][j]
      if j + 1 < len(x[0]):
        energy += BETA * y[i][j] * y[i][j + 1]
  return energy

def count_pixels_in_Z_area(y):
  count = 0
  for i in xrange(125, 162 + 1):
    for j in xrange(143, 174 + 1):
      if y[i][j] > 0:
        count += 1
  return count

def maybe_flip_one_pixel(x, y, i, j):
  alpha = ETA * x[i][j]
  indices = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
  for ii, jj in indices:
    if ii >= 0 and jj >= 0 and ii < len(x) and jj < len(x[0]):
      alpha += BETA * y[ii][jj]
  probability = 1.0 / (1.0 + math.exp(-2.0 * alpha))
  u = random.random()
  if u < probability:
    y[i][j] = 1
  else:
    y[i][j] = -1


def run_gibbs_iteration(x, y):
  for i in xrange(len(x)):
    for j in xrange(len(x[0])):
      maybe_flip_one_pixel(x, y, i, j)


def get_posterior_by_sampling(filename,
                              initialization='same',
                              logfile=None):
  assert initialization in ['same', 'neg', 'rand']
  x = read_txt_file(filename)
  if initialization == 'same':
    y = np.array(x, copy=True)
  elif initialization == 'neg':
    y = 1.0 - x
  else:
    y = 2 * np.random.randint(2, size=np.shape(x)) - 1

  if logfile:
    log_writer = open(logfile, 'w')
  for iteration in xrange(MAX_BURNS):
    run_gibbs_iteration(x, y)
    energy = find_energy(x, y)
    if logfile:
      log_writer.write('{0}\t{1}\t{2}\n'.format(1 + iteration, energy, 'B'))

  posterior = np.tile(0.0, np.shape(x))
  z_values = []
  for iteration in xrange(MAX_SAMPLES):
    run_gibbs_iteration(x, y)
    for i in xrange(len(x)):
      for j in xrange(len(x[0])):
        if y[i][j] > 0:
          posterior[i][j] += 1.0 / MAX_SAMPLES
    z_values.append(count_pixels_in_Z_area(y))
    energy = find_energy(x, y)
    if logfile:
      log_writer.write('{0}\t{1}\t{2}\n'.format(1 + MAX_BURNS + iteration,
                                                energy, 'S'))

  if logfile:
    log_writer.close()

  return posterior, z_values

def denoise_from_posterior(posterior):
  denoised = np.tile(0, posterior.shape)
  for i in xrange(len(posterior)):
    for j in xrange(len(posterior[0])):
      denoised[i][j] = 1 if posterior[i][j] > 0.5 else -1
  return denoised

def denoise_by_majority_voting(x):
  y = np.array(x, copy=True)
  for iteration in xrange(30):
    for i in xrange(len(x)):
      for j in xrange(len(x[0])):
        votes_positive = 1 if x[i][j] > 0 else 0
        votes_total = 1
        indices = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
        for ii, jj in indices:
          if ii >= 0 and jj >= 0 and ii < len(x) and jj < len(x[0]):
            if y[ii][jj] > 0:
              votes_positive += 1
            votes_total += 1
        if votes_positive > votes_total - votes_positive:
          y[i][j] = 1
        else:
          y[i][j] = -1
  return y

# ===========================================
# Helper functions for plotting etc
# ===========================================


def plot_energy(logfile):
  '''
    logfile: a file with energy log, each row should have three terms separated
    by a \t:
        iteration: iteration number
        energy: the energy at this iteration
        S or B: indicates whether it's burning in or a sample
    e.g.
        1   -202086.0   B
        2   -210446.0   B
        ...
  '''
  its_burn, energies_burn = [], []
  its_sample, energies_sample = [], []
  with open(logfile, 'r') as f:
    for line in f:
      it, en, phase = line.strip().split()
      if phase == 'B':
        its_burn.append(it)
        energies_burn.append(en)
      elif phase == 'S':
        its_sample.append(it)
        energies_sample.append(en)
      else:
        print 'bad phase: -%s-' % phase

  plt.clf()
  p1, = plt.plot(its_burn, energies_burn, 'r')
  p2, = plt.plot(its_sample, energies_sample, 'b')
  plt.title(logfile)
  plt.legend([p1, p2], ['burn in', 'sampling'])
  plot_file = logfile[:logfile.rfind('.')] + '.png'
  plt.savefig(plot_file)


def save_three_images(original, noisy, denoised, output_filename):
  assert original.shape == noisy.shape and noisy.shape == denoised.shape
  BARRIER = 5
  n = len(original)
  m = len(original[0])
  errors_before = 0.0
  errors_after = 0.0
  for i in xrange(n):
    for j in xrange(m):
      if original[i][j] != noisy[i][j]:
        errors_before += 1.0 / (n * m)
      if original[i][j] != denoised[i][j]:
        errors_after += 1.0 / (n * m)
  print 'Comparison for {0}: error rate before {1}, error rate after {2}'.format(
      output_filename, errors_before, errors_after)
  concat = [[] for i in xrange(n)]
  for i in xrange(n):
    for j in xrange(m):
      concat[i].append(original[i][j])
    for j in xrange(BARRIER):
      concat[i].append(1.0)
    for j in xrange(m):
      concat[i].append(noisy[i][j])
    for j in xrange(BARRIER):
      concat[i].append(1.0)
    for j in xrange(m):
      concat[i].append(denoised[i][j])
  concat = 1.0 - np.array(concat) # actually black is zero
  channeled = np.reshape(np.repeat(concat, 3), concat.shape + (3,))
  plt.clf()
  plt.imshow(channeled)
  plt.savefig(output_filename)

def plot_histogram(z_values, output_filename):
  plt.clf()
  plt.title(output_filename)
  plt.hist(z_values)
  plt.savefig(output_filename)

def read_txt_file(filename):
  '''
    filename: image filename in txt
    return:   2-d array image
  '''
  f = open(filename, 'r')
  lines = f.readlines()
  height = int(lines[0].split()[1].split('=')[1])
  width = int(lines[0].split()[2].split('=')[1])
  x = [[0 for j in xrange(width)] for i in xrange(height)]
  for line in lines[2:]:
    i, j, val = [int(entry) for entry in line.split()]
    x[i][j] = val
  return np.array(x)


random.seed(0x31337)
np.random.seed(0x31337)
#==================================
# doing part (c)
#==================================
print 'Doing part (c)'
posterior20, z_values20 = get_posterior_by_sampling('noisy_20.txt', 'same', logfile='part_c_same.txt')
plot_energy('part_c_same.txt')
get_posterior_by_sampling('noisy_20.txt', 'neg', logfile='part_c_neg.txt')
plot_energy('part_c_neg.txt')
get_posterior_by_sampling('noisy_20.txt', 'rand', logfile='part_c_rand.txt')
plot_energy('part_c_rand.txt')

#==================================
# doing part (d)
#==================================
original = read_txt_file('orig.txt')
noisy10 = read_txt_file('noisy_10.txt')
noisy20 = read_txt_file('noisy_20.txt')
print 'Doing part (d)'
posterior10, z_values10 = get_posterior_by_sampling('noisy_10.txt', 'same')
denoised10 = denoise_from_posterior(posterior10)
save_three_images(original, noisy10, denoised10, 'part_d_comparison_10.png')

denoised20 = denoise_from_posterior(posterior20)
save_three_images(original, noisy20, denoised20, 'part_d_comparison_20.png')

#==================================
# doing part (e)
#==================================
print 'Doing part (e)'
majority10 = denoise_by_majority_voting(noisy10)
save_three_images(original, noisy10, majority10, 'part_e_comparison_10.png')

majority20 = denoise_by_majority_voting(noisy20)
save_three_images(original, noisy20, majority20, 'part_e_comparison_20.png')

#==================================
# doing part (f)
#==================================
print 'Doing part (f)'
print 'Z values per iteration, 10% error rate'
print z_values10
print 'Z values per iteration, 20% error rate'
print z_values20
plot_histogram(z_values10, 'part_f_histogram10.png')
plot_histogram(z_values20, 'part_f_histogram20.png')
