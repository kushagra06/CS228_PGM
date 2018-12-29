# Gibbs sampling algorithm to denoise an image
# Author : Gunaa AV, Isaac Caswell
# Edits : Bo Wang, Kratarth Goel, Aditya Grover
# Date : 2/17/2017

import math
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict


def get_energy(Y, X):
  '''
  Returns the enery of this state.
  '''
  t = 0.0
  N, M = Y.shape
  # Make sure we only count each edge once!
  return -1*(np.sum(X*Y) + np.sum(Y[:N-1, :]*Y[1:, :])
             + np.sum(Y[:, :M-1]*Y[:, 1:]))


def markov_blanket(i, j, Y, X):
  '''
  return:
      the a list of Y values that are markov blanket of Y[i][j]
      e.g. if i = j = 1,
          the function should return [Y[0][1], Y[1][0], Y[1][2], Y[2][1],
          X[1][1]]
  '''
  return [Y[i-1][j], Y[i][j-1], Y[i][j+1], Y[i+1][j], X[i][j]]


def sampling_prob(markov_blanket):
  '''
  markov_blanket: a list of the values of a variable's Markov blanket
      The order doesn't matter (see part (a)). e.g. [1,1,-1,1]
  return:
       a real value which is the probability of a variable being 1 given its
       Markov blanket
  '''
  ########
  x = 2*np.sum(markov_blanket)
  return np.exp(x - np.log(1 + np.exp(x)))


def sample(i, j, Y, X, DUMB_SAMPLE=0):
  '''
  return a new sampled value of Y[i][j]
  It should be sampled by
      (i) the probability condition on all the other variables if
        DUMB_SAMPLE = 0
      (ii) the consensus of Markov blanket if DUMB_SAMPLE = 1
  '''
  blanket = markov_blanket(i, j, Y, X)

  if not DUMB_SAMPLE:
    prob = sampling_prob(blanket)
    if random.random() < prob:
      return 1
    else:
      return -1
  else:
    c_b = Counter(blanket)
    if c_b[1] >= c_b[-1]:
      return 1
    else:
      return -1


def get_posterior_by_sampling(filename, initialization='same', logfile=None,
                              DUMB_SAMPLE=0):
  '''
  Do Gibbs sampling and compute the energy of each assignment for the image
  specified in filename.
  If not dumb_sample, it should run MAX_BURNS iterations of burn in and then
  MAX_SAMPLES iterations for collecting samples.
  If dumb_sample, run MAX_SAMPLES iterations and returns the final image.

  filename: file name of image in txt
  initialization: 'same' or 'neg' or 'rand'
  logfile: the file name that stores the energy log (will use for plotting
      later) look at the explanation of plot_energy to see detail
  DUMB_SAMPLE: equals 1 if we want to use the trivial reconstruction in part (e)

  return value: posterior, Y, frequencyZ
      posterior: an 2d-array with the same size of Y, the value of each entry
          should be the probability of that being 1 (estimated by the Gibbs
          sampler)
      Y: The final image (for DUMB_SAMPLE = 1, in part (e))
      frequencyZ: a dictionary with key: count the number of 1's in the Z region
                                    value: frequency of such count
  '''
  print("Read the file")
  Y = read_txt_file(filename)
  X = np.copy(Y)

  N, M = Y.shape
  if initialization == 'neg':
    Y = -Y
  if initialization == 'rand':
    Y[np.where(Y != 0)] = np.random.choice([1, -1], size=((N-2)*(M-2)))
  MAX_BURNS = 100
  MAX_SAMPLES = 1000

  def perform_sampling(log_fn):
    t = 0
    if not DUMB_SAMPLE:
      for _ in range(MAX_BURNS):
        for i in range(1, N-1):
          for j in range(1, M-1):
            Y[i][j] = sample(i, j, Y, X, DUMB_SAMPLE)

        t += 1
        if t % 10 == 0:
          print("Completed burn-in sample %s" % t)
        log_fn(t, "B", Y)

    counts = np.zeros((N, M))
    frequencyZ_counts = []
    for _ in range(MAX_SAMPLES):
      count_z = 0
      sampled = np.zeros(Y.shape)
      for i in range(1, N-1):
        for j in range(1, M-1):
          sampled[i][j] = sample(i, j, Y, X, DUMB_SAMPLE)
          if DUMB_SAMPLE:
            Y[i][j] = sampled[i][j]
      if not DUMB_SAMPLE:
        is_one_in_sample = (sampled == 1)
        counts += (is_one_in_sample)
        count_z = np.sum(is_one_in_sample[125:163, 143:175])
      frequencyZ_counts.append(count_z)
      t += 1
      if t % 10 == 0:
        print("Completed sample %s from posterior" %
              (t - (0 if DUMB_SAMPLE else MAX_BURNS)))
      log_fn(t, "S", sampled)

    return counts, frequencyZ_counts

  if logfile is not None:
    with open(logfile, 'w') as log:
      counts, frequencyZ_counts = perform_sampling(
          lambda i, typ, Ys: log.write("\t".join(map(
              str, [i, get_energy(Ys, X), typ])) + "\n"))
  else:
    counts, frequencyZ_counts = perform_sampling(lambda *args: None)

  return counts / float(MAX_SAMPLES), Y, Counter(frequencyZ_counts)


def denoise_image(filename, initialization='rand', logfile=None, DUMB_SAMPLE=0):
  '''
  Do Gibbs sampling on the image and return the denoised one and frequencyZ
  '''
  posterior, Y, frequencyZ = \
      get_posterior_by_sampling(
          filename, initialization, logfile=logfile, DUMB_SAMPLE=DUMB_SAMPLE)

  N, M = Y.shape
  if DUMB_SAMPLE:
    for i in range(N):
      for j in range(M):
        Y[i][j] = .5*(1.0-Y[i][j])  # 1, -1 --> 1, 0
    return Y, frequencyZ
  else:
    denoised = np.zeros(posterior.shape)
    denoised[np.where(posterior < .5)] = 1
    return denoised, frequencyZ


# ===========================================
# Helper functions for plotting etc
# ===========================================

def plot_energy(filename):
  '''
  filename: a file with energy log, each row should have three terms separated 
    by a \t:
      iteration: iteration number
      energy: the energy at this iteration
      S or B: indicates whether it's burning in or a sample
  e.g.
      1   -202086.0   B
      2   -210446.0   S
      ...
  '''
  its_burn, energies_burn = [], []
  its_sample, energies_sample = [], []
  with open(filename, 'r') as f:
    for line in f:
      it, en, phase = line.strip().split()
      if phase == 'B':
        its_burn.append(float(it))
        energies_burn.append(float(en))
      elif phase == 'S':
        its_sample.append(float(it))
        energies_sample.append(float(en))
      else:
        print("bad phase: -%s-" % phase)

  p1, = plt.plot(its_burn, energies_burn, 'r')
  p2, = plt.plot(its_sample, energies_sample, 'b')
  plt.title(filename)
  plt.legend([p1, p2], ["burn in", "sampling"])
  plt.savefig(filename)
  plt.close()


def read_txt_file(filename):
  '''
  filename: image filename in txt
  return:   2-d array image
  '''
  f = open(filename, "r")
  lines = f.readlines()
  height = int(lines[0].split()[1].split("=")[1])
  width = int(lines[0].split()[2].split("=")[1])
  Y = [[0]*(width+2) for i in range(height+2)]
  for line in lines[2:]:
    i, j, val = [int(entry) for entry in line.split()]
    Y[i+1][j+1] = val
  return np.array(Y)


def convert_to_png(denoised_image, title):
  '''
  save array as a png figure with given title.
  '''
  plt.imshow(denoised_image, cmap=plt.cm.gray)
  plt.title(title)
  plt.savefig(title + '.png')


def get_error(img_a, img_b):
  '''
  compute the fraction of all pixels that differ between the two input images.
  '''
  N = len(img_b[0])*len(img_b)*1.0
  return sum([sum([1 if img_a[row][col] != img_b[row][col] else 0
                   for col in range(len(img_a[0]))])
              for row in range(len(img_a))]
             ) / N


#==================================
# doing part (c), (d), (e), (f)
#==================================

def perform_part_c():
  '''
  Run denoise_image function with different initialization and plot out the
  energy functions.
  '''
  global denoised_20, frequencyZ_20

  denoise_image("noisy_20.txt", initialization='rand',
                logfile='log_rand', DUMB_SAMPLE=0)
  denoise_image("noisy_20.txt", initialization='neg',
                logfile='log_neg', DUMB_SAMPLE=0)
  denoised_20, frequencyZ_20 = denoise_image("noisy_20.txt",
                                             initialization='same',
                                             logfile='log_same', DUMB_SAMPLE=0)

  # plot out the energy functions
  plot_energy("log_rand")
  plot_energy("log_neg")
  plot_energy("log_same")


def perform_part_d():
  '''
  Run denoise_image function with different noise levels of 10% and 20%, and
  report the errors between denoised images and original image
  '''
  global denoised_10, frequencyZ_10
  denoised_10, frequencyZ_10 = denoise_image("noisy_10.txt",
                                             initialization='same',
                                             logfile=None, DUMB_SAMPLE=0)
  orig_img = read_txt_file("orig.txt")
  orig_img = .5 * (1 - orig_img)

  # save denoised images and original image to png figures
  convert_to_png(denoised_10, "denoised_10")
  convert_to_png(denoised_20, "denoised_20")
  convert_to_png(orig_img, "orig_img")

  N, M = orig_img.shape
  print("restoration error from %s noise: %s." %
        (10, np.sum((orig_img != denoised_10)[1:N-1, 1:M-1]) / float(
            (N-1) * (M-1))))
  print("restoration error from %s noise: %s." %
        (20, np.sum((orig_img != denoised_20)[1:N-1, 1:M-1]) / float(
            (N-1) * (M-1))))


def perform_part_e():
  '''
  Run denoise_image function using dumb sampling with different noise levels of
  10% and 20%.
  '''
  orig_img = read_txt_file("orig.txt")
  orig_img = .5 * (1 - orig_img)

  denoised_dumb_10, _ = denoise_image("noisy_10.txt", initialization='same',
                                      logfile=None, DUMB_SAMPLE=1)
  denoised_dumb_20, _ = denoise_image("noisy_20.txt", initialization='same',
                                      logfile=None, DUMB_SAMPLE=1)

  # save denoised images to png figures
  convert_to_png(denoised_dumb_10, "denoised_dumb_10")
  convert_to_png(denoised_dumb_20, "denoised_dumb_20")

  N, M = orig_img.shape
  print("restoration error from %s%% noise: %s." %
        (10, np.sum((orig_img != denoised_dumb_10)[1:N-1, 1:M-1]) / float(
            (N-1) * (M-1))))
  print("restoration error from %s%% noise: %s." %
        (20, np.sum((orig_img != denoised_dumb_20)[1:N-1, 1:M-1]) / float(
            (N-1) * (M-1))))


def perform_part_f():
  '''
  Run Z square analysis
  '''

  # d, f = denoise_image('noisy_10.txt',
  #                     initialization='same', logfile=None)
  d, f = denoised_10, frequencyZ_10
  width = 1.0
  plt.clf()
  plt.bar(f.keys(), f.values(), width, color='b')
  plt.savefig('f_10' + '.png')
  plt.show()
  # d, f = denoise_image('noisy_20.txt',
  #                     initialization='same', logfile=None)
  d, f = denoised_20, frequencyZ_20
  plt.clf()
  plt.bar(f.keys(), f.values(), width, color='b')
  plt.savefig('f_20' + '.png')
  plt.show()


if __name__ == "__main__":
  perform_part_c()
  perform_part_d()
  perform_part_e()
  perform_part_f()
