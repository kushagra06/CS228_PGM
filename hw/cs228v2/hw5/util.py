import numpy as np
import matplotlib.pyplot as plt


def resize_array(a, n, m):
  while len(a) < n:
    a.append([])
  for i in xrange(len(a)):
    while len(a[i]) < m:
      a[i].append(None)


def read_data(filename, labeled=True):
  z = []
  x = []
  n, m = 0, 0
  with open(filename, 'r') as f:
    lines = f.readlines()
    for line in lines[1:]:
      if labeled:
        i, j, z_value, x1_value, x2_value = line.split()
      else:
        i, j, x1_value, x2_value = line.split()
        z_value = '-1'
      i, j = int(i) - 1, int(j) - 1
      n = max(n, i + 1)
      m = max(m, j + 1)
      resize_array(z, n, m)
      resize_array(x, n, m)
      z[i][j] = float(z_value)
      x[i][j] = np.array([float(x1_value), float(x2_value)])
  return x, z, n, m


def section(name):
  print '-' * 40
  print name
  print '-' * 40


def plot_z_distribution(probabilities_z, plot_filename):
  plt.clf()
  x = []
  y = []
  colors = []
  for i in xrange(len(probabilities_z)):
    for j in xrange(len(probabilities_z[i])):
      x.append(i)
      y.append(j)
      colors.append('r' if probabilities_z[i][j] > 0.5 else 'b')
  plt.scatter(x, y, c=colors)
  plt.savefig(plot_filename)

def plot_individual_inclinations(x, probabilities_z, mean0, mean1, plot_filename):
  plt.clf()
  domain0 = []
  range0 = []
  domain1 = []
  range1 = []
  for i in xrange(len(x)):
    for j in xrange(len(x[i])):
      if probabilities_z[i][j] < 0.5:
        domain0.append(x[i][j][0])
        range0.append(x[i][j][1])
      else:
        domain1.append(x[i][j][0])
        range1.append(x[i][j][1])

  plt.plot(domain0, range0, 'b+')
  plt.plot(domain1, range1, 'r+')
  p1, = plt.plot(mean0[0], mean0[1], 'kd')
  p2, = plt.plot(mean1[0], mean1[1], 'kd')
  plt.savefig(plot_filename)

def print_precinct_leanings(posterior_y):
  for i in xrange(len(posterior_y)):
    print 'Precinct {0}\tP={1}\t{2}'.format(i, posterior_y[i], '+++' if posterior_y[i] > 0.5 else '---')
