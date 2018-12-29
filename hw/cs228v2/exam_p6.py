import random
import math

EPS = 1e-10
ITERATIONS = 20
SAMPLE_SIZE = 100
iteration = 123213 # this has to be a global variable

def normalize(p):
  if p < EPS:
    return EPS
  if p > 1.0 - EPS:
    return 1.0 - EPS
  return p

def generate_input():
  theta = random.random()
  p = random.random()
  q = random.random()
  cH = 0
  cT = 0
  cM = 0
  for coin_toss in xrange(SAMPLE_SIZE):
    entropy1 = random.random()
    entropy2 = random.random()
    if entropy1 < theta:
      if entropy2 < p:
        cH += 1
      else:
        cT += 1
    else:
      if entropy2 < q * q:
        cH += 1
      elif entropy2 < q * q + 2 * q * (1 - q):
        cM += 1
      else:
        cT += 1
  return theta, p, q, cH, cM, cT

def calculate_log_likelihood(theta, p, q, cH, cM, cT):
  return (cH * math.log(theta * p + (1 - theta) * q * q) if cH > 0 else 0.0) + \
         (cM * math.log((1 - theta) * 2 * q * (1 - q)) if cM > 0 else 0.0)+ \
         (cT * math.log(theta * (1 - p) + (1 - theta) * (1 - q) * (1 - q)) if cT > 0 else 0.0)

def calculate_precise_solution(cH, cM, cT):
  norm = 1.0 * (cH + cM + cT)
  q = 0.5 # because why not
  theta = 1.0 - cM / norm / (2 * q * (1 - q))
  p = (cH / norm - (1 - theta) * q * q) / theta
  return theta, p, q

def solve_by_joint_gradient_ascent(cH, cM, cT):
  evaluate = lambda theta, p, q: calculate_log_likelihood(theta, p, q, cH, cM, cT)
  random.seed(iteration) # initialize gradient ascent and EM the same way
  theta = random.random()
  p = random.random()
  q = random.random()
  lr = 1.0 / 1024
  for gradient_ascent_iteration in xrange(100):
    dt = cH * 1.0 / (theta * p + (1 - theta) * q * q) * (p - q * q) + \
         cM * 1.0 / ((1 - theta) * 2 * q * (1 - q)) * 2 * q * (q - 1) + \
         cT * 1.0 / (theta * (1 - p) + (1 - theta) * (1 - q) * (1 - q)) * (1 - p - (1 - q) * (1 - q))
    dp = cH * 1.0 / (theta * p + (1 - theta) * q * q)  * theta + \
         cT * 1.0 / (theta * (1 - p) + (1 - theta) * (1 - q) * (1 - q)) * -theta
    dq = cH * 1.0 / (theta * p + (1 - theta) * q * q) * (1 - theta) * 2 * q + \
         cM * 1.0 / ((1 - theta) * 2 * q * (1 - q)) * (1 - theta) * (2 - 4 * q) + \
         cT * 1.0 / (theta * (1 - p) + (1 - theta) * (1 - q) * (1 - q)) * (1 - theta) * 2 * (q - 1)
    while True:
      new_theta = normalize(theta + lr * dt)
      new_p = normalize(p + lr * dp)
      new_q = normalize(q + lr * dq)
      if evaluate(new_theta, new_p, new_q) > evaluate(theta, p, q) or lr < EPS:
        break
      else:
        lr *= 0.5
    theta = new_theta
    p = new_p
    q = new_q
  return theta, p, q

def solve_by_EM(cH, cM, cT):
  random.seed(iteration) # initialize gradient ascent and EM the same way
  theta = random.random()
  p = random.random()
  q = random.random()
  for em_iteration in xrange(20):
    # E step
    pH1 = theta * p
    pH2 = (1 - theta) * q * q
    cH1 = cH * pH1 / (pH1 + pH2)
    cH2 = cH * pH2 / (pH1 + pH2)
    pT1 = theta * (1 - p)
    pT2 = (1 - theta) * (1 - q) * (1 - q)
    cT1 = cT * pT1 / (pT1 + pT2)
    cT2 = cT * pT2 / (pT1 + pT2)
    # M step
    p = cH1 / (cH1 + cT1) if cH1 + cT1 > 0 else 0.5
    q = normalize( (2.0 * cH2 + cM) / (2 * cT2 + 2 * cM + 2 * cH2) )
    theta = (cH1 + cT1) / (cH + cM + cT)
  return theta, p, q

def test_three_algorithms(cH, cM, cT):
  print 'CLOSED_FORM ',
  theta, p, q = calculate_precise_solution(cH, cM, cT)
  theta = normalize(theta)
  print calculate_log_likelihood(theta, p, q, cH, cM, cT), theta, p, q
  print 'GRAD_ASCENT ',
  theta, p, q = solve_by_joint_gradient_ascent(cH, cM, cT)
  print calculate_log_likelihood(theta, p, q, cH, cM, cT), theta, p, q
  print 'EM_SOLUTION ',
  theta, p, q = solve_by_EM(cH, cM, cT)
  print calculate_log_likelihood(theta, p, q, cH, cM, cT), theta, p, q

def run_testing_iterations():
  random.seed(0x3133)
  for iteration in xrange(ITERATIONS):
    theta, p, q, cH, cM, cT = generate_input()
    print '---Iteration', iteration
    print 'DATA        ', cH, cM, cT
    print 'CORRECT     ', calculate_log_likelihood(theta, p, q, cH, cM, cT), theta, p, q
    test_three_algrothims(cH, cM, cT)
    print '---\n'

if __name__ == '__main__':
  #run_testing_iterations()
  test_three_algorithms(1, 10, 2)
