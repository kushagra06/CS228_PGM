import random
import time

import threading

MODULO = 1000000007
ITERATIONS = int(1e8)
BATCHES = 10
random.seed(0x31337)
INPUT = [random.randint(0, MODULO - 1) for i in xrange(BATCHES)]

results = []

def run_calculation(key):
  value = key
  for iteration in xrange(ITERATIONS):
    value = (3 * value * value + 7 * value + 11) % MODULO
  results.append((key, value))

class CalculationThread(threading.Thread):

  def __init__(self, thread_id, input_key):
    threading.Thread.__init__(self)
    self.thread_id = thread_id
    self.key = input_key

  def run(self):
    print "Starting", self.thread_id
    run_calculation(self.key)
    print "Exiting", self.thread_id


def parallel():
  threads = [CalculationThread(i, INPUT[i]) for i in xrange(len(INPUT))]
  for t in threads:
    t.start()
  for t in threads:
    t.join()

def single_thread():
  for key in INPUT:
    run_calculation(key)

#start_time = time.time()
#single_thread()
#print("--- %s seconds for single thread ---" % (time.time() - start_time))

start_time = time.time()
parallel()
print("--- %s seconds for multi threaded ---" % (time.time() - start_time))
