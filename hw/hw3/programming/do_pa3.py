###############################################################################
# Finishes PA 3
# author: Ya Le, Billy Jun, Xiaocheng Li
# date: Jan 25, 2018
###############################################################################

# Utility code for PA3
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import itertools
from factor_graph import *
from factors import *


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
    (We will provide the code using this function, so you don't need to worry too much about it)

    return: image data in matrix form
    '''
    img = sio.loadmat(fname)
    return img[iname]


def applyChannelNoise(y, epsilon):
    '''
    :param y - codeword with 2N entries
    :param epsilon - the probability that each bit is flipped to its complement

    return corrupt message yTilde
    yTilde_i is obtained by flipping y_i with probability epsilon
    '''
    ##########################################################################
    yTilde = np.mod(
        y + np.random.choice([0, 1], size=len(y), p=[1-epsilon, epsilon]
                             ).reshape(y.shape), 2)
    ##########################################################################
    return yTilde


def encodeMessage(x, G):
    '''
    :param - x orginal message
    :param[in] G generator matrix
    :return codeword y=Gx mod 2
    '''
    return np.mod(np.dot(G, x), 2)


def constructFactorGraph(yTilde, H, epsilon):
    '''
    :param - yTilde: observed codeword
        type: numpy.ndarray containing 0's and 1's
        shape: 2N
    :param - H parity check matrix
             type: numpy.ndarray
             shape: N x 2N
    :param epsilon - the probability that each bit is flipped to its complement

    return G factorGraph

    You should consider two kinds of factors:
    - M unary factors
    - N each parity check factors
    '''
    N = H.shape[0]
    M = H.shape[1]
    G = FactorGraph(numVar=M, numFactor=N+M)
    G.var = range(M)
    ##############################################################
    # To do: your code starts here
    # Add unary factors
    factorIndex = 0
    for var in G.var:
        scope = [var]
        card = [2]
        # [P(Y = 0 | Y_t), P(Y = 1 | Y_t)]
        val = np.array([1-epsilon, epsilon] if yTilde[var][0]
                       == 0 else [epsilon, 1 - epsilon])
        G.factors.append(Factor(scope=[var], card=[2],
                                val=val, name="Y"+str(var)))
        G.varToFactor[var].append(factorIndex)
        G.factorToVar[factorIndex].append(var)
        factorIndex += 1

    # Add parity factors
    # You may find the function itertools.product useful
    # (https://docs.python.org/2/library/itertools.html#itertools.product)
    pIndex = 1
    pFactors = []
    for row in H:
        scope = [var for var in G.var if row[var] == 1]
        card = [2 for _ in scope]
        val = np.zeros(tuple(card))
        for prod in itertools.product([0, 1], repeat=len(scope)):
            val[prod] = 1.0 if np.mod(np.sum(prod), 2) == 0 else 0.0
        name = "P"+str(pIndex)
        pIndex += 1
        pFactors.append(Factor(scope=scope, card=card, val=val, name=name))

        G.factorToVar[factorIndex] = scope
        for var in scope:
            G.varToFactor[var].append(factorIndex)
        factorIndex += 1

    G.factors += pFactors
    ##############################################################
    return G


def do_part_a():
    yTilde = np.array([[1, 1, 1, 1, 1, 1]]).reshape(6, 1)
    print("yTilde.shape", yTilde.shape)
    H = np.array([
        [0, 1, 1, 0, 1, 0],
        [0, 1, 0, 1, 1, 0],
        [1, 0, 1, 0, 1, 1]])
    epsilon = 0.05
    G = constructFactorGraph(yTilde, H, epsilon)
    ##############################################################
    # To do: your code starts here
    # Design two invalid codewords ytest1, ytest2 and one valid codewords ytest3.
    #  Report their weights respectively.

    ##############################################################
    ytest1 = np.array([0, 1, 1, 0, 1, 0])
    ytest2 = np.array([1, 0, 1, 1, 0, 1])
    ytest3 = np.array([1, 0, 1, 1, 1, 1])
    print(
        G.evaluateWeight(ytest1),
        G.evaluateWeight(ytest2),
        G.evaluateWeight(ytest3))


def do_part_c():
    '''
    In part b, we provide you an all-zero initialization of message x, you should
    apply noise on y to get yTilde, and then do loopy BP to obatin the
    marginal probabilities of the unobserved y_i's.
    '''
    G, H = loadLDPC('ldpc36-128.mat')

    print(H)
    print(H.shape)
    epsilon = 0.05
    N = G.shape[1]
    x = np.zeros((N, 1), dtype='int32')
    y = encodeMessage(x, G)
    ##############################################################
    # To do: your code starts here
    yTilde = applyChannelNoise(y, epsilon)
    G = constructFactorGraph(yTilde, H, epsilon)
    values = []
    G.runParallelLoopyBP(50)
    for var in G.var:
        values.append(G.estimateMarginalProbability(var)[1])

    plt.figure()
    plt.title("Plot of the estimated posterior probability P(Yi=1|Y~)")
    plt.ylabel("Probability of Bit Being 1")
    plt.xlabel("Bit Index of Received Message")
    plt.bar(range(len(G.var)), values)
    plt.savefig('5c', bbox_inches='tight')
    plt.show()
    plt.close()

    # Verify we get a valid codeword.
    MMAP = G.getMarginalMAP()
    print("The probability of our assignment is %s." % G.evaluateWeight(MMAP))

    ##############################################################


def do_part_de(numTrials, error, iterations=50):
    '''
    param - numTrials: how many trials we repreat the experiments
    param - error: the transmission error probability
    param - iterations: number of Loopy BP iterations we run for each trial
    '''
    G, H = loadLDPC('ldpc36-128.mat')
    ##############################################################
    # To do: your code starts here
    N = G.shape[1]
    x = np.zeros((N, 1), dtype='int32')
    y = encodeMessage(x, G)

    plt.figure()
    plt.title("Plot of the Hamming Distance Between MMAP and True Over 10 "
              "Trials")
    plt.ylabel("Hamming Distance")
    plt.xlabel("Iteration Number of Loopy Belief Propagation")
    for trial in range(10):
        ##############################################################
        # To do: your code starts here
        yTilde = applyChannelNoise(y, error)
        G = constructFactorGraph(yTilde, H, error)
        values = []
        for it in range(iterations):
            G.runParallelLoopyBP(1)
            if it % 10 == 0 and it > 0:
                print("Finished iteration %s of Loopy" % it)
            MMAP = G.getMarginalMAP()
            hamming_distance = np.sum(MMAP)
            values.append(hamming_distance)

        plt.plot(values)

    plt.savefig('5d_epsilon=' + str(int(100*error)), bbox_inches='tight')
    plt.show()
    plt.close()

    ##############################################################


def do_part_fg(error):
    '''
    param - error: the transmission error probability
    '''
    G, H = loadLDPC('ldpc36-1600.mat')
    img = loadImage('images.mat', 'cs242')
    ##############################################################
    # To do: your code starts here
    # You should flattern img first and treat it as the message x in the
    # previous parts.
    original_shape = img.shape
    N = G.shape[1]
    x = img.reshape((N, 1))
    y = encodeMessage(x, G)
    yTilde = applyChannelNoise(y, error)
    G = constructFactorGraph(yTilde, H, error)
    plot_iterations = [0, 1, 2, 3, 5, 10, 20, 30]
    plt.figure()
    for it in range(31):
        G.runParallelLoopyBP(1)
        if it in plot_iterations:
            MMAP = G.getMarginalMAP()
            imgSampled = MMAP[:N].reshape(original_shape)
            i = plot_iterations.index(it)
            plt.subplot(1, 8, i + 1)
            plt.imshow(imgSampled)
            plt.title("Sample at iteration: " + str(it))
    plt.tight_layout()
    plt.savefig('5fg_error=' + str(int(100*error)), bbox_inches='tight')
    plt.show()
    plt.close()

################################################################

print('Doing part (a): Should see 0.0, 0.0, >0.0')
do_part_a()
print('Doing part (c)')
do_part_c()
print('Doing part (d)')
do_part_de(10, 0.06)
print('Doing part (e)')
do_part_de(10, 0.08)
do_part_de(10, 0.10)
print('Doing part (f)')
do_part_fg(0.06)
print('Doing part (g)')
do_part_fg(0.10)
