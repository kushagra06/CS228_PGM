###############################################################################
# utility functions for manipulating factors; ported from Daphne Koller's
# Matlab utility code
# author: Ya Le, Billy Jun, Xiaocheng Li
# date: Jan 25, 2018
# You don't need to modify this file for PA3
###############################################################################


import numpy as np


def assignment_to_indices(A, card):
    """
    :param - A: an assignment
    :param list card: a list of the cardinalities of the variables in the assignment

    Given variables X_1, ..., X_n, let B be the array with dimension (card(X_1), ..., card(X_n)).
    A is an array, each row corresponding to an assignment (x_1, ..., x_n).
    This function returns the indices in B that correspond to assignments of (X_1, ..., X_n) in A.
    For example, assignment_to_indices([[0,1],[1,2]],[2,3]) returns array([1, 5], dtype=int32)

    """
    A = np.array(A, copy=False)
    card = np.array(card, copy=False)
    C = card.flatten()
    if np.any(np.shape(A) == 1):
        I = np.cumprod(np.concatenate(([1.0], C[:0:-1]))) * (A.T).flatten()
    else:
        B = A[:, ::-1]
        I = np.sum(np.tile(np.cumprod(np.concatenate(([1.0], C[:0:-1]))),
                           (B.shape[0], 1)) * B, axis=1)
    return np.array(I, dtype='int32')


def indices_to_assignment(I, card):
    """
    :param - I: a list of indices
    :param list card: a list of the cardinalities of the variables in the assignment
    """
    I = np.array(I, copy=False)
    card = np.array(card, copy=False)
    C = card.flatten()
    A = np.mod(np.floor(
        np.tile(I.flatten().T, (len(card), 1)).T /
        np.tile(np.cumprod(np.concatenate(([1.0], C[:0:-1]))), (len(I), 1))),
        np.tile(C[::-1], (len(I), 1)))
    return A[:, ::-1]


def intersection_indices(a, b):
    """
    :param list a, b: two lists of variables from different factors.

    returns a tuple of 
        (indices in a of the variables that are in both a and b,
        indices of those same variables within the list b)
        For example, intersection_indices([1,2,5,4,6],[3,5,1,2]) returns 
        ([0, 1, 2], [2, 3, 1]).
    """
    bind = {}
    for i, elt in enumerate(b):
        if elt not in bind:
            bind[elt] = i
    mapA = []
    mapB = []
    for i, itm in enumerate(a):
        if itm in bind:
            mapA.append(i)
            mapB.append(bind.get(itm))
    return mapA, mapB


class Factor:

    def __init__(self, f=None, scope=[], card=[], val=None, name="[unnamed]"):
        """
        :param Factor f: if this parameter is not None, then the constructor makes a 
            copy of it.
        :param list scope: a list of variable names that are in the scope of this factor
        :param list card: a list of integers coresponding to the cardinality of each variable
            in scope
        :param np.ndarray val: an array coresponding to the values of different assignments 
            to the factor. val is a numpy.ndarray of shape self.card. Therefore, if this factor is over
            three binary variables, self.val will be an array of shape (2,2,2)
        :param str name: the name of the factor.  Useful for debugging only--no functional
            purpose.
        """
        assert len(scope) == len(card)

        # self.scope: a list of the variables over which this Factor defines a
        # distribution
        self.scope = scope

        # self.card: the cardinality of each variable in self.scope
        self.card = card

        # use the name field for debugging, imo
        self.name = name

        self.val = val

        if f is not None:
            self.scope = list(f.scope)
            self.card = list(f.card)
            self.val = np.array(f.val, copy=True)
            self.name = f.name

    def compose_factors(self, f, operator, opname="op"):
        """
        Returns a factor that is the result of composing this 
        factor under the operator specified by the parameter operator.
        This is a general function that can be used to sum/multiply/etc factors.

        :param Factor f: the factor by which to multiply/sum/etc this factor.
        :param function f: a function taking two arrays and returning a third.
        :param str opname: a string naming the operation.  Optional but nice for visualization.

        :rtype: Factor

        --------------------------------------------------------------------------------
        You may find the following functions useful for this implementation: 
            -intersection_indices
            -assignment_to_indices
            -indices_to_assignment 

        Depending on your implementation, the numpy function np.reshape and the numpy.ndarray 
        field arr.flat may be useful for this as well, when dealing with the duality between 
        the two representations of the values of a factor.  (Again, these two representations
        are multidimensional array versus vector, and are navigated via the functions 
        assignment_to_indices and indices_to_assignment)
        """
        # modify this to be the composition of two Factors and then return it
        g = Factor()
        g.name = "(%s %s %s)" % (self.name, opname, f.name)

        if len(f.scope) == 0:
            return Factor(self)
        if len(self.scope) == 0:
            return Factor(f)
        g.scope = list(set(self.scope) | set(f.scope))

        # Below regamarole just sets the cardinality of the variables in the
        # scope of g.
        g.card = np.zeros(len(g.scope), dtype='int32')
        _, m1 = intersection_indices(self.scope, g.scope)
        g.card[m1] = self.card
        _, m2 = intersection_indices(f.scope, g.scope)
        g.card[m2] = f.card

        # initialize g's value to zero
        g.val = np.zeros(g.card)

        a = indices_to_assignment(range(np.prod(g.card)), g.card)
        i1 = assignment_to_indices(a[:, m1], self.card)
        i2 = assignment_to_indices(a[:, m2], f.card)
        g.val = np.reshape(operator(self.val.flat[i1], f.val.flat[i2]), g.card)
        return g

    def sum(self, f):
        """
        Returns a factor that is the result of adding this factor with factor f.

        :param Factor f: the factor by which to multiply this factor.
        :rtype: Factor
        """
        return self.compose_factors(f, operator=lambda x, y: x+y, opname="+")

    def multiply(self, f):
        """
        Returns a factor that is the result of multiplying this factor with factor f.

        Looking at Factor.sum() might be helpful to implement this function.  This is
        very simple, but I want to make sure you know how to use lambda functions.

        :param Factor f: the factor by which to multiply this factor.
        :rtype: Factor
        """
        return self.compose_factors(f, operator=lambda x, y: x*y, opname="*")

    def divide(self, f):
        """
        Returns a factor that is the result of dividing this factor with factor f.

        Looking at Factor.sum() might be helpful to implement this function.  This is
        very simple, but I want to make sure you know how to use lambda functions.

        :param Factor f: the factor by which to divide this factor.
        :rtype: Factor
        """
        return self.compose_factors(f, operator=lambda x, y: x/y, opname="/")

    def marginalize_all_but(self, var):
        """
        returns a copy of this factor marginalized over all variables except those
        present in var

        Inputs:
        - var (set of ints): indices of variables that will NOT be marginalized over

        Outputs:
        - g (Factor): the new factor marginalized over all variables except those
            present in var

        """
        if len(self.scope) == 0 or len(var) == 0:
            return Factor(self)
        for v in var:
            if v not in self.scope:
                return Factor()
        g = Factor()
        g.scope = list(var)
        g.card = np.zeros(len(g.scope), dtype='int32')
        ms, mg = intersection_indices(self.scope, g.scope)
        for i, msi in enumerate(ms):
            g.card[mg[i]] = self.card[msi]
        g.val = np.zeros(g.card)
        sa = indices_to_assignment(range(np.prod(self.card)), self.card)
        indxG = assignment_to_indices(sa[:, ms], g.card)
        for i in range(np.prod(self.card)):
            g.val.flat[indxG[i]] += self.val.flat[i]
        return g

    def observe(self, var, val):
        """
        Returns a version of this factor with variable var observed as having taken on value val.
        if var is not in the scope of this Factor, a duplicate of this factor is returned.

        :param str var: the observed variable
        :param int val: the value that variable took on
        :return: a Factor corresponding to this factor with var observed at val

        This will involve zeroing out certain rows/columns, and may involve reordering axes.
        """
        f = Factor(self)  # make copy.  You'll modify this.
        f.name = "(%s with variable %s observed as %s)" % (self.name, var, val)
        if var not in self.scope:
            return Factor(self)

        idx = f.scope.index(var)

        order = range(len(f.scope))

        varLoc = f.scope.index(var)
        order[0] = varLoc
        order[varLoc] = 0
        factor = f.val
        permuted = np.transpose(factor, order)
        for j in xrange(f.card[idx]):
            if j != val:
                permuted[j].fill(0.0)
        return f

    def normalize(self):
        """
        Normalize f to a probability distribution
        """
        f = Factor(self)
        f.val /= np.sum(f.val.flatten())
        return f

    def __repr__(self):
        """
        returns a descriptive string representing this factor!
        """
        r = "Factor object with scope %s and corresponding cardinalities %s" % (
            self.scope, self.card)
        r += "\nCPD:\n" + str(self.val)
        if self.name:
            r = "Factor %s:\n" % self.name + r
        return r + "\n"

    def __str__(self):
        """
        returns a nice string representing this factor!  Note that we can now use string formatting
        with %s and this will cast our class into somethign nice and readable.
        """
        return self.name
