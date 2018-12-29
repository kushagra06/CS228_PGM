###############################################################################
# utility functions for manipulating factors; ported from Daphne Koller's
# Matlab utility code
# author: Billy Jun, Xiaocheng Li
# date: Jan 31, 2016
# You don't need to modify this file for PA3
###############################################################################

import numpy as np


def assignments_to_indices(A, card):
  """
    :param - A: an assignment
    :param list card: a list of the cardinalities of the variables in the
    assignment
    """
  A = np.array(A, copy=False)
  card = np.array(card, copy=False)
  C = card.flatten()
  if np.any(np.shape(A) == 1):
    I = np.cumprod(np.concatenate(([1.0], C[:0:-1]))) * (A.T).flatten()
  else:
    B = A[:, ::-1]
    I = np.sum(np.tile(np.cumprod(np.concatenate(([1.0], C[:0:-1]))), \
            (B.shape[0], 1)) * B, axis=1)
  return np.array(I, dtype="int32")


def indices_to_assignments(I, card):
  """
    :param - I: a list of indices
    :param list card: a list of the cardinalities of the variables in the
    assignment
    """
  I = np.array(I, copy=False)
  card = np.array(card, copy=False)
  C = card.flatten()
  A = np.mod(np.floor(
          np.tile(I.flatten().T, (len(card), 1)).T / \
          np.tile(np.cumprod(np.concatenate(([1.0], C[:0:-1]))), (len(I), 1))), \
          np.tile(C[::-1], (len(I), 1)))
  return A[:, ::-1]


def intersection_indices(a, b):
  """
    :param list a, b: two lists of variables from different factors.

    returns a tuple of
        (indices in a of the variables in a that are not in b,
        indices of those same variables within the list b)
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
        :param Factor f: if this parameter is not None, then the constructor
        makes a
            copy of it.
        :param list scope: a list of variable names that are in the scope of
        this factor
        :param list card: a list of integers coresponding to the cardinality of
        each variable
            in scope
        :param np.ndarray val: an array coresponding to the values of different
        assignments
            to the factor. val is a numpy.ndarray of shape self.card. Therefore,
            if this factor is over
            three binary variables, self.val will be an array of shape (2,2,2)
        :param str name: the name of the factor.  Useful for debugging only--no
        functional
            purpose.
        """
    assert len(scope) == len(card)

    # self.scope: a list of the variables over which this Factor defines a ddistribution
    self.scope = list(scope)

    # self.card: the cardinality of each variable in self.scope
    self.card = list(card)

    # use the name field for debugging, imo
    self.name = name

    self.val = np.array(val, copy=True)

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
        :param str opname: a string naming the operation.  Optional but nice for
        visualization.

        :rtype: Factor

        --------------------------------------------------------------------------------
        You may find the following functions useful for this implementation:
            -intersection_indices
            -assignments_to_indices
            -indices_to_assignments

        Depending on your implementation, the numpy function np.reshape and the
        numpy.ndarray
        field arr.flat may be useful for this as well, when dealing with the
        duality between
        the two representations of the values of a factor.  (Again, these two
        representations
        are multidimensional array versus vector, and are navigated via the
        functions
        assignments_to_indices and indices_to_assignments)
        """

    g = Factor(
    )  # modify this to be the composition of two Factors and then return it
    g.name = "(%s %s %s)" % (self.name, opname, f.name)
    if len(f.scope) == 0:
      return Factor(self)
    if len(self.scope) == 0:
      return Factor(f)
    g.scope = list(set(self.scope) | set(f.scope))

    # Below regamarole just sets the cardinality of the variables in the scope of g.
    g.card = np.zeros(len(g.scope), dtype="int32")
    _, m1 = intersection_indices(self.scope, g.scope)
    g.card[m1] = self.card
    _, m2 = intersection_indices(f.scope, g.scope)
    g.card[m2] = f.card

    #initialize g's value to zero
    g.val = np.zeros(g.card)

    a = indices_to_assignments(range(np.prod(g.card)), g.card)
    i1 = assignments_to_indices(a[:, m1], self.card)
    i2 = assignments_to_indices(a[:, m2], f.card)
    g.val = np.reshape(operator(self.val.flat[i1], f.val.flat[i2]), g.card)
    return g

  def sum(self, f):
    """
        Returns a factor that is the result of adding this factor with factor f.

        :param Factor f: the factor by which to multiply this factor.
        :rtype: Factor
        """
    return self.compose_factors(f, operator=lambda x, y: x + y, opname="+")

  def multiply(self, f):
    """
        Returns a factor that is the result of multiplying this factor with
        factor f.

        Looking at Factor.sum() might be helpful to implement this function.
        This is
        very simple, but I want to make sure you know how to use lambda
        functions.

        :param Factor f: the factor by which to multiply this factor.
        :rtype: Factor
        """
    return self.compose_factors(f, operator=lambda x, y: x * y, opname="*")

  def marginalize_all_but(self, variables, marginalization_type="sum"):
    """
        returns a factor that is like unto this one except that all variables
        except those
        in the set var have been marginalized out.

        :param set variables: a set of the variables not to be marginalized out.
        :param str marginalization_type: either "sum", signifying
        sum-marginalization,
            or  "max", signifying max-marginalization.
        :rtype: Factor

        --------------------------------------------------------------------------------
        Once you've understood how to navigate our representation in
        compose_factors,
        this implementation shouldn't contain too many surprises.  It is however
        a nontrivial
        amount of code (25 lines by our reckoning)
        """
    assert marginalization_type in ["sum", "max"]
    ops = {"sum": lambda x, y: x + y, "max": lambda x, y: max((x, y))}
    op = ops[marginalization_type]
    g = Factor()
    marginalized_out = ", ".join([str(v) for v in set(self.scope) - set(variables)])
    g.name = "(\%s_{%s} %s)" % (marginalization_type, marginalized_out,
                                self.name[:10] + "...")
    g.scope = list(set(variables) & set(self.scope))
    g.card = [self.card[self.scope.index(variable)] for variable in g.scope]
    g.val = np.tile(0.0, g.card)
    for index in xrange(np.prod(self.card)):
      assignment = indices_to_assignments([index], self.card)[0]
      g_assignment = [assignment[self.scope.index(variable)]
                      for variable in g.scope]
      g_index = assignments_to_indices([g_assignment], g.card)[0]
      g.val.flat[g_index] = op(g.val.flat[g_index], self.val.flat[index])
    return g

  def sumMarginalizeAllBut(self, var):
    if len(self.scope) == 0 or len(var) == 0:
      return Factor(self)
    for v in var:
      if v not in self.scope:
        return Factor()
    g = Factor()
    g.scope = list(var)
    g.card = np.zeros(len(g.scope), dtype="int32")
    ms, mg = intersection_indices(self.scope, g.scope)
    for i, msi in enumerate(ms):
      g.card[mg[i]] = self.card[msi]
    g.val = np.zeros(g.card)
    sa = indices_to_assignments(range(np.prod(self.card)), self.card)
    indxG = assignments_to_indices(sa[:, ms], g.card)
    for i in range(np.prod(self.card)):
      g.val.flat[indxG[i]] += self.val.flat[i]
    return g

  def maxMarginalizeAllBut(self, var):
    if len(self.scope) == 0 or len(var) == 0:
      return Factor(self)
    for v in var:
      if v not in self.scope:
        return Factor()
    g = Factor()
    g.scope = list(var)
    g.card = np.zeros(len(g.scope), dtype="int32")
    ms, mg = intersection_indices(self.scope, g.scope)
    for i, msi in enumerate(ms):
      g.card[mg[i]] = self.card[msi]
    g.val = np.zeros(g.card) + float("-inf")
    sa = indices_to_assignments(range(np.prod(self.card)), self.card)
    indxG = assignments_to_indices(sa[:, ms], g.card)
    for i in range(np.prod(self.card)):
      g.val.flat[indxG[i]] = max(g.val.flat[indxG[i]], self.val.flat[i])
    return g

  def observe(self, var, val):
    """
        Returns a version of this factor with variable var observed as having
        taken on value val.
        if var is not in the scope of this Factor, a duplicate of this factor is
        returned.

        :param str var: the observed variable
        :param int val: the value that variable took on
        :return: a Factor corresponding to this factor with var observed at val

        This will involve zeroing out certain rows/columns, and may involve
        reordering axes.
        """
    f = Factor(self)  #make copy.  You'll modify this.
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
        returns a nice string representing this factor!  Note that we can now
        use string formatting
        with %s and this will cast our class into somethign nice and readable.
        """
    return self.name
