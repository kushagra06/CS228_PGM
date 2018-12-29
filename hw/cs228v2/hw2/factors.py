###############################################################################
# utility functions for manipulating factors; ported from Daphne Koller's
# Matlab utility code
# author: Billy Jun
# date: Jan 20, 2015
###############################################################################
#
# an assignment look like the following, for a 4-valued variable given its four-valued parent:
#  [[ 0.  0.]
#  [ 0.  1.]
#  [ 0.  2.]
#  [ 0.  3.]
#  [ 1.  0.]
#  [ 1.  1.]
#  [ 1.  2.]
#  [ 1.  3.]
#  [ 2.  0.]
#  [ 2.  1.]
#  [ 2.  2.]
#  [ 2.  3.]
#  [ 3.  0.]
#  [ 3.  1.]
#  [ 3.  2.]
#  [ 3.  3.]]

import numpy as np


def assignment_to_indices(A, card):
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


def indices_to_assignment(I, card):
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

# used to print factor names
def shorten(string):
  if len(string) > 40:
    return string[:32] + "..." + string[-5:]
  else:
    return string

class Factor:

  def __init__(self, f=None, scope=[], card=[], val=None, name="[unnamed]", logfactor=False):
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

    if f is not None:
      self.scope = list(f.scope)
      self.card = list(f.card)
      self.val = np.array(f.val, copy=True)
      self.name = f.name
      self.logfactor = f.logfactor
      return

    assert len(scope) == len(card)

    # self.scope: a list of the variables over which this Factor defines a ddistribution
    self.scope = scope

    # self.card: the cardinality of each variable in self.scope
    self.card = card

    # factor table
    self.val = np.array(val, copy=True)

    # use the name field for debugging, imo
    self.name = name

    # this means we store and process this factor in log-space
    self.logfactor = logfactor
    if self.logfactor:
      self.val = np.log(self.val)

  def compose_factors(self, other, operator, opname="op"):
    """
        Returns a factor that is the result of composing this
        factor under the operator specified by the parameter operator.
        This is a general function that can be used to sum/multiply/etc factors.

        :param Factor other: the factor by which to multiply/sum/etc this
        factor.
        :param function operator: a function taking two arrays and returning a
        third.
        :param str opname: a string naming the operation.  Optional but nice for
        visualization.

        :rtype: Factor

        --------------------------------------------------------------------------------
        You may find the following functions useful for this implementation:
            -intersection_indices
            -assignment_to_indices
            -indices_to_assignment

        Depending on your implementation, the numpy function np.reshape and the
        numpy.ndarray
        field arr.flat may be useful for this as well, when dealing with the
        duality between
        the two representations of the values of a factor.  (Again, these two
        representations
        are multidimensional array versus vector, and are navigated via the
        functions
        assignment_to_indices and indices_to_assignment)
        """

    assert self.logfactor == other.logfactor

    combination = Factor(
    )  # modify this to be the composition of two Factors and then return it
    combination.name = "(%s %s %s)" % (self.name, opname, other.name)
    other_not_in_self_indices = [other.scope.index(var)
                                 for var in set(other.scope) - set(self.scope)]
    combination.scope = self.scope + [other.scope[i]
                                      for i in other_not_in_self_indices]
    combination.card = self.card + [other.card[i]
                                    for i in other_not_in_self_indices]
    combination.val = np.tile(0.0, combination.card)
    for index in xrange(np.prod(combination.card)):
      assignment = indices_to_assignment([index], combination.card)[0]
      assignment_self = [assignment[combination.scope.index(self_var)]
                         for self_var in self.scope]
      assignment_other = [assignment[combination.scope.index(other_var)]
                          for other_var in other.scope]
      index_self = assignment_to_indices([assignment_self], self.card)[0]
      index_other = assignment_to_indices([assignment_other], other.card)[0]
      combination.val.flat[index] = operator(self.val.flat[index_self],
                                             other.val.flat[index_other])

    combination.logfactor = self.logfactor
    return combination

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

  def combine(self, other):
    if self.logfactor != other.logfactor:
      print repr(self)
      print repr(other)
    assert self.logfactor == other.logfactor
    if not self.logfactor:
      return self.multiply(other)
    else:
      return self.sum(other)

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
    self_val = self.val if not self.logfactor else np.exp(self.val)
    g = Factor()
    marginalized_out = ", ".join([str(v) for v in set(self.scope) - set(variables)])
    g.name = "(\%s_{%s} %s)" % (marginalization_type, marginalized_out,
                                self.name)
    g.scope = list(set(variables) & set(self.scope))
    g.card = [self.card[self.scope.index(variable)] for variable in g.scope]
    g.val = np.tile(0.0, g.card)
    for index in xrange(np.prod(self.card)):
      assignment = indices_to_assignment([index], self.card)[0]
      g_assignment = [assignment[self.scope.index(variable)]
                      for variable in g.scope]
      g_index = assignment_to_indices([g_assignment], g.card)[0]
      g.val.flat[g_index] = op(g.val.flat[g_index], self_val.flat[index])
    if self.logfactor:
      g.logfactor = True
      if g.card:
        assert min(g.val) >= 0.0
      else:
        assert g.val >= 0.0
      g.val = np.log(g.val + 1e-50)
    return g

  def observe_variable(self, var, val):
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
    f = Factor(self)
    f.name = "(%s with variable %s observed as %s)" % (self.name, var, val)
    if var not in self.scope:
      return f
    for index in xrange(np.prod(f.card)):
      assignment = indices_to_assignment([index], f.card)[0]
      if assignment[f.scope.index(var)] != val:
        f.val.flat[index] = 0.0 if not f.logfactor else np.min(f.val) - 1e2
    return f

  # evidence is a dict of the form {val_idx -> var}
  def observe(self, evidence):
    observed = self
    for (var, val) in evidence.items():
      observed = observed.observe_variable(var, val)
    return observed

  def __repr__(self):
    """
        returns a descriptive string representing this factor!
        """
    r = "Factor object with scope %s and corresponding cardinalities %s" % (
        self.scope, self.card)
    r += "\nCPD:\n" + str(self.val)
    if self.name:
      r = "Factor %s:\n" % shorten(self.name) + r
    if self.logfactor:
      r = "LOGFACTOR: " + r
    return r + "\n"

  def __str__(self):
    """
        returns a nice string representing this factor!  Note that we can now
        use string formatting
        with %s and this will cast our class into somethign nice and readable.
        """
    return self.name

#===============================================================================
# Testing script

if __name__ == "__main__":
  """
    let's define a Markov chain to sanity check your implementation:
    """
  np.random.seed(4)
  cards = [2, 2, 3]
  var_names = ["A", "B", "C"]

  # P(A|B)
  f = Factor(scope=var_names[0:2], card=cards[0:2], name="p(A|B)")
  f_val = np.random.random(cards[0:2])
  f_val /= f_val.sum(axis=0)
  f.val = f_val

  # P(B|C)
  g = Factor(scope=var_names[1:], card=cards[1:], name="p(B|C)")
  g_val = np.random.random(cards[1:])
  g_val /= g_val.sum(axis=0)
  g.val = g_val

  # P(C)
  h = Factor(scope=var_names[-1:], card=cards[-1:], name="p(C)")
  h_val = np.random.random(cards[-1:])
  h_val /= h_val.sum(axis=0)
  h.val = h_val

  print "We have constructed a Markov chain of the following form:"
  print "%s -> %s -> %s" % (h, g, f)
  print "let's look into these factors in more detail: "

  #===============================================================================
  # Tests:

  print "-" * 80
  print repr(f)
  print repr(g)
  print repr(h)

  print "-" * 80
  fg = f.multiply(g)
  gf = g.multiply(f)
  # check that factor multiplication is commutative
  assert set(list(fg.val.flat)) == set(list(gf.val.flat))
  print "set of probabilites is the same for FG and GF"
  joint = fg.multiply(h)
  print repr(joint)

  print "-" * 80
  observed_f = f.observe_variable("B", 1)
  print repr(observed_f)

  print "-" * 80
  marginalized_joint = joint.marginalize_all_but(set(["B"]))
  print repr(marginalized_joint)

  print "-" * 80
  fully_marginalized = joint.marginalize_all_but([])
  print repr(fully_marginalized)


# Running this testing script should give you the following values:
"""We have constructed a Markov chain of the following form: p(C) -> p(B|C) -> p(A|B) let's look into these factors in more detail: -------------------------------------------------------------------------------- Factor p(A|B): Factor object with scope ['A', 'B'] and corresponding cardinalities [2, 2] CPD: [[ 0.49854243  0.43360644]

 [ 0.50145757  0.56639356]]

Factor p(B|C):
Factor object with scope ['B', 'C'] and corresponding cardinalities [2, 3]
CPD:
[[ 0.99114969  0.46067461  0.69187016]
 [ 0.00885031  0.53932539  0.30812984]]

Factor p(C):
Factor object with scope ['C'] and corresponding cardinalities [3]
CPD:
[ 0.42356358  0.10743397  0.46900246]

--------------------------------------------------------------------------------
Factor ((p(A|B) * p(B|C)) * p(C)):
Factor object with scope ['A', 'C', 'B'] and corresponding cardinalities [2 3 2]
CPD:
[[[ 0.20929555  0.00162545]
  [ 0.02467391  0.02512397]
  [ 0.16177144  0.06266205]]

 [[ 0.21051936  0.00212322]
  [ 0.02481819  0.0328179 ]
  [ 0.16271737  0.0818516 ]]]

--------------------------------------------------------------------------------
Factor (p(A|B) with variable B observed as 1):
Factor object with scope ['A', 'B'] and corresponding cardinalities [2, 2]
CPD:
[[ 0.          0.43360644]
 [ 0.          0.56639356]]

--------------------------------------------------------------------------------
Factor (\sum_{A, C} ((p(A|B) * p(B|C)) * p(C))):
Factor object with scope ['B'] and corresponding cardinalities [2]
CPD:
[ 0.79379582  0.20620418]

[Finished in 0.2s]
"""
