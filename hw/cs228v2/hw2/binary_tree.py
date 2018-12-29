###############################################################################
# simple binary tree class.  Will be used to construct the clique tree for this
# problem.  It has functions specific to this problem.
#
# author: Billy Jun, Isaac Caswell
# date: Jan 20, 2015, Jan 15, 2016
###############################################################################


class BinaryTreeNode:

  def __init__(self):
    self.idx = None  # self.idx is an index of this node within some greater tree
    self.data = None  # self.data will be an animal or ancestor name in our case
    self.left = None  # self.left is a BinaryTreeNode object
    self.right = None  # self.right is also a BinaryTreeNode object.  Go figure.
    self.factor = None  # factor including this node and its parent (unless it's root)
    self.combination = None # \phi_{X_i} in VE algorithm, i.e. combination of all factors including this node or its children
    self.message_to_parent = None  # message from the node to its parent
    self.message_from_parent = None

  def is_leaf(self):
    return self.left == None and self.right == None

  def parse_from_string(self, str_to_parse):
    """
        parses a BinaryTreeNode from a string of nested parentheses.
        The string str_to_parse might look like the following:

        ((((((human,baboon),marmoset),((rat,mouse),rabbit)),(snail,goat)),elephant),platypus)
        """

    if '(' not in str_to_parse and ',' not in str_to_parse:
      # it's a leaf node!
      self.data = str_to_parse
      return

    str_to_parse = str_to_parse[1:-1]  #remove parentheses
    num_open = 0
    for i in xrange(len(str_to_parse)):
      if num_open == 0 and str_to_parse[i] == ',':
        self.left = BinaryTreeNode()
        self.left.parse_from_string(str_to_parse[:i])
        self.right = BinaryTreeNode()
        self.right.parse_from_string(str_to_parse[(i + 1):])
        break
      elif str_to_parse[i] == '(':
        num_open += 1
      elif str_to_parse[i] == ')':
        num_open -= 1

  def __repr__(self):
    """
        Returns a representation of the binary tree.    This may be useful for
        debugging.
        usage:

        tree = BinaryTreeNode
        print repr(tree)

        A note for the curious: obj.__repr__(), corresponding to the builtin
        function repr(obj),
        and obj.__str__(), corresponding to the builtin function str(obj), are
        identical except
        that the former aims for being informative (and therefore often longer)
        whereas the latter
        aims for being readable.
        """
    if self.is_leaf():
      return '{0}-{1}'.format(self.idx, self.data)
    return '(' + repr(self.left) + ',' + repr(self.right) + ')'

  def assign_idx(self, interiorNodes, rf):
    """
        :param list rf: a singleton list to keep track of the indices. It should
        by all rights be
            an integer, but it needs to be referenced by all the various stack
            frames, so this was
            an easy workaround.

        This function labels each interior node with an index corresponding to
        its place in the tree.
        For instance, in a tree with three leaf nodes and two interior nodes,
        the interior nodes
        will be labeled "animal_1" and "animal_0"

        """
    if self.left != None:
      self.left.assign_idx(interiorNodes, rf)
    if self.right != None:
      self.right.assign_idx(interiorNodes, rf)
    self.idx = rf[0]
    if self.data == None:
      interiorNodes.append(self.idx)
      self.data = 'animal_' + str(self.idx)
    rf[0] += 1
