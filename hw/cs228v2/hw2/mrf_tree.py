import numpy as np

from factors import Factor
from binary_tree import BinaryTreeNode

class MRFTreeModel():
  def __init__(self, root, prior, cpds, use_logspace=False):
    self.use_logspace = use_logspace
    self.root = root
    self.root.factor = Factor(scope=[root.idx], card=[4], val=prior, name="RootPrior", logfactor=self.use_logspace)
    self.setup_factors(root.left, root.idx, cpds)
    self.setup_factors(root.right, root.idx, cpds)

  def setup_factors(self, node, parent_idx, cpds):
    if node is None:
      return
    self.setup_factors(node.left, node.idx, cpds)
    self.setup_factors(node.right, node.idx, cpds)
    node.factor = Factor(scope=[node.idx, parent_idx], card=[4, 4], val=cpds, name="F({0},{1})".format(node.idx, parent_idx), logfactor=self.use_logspace)

  def pass_messages_up(self, node, parent_idx, evidence, marginalization_type, print_messages):
    if node is None:
      return
    self.pass_messages_up(node.left, node.idx, evidence, marginalization_type, print_messages)
    self.pass_messages_up(node.right, node.idx, evidence, marginalization_type, print_messages)
    node.combination = node.factor
    if node.left:
      node.combination = node.combination.combine(node.left.message_to_parent)
    if node.right:
      node.combination = node.combination.combine(node.right.message_to_parent)
    node.combination = node.combination.observe(evidence)
    node.message_to_parent = node.combination.marginalize_all_but([parent_idx], marginalization_type)
    if print_messages:
      print "Message from {0} to {1}".format(node.idx, parent_idx)
      print repr(node.message_to_parent)

  def pass_messages_down(self, node, marginalization_type, print_messages):
    if node.is_leaf():
      return
    combination = node.message_from_parent
    combination = combination.combine(node.left.message_to_parent)
    combination = combination.combine(node.right.message_to_parent)
    self.marginal_distributions[node.idx] = combination.marginalize_all_but([node.idx], marginalization_type).val
    if not self.use_logspace:
      self.marginal_distributions[node.idx] /= self.p_evidence
    else:
      self.marginal_distributions[node.idx] -= self.p_evidence
    node.left.message_from_parent = \
        node.message_from_parent.\
        combine(node.right.message_to_parent).\
        combine(node.left.factor).marginalize_all_but([node.left.idx], marginalization_type)
    if print_messages:
      print "Message from {0} to {1}".format(node.idx, node.left.idx)
      print repr(node.left.message_from_parent)
    self.pass_messages_down(node.left, marginalization_type, print_messages)
    node.right.message_from_parent = \
        node.message_from_parent.\
        combine(node.left.message_to_parent).\
        combine(node.right.factor).marginalize_all_but([node.right.idx], marginalization_type)
    if print_messages:
      print "Message from {0} to {1}".format(node.idx, node.right.idx)
      print repr(node.right.message_from_parent)
    self.pass_messages_down(node.right, marginalization_type, print_messages)

  def run_message_passing(self, evidence, marginalization_type, print_messages):
    self.evidence = evidence # used in print_data()
    self.pass_messages_up(self.root, -1, evidence, marginalization_type, print_messages)
    self.p_evidence = self.root.message_to_parent.val.flat[0]
    self.marginal_distributions = {}
    self.root.message_from_parent = self.root.factor
    self.pass_messages_down(self.root, marginalization_type, print_messages)
    assert len(self.root.message_to_parent.val.flat) == 1

  def run_traceback_map(self, node, joint):
    # evidence was already observed, the only thing left
    # is to observe already known inner node values
    factor = node.combination.observe(joint)
    if node == self.root:
      assert factor.scope == [self.root.idx]
    else:
      factor = factor.marginalize_all_but([node.idx])
    joint[node.idx] = np.argmax(factor.val)
    if self.evidence.has_key(node.idx):
      assert joint[node.idx] == self.evidence[node.idx]
      del joint[node.idx]
    if not node.is_leaf():
      self.run_traceback_map(node.left, joint)
      self.run_traceback_map(node.right, joint)
    return np.max(factor.val)

  def find_marginal_distributions(self, evidence={}, print_messages=False):
    self.run_message_passing(evidence, "sum", print_messages)

  def find_joint_most_likely(self, evidence={}):
    self.run_message_passing(evidence, "max", False)
    joint = {}
    likelihood = self.run_traceback_map(self.root, joint)
    return likelihood, joint

  def print_p_evidence_and_marginals(self):
    print "Evidence:"
    print self.evidence
    print ("Log " if self.use_logspace else "") + "P(Evidence) =", self.p_evidence
    print "Conditional marginal distributions for every inner node:"
    for node_idx, distribution in self.marginal_distributions.items():
      print node_idx, distribution, "sum =", sum(distribution)


if __name__ == '__main__':
  print "mrf tree"

