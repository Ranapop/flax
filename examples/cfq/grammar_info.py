

class GrammarInfo():
  """
  Class with grammar info stored as jax numpy arrays. This data type is used to
  send grammar information to the flax modules.
  Attributes:
    nodes_to_action_types: A mapping from node types to action types stored as a
      binary vector. If the node is a head in a rule, the action will be an
      ApplyRule, and GenToken otherwise.
    expanded_nodes: A tuple of arrays showing how predicted branches should be
      expanded into nodes. The first array contains rule -> nodes and is
      padded with zeroes to have a fixed size while the 2nd array contains the
      number of nodes for each rule.
  """

  def __init__(self, nodes_to_action_types, expanded_nodes):
    self.nodes_to_action_types = nodes_to_action_types
    self.expanded_nodes = expanded_nodes