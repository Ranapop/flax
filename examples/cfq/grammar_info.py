from typing import List

import numpy as np
import jax
from jax import numpy as jnp

from examples.cfq.syntax_based.grammar import Grammar
import examples.cfq.syntax_based.asg as asg

class GrammarInfo():
  """
  Class with grammar info stored as jax numpy arrays. This data type is used to
  send grammar information to the flax modules.
  Attributes:
    node_vocab: Dict of node (str) -> index.
    node_vocab_size: No of nodes.
    rule_vocab_size: No of rules.
    grammar_entry: grammar entry node (index).
    nodes_to_action_types: A mapping from node types to action types stored as a
      binary vector. If the node is a head in a rule, the action will be an
      ApplyRule, and GenToken otherwise.
    expanded_nodes: A tuple of arrays showing how predicted branches should be
      expanded into nodes. The first array contains rule -> nodes and is
      padded with zeroes to have a fixed size while the 2nd array contains the
      number of nodes for each rule.
  """

  def __init__(self, grammar: Grammar):
    node_types, node_flags = grammar.collect_node_types()
    self.node_vocab = self.construct_vocab(node_types)
    self.node_vocab_size = len(node_types)
    self.rule_vocab_size = len(grammar.branches)
    self.grammar_entry = self.node_vocab[grammar.grammar_entry]
    self.nodes_to_action_types = self.construct_nodes_to_action_types(
      node_types, node_flags)
    expanded_nodes_list = grammar.get_expanded_nodes(self.node_vocab)
    self.expanded_nodes = self.get_expanded_nodes_array(expanded_nodes_list)
    self.max_node_expansion = self.expanded_nodes[0].shape[1]
    self.valid_rules_by_nodes = self.get_valid_rules_by_nodes(grammar)

  def construct_vocab(self, items_list: List[str]):
    """Constructs vocabulary from list (word -> index). Assumes list contains no
    duplicates."""
    vocab = {}
    for i in range(len(items_list)):
      vocab[items_list[i]] = i
    return vocab

  def construct_nodes_to_action_types(self,
                                      node_types: List[str],
                                      node_flags: List[int]) -> jnp.array:
    """Construct a dictionary node type idx -> action type.

    Args:
      node_types: list of node types.
      node_flags: flag array speciffying if a node is a rule node (1) or not (0).
    Returns:
     Array node idx -> action type.
    """
    nodes_to_action_types = np.zeros((len(node_types)))
    for i in range(len(node_types)):
      node_idx = self.node_vocab[node_types[i]]
      if node_flags[i] == 1:
        action_type = asg.APPLY_RULE
      else:
        action_type = asg.GENERATE_TOKEN
      nodes_to_action_types[i] = action_type
    return nodes_to_action_types

  def get_expanded_nodes_array(self, expanded_nodes: List[List[int]]):
    """
    Gets as parameter a list of lists, a list of nodes expansion for each
    rule. The last list will be an empty list, found at position rule_vocab_size
    and will be used when generating a token.
    
    This list will be transformed into a 2D jnp array, with padding to not have
    a ragged shape. The length of each list will be also returned in a separate
    array.

    Returns
      (node_expansions, lengths).
    """
    no_lists = len(expanded_nodes)
    lists_lengths = [len(l) for l in expanded_nodes]
    max_list_len = max(lists_lengths)
    node_expansions_array = jnp.zeros((no_lists, max_list_len), dtype=jnp.int32)
    lengths_array = jnp.array(lists_lengths)
    for i in range(no_lists):
      current_list = expanded_nodes[i]
      length = len(current_list)
      current_list.reverse()
      reversed_nodes = jnp.array(current_list)
      indexes = jax.ops.index[i, 0:length]
      node_expansions_array = jax.ops.index_update(
        node_expansions_array, indexes, reversed_nodes)
    return node_expansions_array, lengths_array

  def get_valid_rules_by_nodes(self, grammar: Grammar):
    mask_shape = (self.node_vocab_size, self.rule_vocab_size)
    mask = jnp.zeros(mask_shape, dtype=bool)
    for node, node_idx in self.node_vocab.items():
      if node in grammar.rules.keys():
        # Get the valid branches.
        valid_rules = grammar.rules[node]
        for r in valid_rules:
          branch = grammar.branches[r]
          branch_id = branch.branch_id
          mask = jax.ops.index_update(mask,
                                      jax.ops.index[node_idx, branch_id],
                                      True)
    return mask
    
