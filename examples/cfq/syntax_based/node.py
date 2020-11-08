# Copyright 2020 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""This module is for constructing the syntax tree from the action sequence."""

import re
from typing import List
from syntax_based.grammar import Grammar, GRAMMAR_STR, TermType
from syntax_based.asg import generate_action_sequence, Action, APPLY_RULE, GENERATE_TOKEN
from collections import deque
class Node:

  def __init__(self, parent: 'Node', value: str, time_step: int):
    self.parent = parent
    # rule head or regex term.
    self.value = value
    self.children = []
    self.time_step = time_step
    # id of rule expanded at this node.
    self.rule_id = None
    # token stored in the node
    self.token = None

  def add_child(self, child: 'Node'):
    self.children.append(child)

  def set_rule_id(self, id: int):
    self.rule_id = id

  def set_token(self, token: str):
    self.token = token

  def __repr__(self):
    return self.value

  def pretty_print_tree(self, depth: int = 0):
    spaces = ''.join(['\t' for d in range(depth)])
    if self.token is None:
      print(spaces, self.value)
    else:
      print(spaces, self.token)
    for child in self.children:
      child.pretty_print_tree(depth+1)

  def get_parent_time_steps(self):
    if self.parent is None:
      parent_step = -1
    else:
      parent_step = self.parent.time_step
    parent_steps = [parent_step]
    for child in self.children:
      parent_steps += child.get_parent_time_steps()
    return parent_steps

  def get_node_types(self):
    node_types = [self.value]
    for child in self.children:
      node_types += child.get_node_types()
    return node_types

  def extract_query(self, grammar: Grammar):
    """DFS traversal of tree. The result of the traversal should be the query. In
    case the parent node is a terminal, the descendents will be simply
    concatenated, e.g. VAR, otherwise the substrings are merged together by spaces.
    When rule nodes are being visited, the grammar rule is traversed and the syntax
    tokens are added, besides visiting the children nodes in the AST.
    """
    if self.token is not None:
      # leaf/token node
      return self.token
    children_substrings = []
    rule_branch = grammar.branches[self.rule_id]
    child_idx = 0
    for term in rule_branch.body:
      if term.term_type == TermType.SYNTAX_TERM:
        children_substrings.append(term.value)
      else:
        child_node = self.children[child_idx]
        child_substr = child_node.extract_query(grammar)
        children_substrings.append(child_substr)
        child_idx += 1
    if self.value.isupper():
      delimiter = ''
    else:
      delimiter = ' '
    return delimiter.join(children_substrings)

def apply_action(frontier_nodes_stack: deque,
                 action: Action,
                 time_step: int,
                 grammar: Grammar):
  """Applies an action (apply rule or generate token). The action extends an
  AST that is under construction by extending the stack of frontier nodes.
  The function returns the extended stack."""
  current_node: Node = frontier_nodes_stack.pop()
  action_type, action_value = action
  if action_type == GENERATE_TOKEN:
    # Fill leaf node with token stored in the action.
    current_node.set_token(action_value)
  else:
    current_node.set_rule_id(action_value)
    new_frontier_nodes = []
    rule_branch = grammar.branches[action_value]
    for term in rule_branch.body:
      term_type = term.term_type
      if term_type == TermType.RULE_TERM or term_type == TermType.REGEX_TERM:
        child = Node(current_node, term.value, time_step)
        new_frontier_nodes.append(child)
        current_node.add_child(child)
    new_frontier_nodes.reverse()
    frontier_nodes_stack.extend(new_frontier_nodes)
  return frontier_nodes_stack


def construct_root(grammar: Grammar):
  return Node(None, grammar.grammar_entry, 0)


def apply_sequence_of_actions(action_sequence: List, grammar: Grammar):
  """Applies a sequence of actions to construct a syntax tree."""
  root = construct_root(grammar)
  frontier_nodes = deque()
  frontier_nodes.append(root)
  time_step = 1
  for action in action_sequence:
    frontier_nodes = apply_action(frontier_nodes, action, time_step, grammar)
    time_step += 1
  return root


if __name__ == "__main__":
  query = """SELECT DISTINCT ?x0 WHERE {
      ?x0 a people.person .
      ?x0 influence.influencenode.influencedby ?x1 .
      ?x1 film.actor.filmnsfilm.performance.character M1 }"""
  grammar = Grammar(GRAMMAR_STR)
  generated_action_sequence = generate_action_sequence(query, grammar)
  print(generated_action_sequence)
  root = apply_sequence_of_actions(generated_action_sequence, grammar)
  parent_time_steps = root.get_parent_time_steps()
  print(parent_time_steps)
  # query = extract_query(root, grammar)
  # print(query)


