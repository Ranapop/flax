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

  def __init__(self, parent: 'Node', value: str):
    self.parent = parent
    # rule head or regex term.
    self.value = value
    self.children = []
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

def apply_action(frontier_nodes_stack: deque, action: Action, grammar: Grammar):
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
        child = Node(current_node, term.value)
        new_frontier_nodes.append(child)
        current_node.add_child(child)
    new_frontier_nodes.reverse()
    frontier_nodes_stack.extend(new_frontier_nodes)
  return frontier_nodes_stack


def construct_root(grammar: Grammar):
  return Node(None, grammar.grammar_entry)


def apply_sequence_of_actions(action_sequence: List, grammar: Grammar):
  """Applies a sequence of actions to construct a syntax tree."""
  root = construct_root(grammar)
  frontier_nodes = deque()
  frontier_nodes.append(root)
  for action in action_sequence:
    if not frontier_nodes:
      # Stop applying sequences when the stack is empty.
      return root
    frontier_nodes = apply_action(frontier_nodes, action, grammar)
  return root


def extract_query(root: Node, grammar: Grammar):
  """DFS traversal of tree. The result of the traversal should be the query. In
  case the parent node is a terminal, the descendents will be simply
  concatenated, e.g. VAR, otherwise the substrings are merged together by spaces.
  When rule nodes are being visited, the grammar rule is traversed and the syntax
  tokens are added, besides visiting the children nodes in the AST.
  """
  if root.token is not None:
    # leaf/token node
    return root.token
  children_substrings = []
  if root.rule_id is None:
    # This means the sequence of actions was incomplete.
    return ''
  rule_branch = grammar.branches[root.rule_id]
  child_idx = 0
  for term in rule_branch.body:
    if term.term_type == TermType.SYNTAX_TERM:
      children_substrings.append(term.value)
    else:
      child_node = root.children[child_idx]
      child_substr = extract_query(child_node, grammar)
      children_substrings.append(child_substr)
      child_idx += 1
  if root.value.isupper():
    delimiter = ''
  else:
    delimiter = ' '
  return delimiter.join(children_substrings)


def pretty_print_tree(root: Node, depth: int = 0):
  spaces = ''.join(['\t' for d in range(depth)])
  if root.token is None:
    print(spaces, root.value)
  else:
    print(spaces, root.token)
  for child in root.children:
    pretty_print_tree(child, depth+1)


def get_parent_time_steps(root: Node,
                          current_step: int = 0, parent_step: int = -1):
  parent_time_steps = [parent_step]
  parent_step = current_step
  current_step += 1
  for child in root.children:
    child_parent_times = get_parent_time_steps(child, current_step, parent_step)
    current_step += len(child_parent_times)
    parent_time_steps += child_parent_times
  return parent_time_steps


def get_node_types(root: Node):
    node_types = [root.value]
    for child in root.children:
      node_types += get_node_types(child)
    return node_types


if __name__ == "__main__":
  query = """SELECT DISTINCT ?x0 WHERE {
      ?x0 a people.person .
      ?x0 influence.influencenode.influencedby ?x1 .
      ?x1 film.actor.filmnsfilm.performance.character M1 }"""
  grammar = Grammar(GRAMMAR_STR)
  generated_action_sequence = generate_action_sequence(query, grammar)
  print(generated_action_sequence)
  root = apply_sequence_of_actions(generated_action_sequence, grammar)
  parent_time_steps = get_parent_time_steps(root)
  print(parent_time_steps)
  # query = extract_query(root, grammar)
  # print(query)


