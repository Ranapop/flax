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
from grammar import Grammar, GRAMMAR_STR
from asg import generate_action_sequence
from collections import deque
from asg import Action, APPLY_RULE, GENERATE_TOKEN

class Node:

  def __init__(self, parent: 'Node', value: str):
    self.parent = parent
    self.value = value
    self.children = []

  def add_child(self, child: 'Node'):
    self.children.append(child)


def apply_action(frontier_nodes_stack: deque, action: Action, grammar: Grammar):
  """Applies an action (apply rule or generate token). The action extends a
  AST that is under construction by extending the stack of frontier nodes.
  The function returns the extended stack."""
  current_node: Node = frontier_nodes_stack.pop()
  action_type, action_value = action
  if action_type == GENERATE_TOKEN:
    # Generate leaf node with token stored in the action.
    child = Node(current_node, action_value)
    current_node.add_child(child)
  else:
    new_frontier_nodes = []
    rule_name = action_value
    head, body = grammar.sub_rules[rule_name]
    if head != current_node.value:
      raise Exception('Invalid action. Got {} and expected {}'.format(
        head, current_node.value))
    rule_tokens = body.split()
    for rule_token in rule_tokens:
      match = re.match(r'\"(.*)\"', rule_token)
      if match:
        # Create a leaf node with a token from the rule (e.g. SELECT).
        node_value = match.groups()[0]
        child = Node(current_node, node_value)
        current_node.add_child(child)
      else:
        # Create new frontier node.
        child = Node(current_node, rule_token)
        current_node.add_child(child)
        new_frontier_nodes.append(child)
    new_frontier_nodes.reverse()
    frontier_nodes_stack.extend(new_frontier_nodes)
  return frontier_nodes_stack

def apply_first_action(action: Action, grammar: Grammar):
  action_type, action_value = action
  if action_type != APPLY_RULE:
    raise Exception('First action should be rule application')
  _, node_value = grammar.sub_rules[action_value]
  root = Node(None, node_value)
  return root

def apply_sequence_of_actions(action_sequence: List, grammar: Grammar):
  root = apply_first_action(action_sequence[0], grammar)
  frontier_nodes = deque()
  frontier_nodes.append(root)
  for action in action_sequence[1:]:
    frontier_nodes = apply_action(frontier_nodes, action, grammar)
  return root

def traverse_tree(root: Node):
  """DFS raversal of tree. The result of the traversal should be the query. In
  case the parent node is a terminal, the descendents will be simply
  concatenated, e.g. VAR, otherwise the substrings are merged together by spaces.
  """
  if len(root.children) == 0:
    return root.value
  children_substrings = []
  for child in root.children:
    child_substr = traverse_tree(child)
    children_substrings.append(child_substr)
  if root.value.isupper():
    delimiter = ''
  else:
    delimiter = ' '
  return delimiter.join(children_substrings)

if __name__=="__main__":
  query = """SELECT DISTINCT ?x0 WHERE {
      ?x0 a people.person .
      ?x0 influence.influencenode.influencedby ?x1 .
      ?x1 film.actor.filmnsfilm.performance.character M1 }"""
  grammar = Grammar(GRAMMAR_STR)
  generated_action_sequence = generate_action_sequence(query, grammar)
  root = apply_sequence_of_actions(generated_action_sequence, grammar)
  query = traverse_tree(root)
  print(query)


    