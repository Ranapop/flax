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
"""Module with unit tests for node.py
Should be run from cfq with:
python -m tests.syntax_based.node_test
"""
from collections import deque
from absl.testing import absltest
from absl.testing import parameterized
# pytype: disable=import-error
from cfq.syntax_based.node import Node, get_parent_time_steps, apply_action
from cfq.syntax_based.grammar import RuleBranch
# pytype: enable=import-error

def add_child_to_parent(parent: Node, child: Node):
    child.parent = parent
    parent.add_child(child)

class FakeGrammar():
  r"""
  Class with fields for the follwing grammar:
  query: select_query
  select_query: select_clause "WHERE" "{" where_clause "}"
  select_clause: "SELECT" "DISTINCT" "?x0"
              | "SELECT" "count(*)"     
  where_clause: where_entry 
              | where_clause "." where_entry
  where_entry: triples_block
            | filter_clause
  filter_clause: "FILTER" "(" var_token "!=" var_token ")"
  triples_block: var_token TOKEN var_token
  var_token: VAR
          | TOKEN 
  VAR: "?x0" | "?x1" | "?x2" | "?x3" | "?x4" | "?x5"
  TOKEN: /[^\s]+/

  Note: r prefaces the comment for running with pytest.
  """
    
  def __init__(self):
    self.branches = [
      RuleBranch(0, 'select_query'),
      RuleBranch(1, 'select_clause "WHERE" "{" where_clause "}"'),
      RuleBranch(2, '"SELECT" "DISTINCT" "?x0"'),
      RuleBranch(3, '"SELECT" "count(*)"'),
      RuleBranch(4, 'where_entry'),
      RuleBranch(5, 'where_clause "." where_entry'),
      RuleBranch(6, 'triples_block'),
      RuleBranch(7, 'filter_clause'),
      RuleBranch(8, '"FILTER" "(" var_token "!=" var_token ")"'),
      RuleBranch(9, 'var_token TOKEN var_token'),
      RuleBranch(10, 'VAR'),
      RuleBranch(11, 'TOKEN'),
      RuleBranch(12, '"?x0"'),
      RuleBranch(13, '"?x1"'),
      RuleBranch(14, '"?x2"'),
      RuleBranch(15, '"?x3"'),
      RuleBranch(16, '"?x4"'),
      RuleBranch(17, '"?x5"'),
      RuleBranch(18, r'/[^\s]+/')
    ]
    self.rules = {
      'query': [0], 'select_query': [1], 'select_clause': [2, 3],
      'where_clause': [4, 5], 'where_entry': [6, 7],
      'triples_block': [9], 'filter_clause': [8],
      'var_token': [10, 11],
      'VAR': [12, 13, 14, 15, 16, 17],
      'TOKEN': [18]
    }
    self.grammar_entry = 'query'

class GrammarTest(parameterized.TestCase):

  def test_get_parent_time_steps(self):
    a = Node(None, 'a')
    b = Node(None, 'b')
    c = Node(None, 'c')
    d = Node(None, 'd')
    e = Node(None, 'e')
    f = Node(None, 'f')
    add_child_to_parent(a, b)
    add_child_to_parent(b, c)
    add_child_to_parent(c, d)
    add_child_to_parent(c, f)
    add_child_to_parent(d, e)
    generated_time_steps = get_parent_time_steps(a)
    expected_time_steps = [-1, 0, 1, 2, 3, 2]
    self.assertEqual(generated_time_steps, expected_time_steps)
  
  #Disable this because FakeGrammar is sent in the tests instead of Grammar.
  # pytype: disable=wrong-arg-types
  def test_apply_action_for_sequence(self):
    """
    Test the action application for the following query:
    SELECT DISTINCT ?x0 WHERE {
      ?x0 a people.person }
    """
    actions = [
      (0, 0), # query -> select_query
      (0, 1), # select_query -> select_clause "WHERE" "{" where_clause "}"
      (0, 2), # select_clause -> "SELECT" "DISTINCT" "?x0"
      (0, 4), # where_clause -> where_entry
      (0, 6), # where_entry -> triples_block
      (0, 9), # triples_block -> var_token TOKEN var_token
      (0, 10), # var_token -> VAR
      (0, 12), # VAR -> "?x0"
      (0, 18), # TOKEN -> /[^\s]+/
      (1, 'a'),
      (0, 11), # var_token -> TOKEN
      (0, 18), # TOKEN -> /[^\s]+/
      (1, 'people.person')
    ]
    grammar = FakeGrammar()
    # Initial frontier: query
    root = Node(None, grammar.grammar_entry)
    frontier_nodes = deque()
    frontier_nodes.append(root)

    # Step 0: ApplyRule query -> select_query
    current_node_value = frontier_nodes[-1].value
    expected_node_value = 'query'
    frontier_nodes, _ = apply_action(frontier_nodes, actions[0], grammar)
    frontier_nodes_values = [n.value for n in frontier_nodes]
    expected_frontier_nodes = ['select_query']
    self.assertEqual(current_node_value, expected_node_value)
    self.assertEqual(frontier_nodes_values, expected_frontier_nodes)
    # Step 1: ApplyRule select_query -> select_clause "WHERE" "{" where_clause "}"
    current_node_value = frontier_nodes[-1].value
    expected_node_value = 'select_query'
    frontier_nodes, _ = apply_action(frontier_nodes, actions[1], grammar)
    frontier_nodes_values = [n.value for n in frontier_nodes]
    expected_frontier_nodes = ['where_clause', 'select_clause']
    self.assertEqual(current_node_value, expected_node_value)
    self.assertEqual(frontier_nodes_values, expected_frontier_nodes)
    # Step 2: ApplyRule select_clause -> "SELECT" "DISTINCT" "?x0"
    current_node_value = frontier_nodes[-1].value
    expected_node_value = 'select_clause'
    frontier_nodes, _ = apply_action(frontier_nodes, actions[2], grammar)
    frontier_nodes_values = [n.value for n in frontier_nodes]
    expected_frontier_nodes = ['where_clause']
    self.assertEqual(current_node_value, expected_node_value)
    self.assertEqual(frontier_nodes_values, expected_frontier_nodes)
    # Step 3: ApplyRule where_clause -> where_entry
    current_node_value = frontier_nodes[-1].value
    expected_node_value = 'where_clause'
    frontier_nodes, _ = apply_action(frontier_nodes, actions[3], grammar)
    frontier_nodes_values = [n.value for n in frontier_nodes]
    expected_frontier_nodes = ['where_entry']
    self.assertEqual(current_node_value, expected_node_value)
    self.assertEqual(frontier_nodes_values, expected_frontier_nodes)
    # Step 4: ApplyRule where_entry -> triples_block
    current_node_value = frontier_nodes[-1].value
    expected_node_value = 'where_entry'
    frontier_nodes, _ = apply_action(frontier_nodes, actions[4], grammar)
    frontier_nodes_values = [n.value for n in frontier_nodes]
    expected_frontier_nodes = ['triples_block']
    self.assertEqual(current_node_value, expected_node_value)
    self.assertEqual(frontier_nodes_values, expected_frontier_nodes)
    # Step 5: ApplyRule triples_block -> var_token TOKEN var_token
    current_node_value = frontier_nodes[-1].value
    expected_node_value = 'triples_block'
    frontier_nodes, _ = apply_action(frontier_nodes, actions[5], grammar)
    frontier_nodes_values = [n.value for n in frontier_nodes]
    expected_frontier_nodes = ['var_token', 'TOKEN', 'var_token']
    self.assertEqual(current_node_value, expected_node_value)
    self.assertEqual(frontier_nodes_values, expected_frontier_nodes)
    # Step 6: ApplyRule var_token -> VAR
    current_node_value = frontier_nodes[-1].value
    expected_node_value = 'var_token'
    frontier_nodes, _ = apply_action(frontier_nodes, actions[6], grammar)
    frontier_nodes_values = [n.value for n in frontier_nodes]
    expected_frontier_nodes = ['var_token', 'TOKEN', 'VAR']
    self.assertEqual(current_node_value, expected_node_value)
    self.assertEqual(frontier_nodes_values, expected_frontier_nodes)
    # Step 7: ApplyRule VAR -> "?x0"
    current_node_value = frontier_nodes[-1].value
    expected_node_value = 'VAR'
    frontier_nodes, _ = apply_action(frontier_nodes, actions[7], grammar)
    frontier_nodes_values = [n.value for n in frontier_nodes]
    expected_frontier_nodes = ['var_token', 'TOKEN']
    self.assertEqual(current_node_value, expected_node_value)
    self.assertEqual(frontier_nodes_values, expected_frontier_nodes)
    # Step 8: ApplyRule TOKEN -> /[^\s]+/
    current_node_value = frontier_nodes[-1].value
    expected_node_value = 'TOKEN'
    frontier_nodes, _ = apply_action(frontier_nodes, actions[8], grammar)
    frontier_nodes_values = [n.value for n in frontier_nodes]
    expected_frontier_nodes = ['var_token', r'[^\s]+']
    self.assertEqual(current_node_value, expected_node_value)
    self.assertEqual(frontier_nodes_values, expected_frontier_nodes)
    # Step 9: GenerateToken 'a'
    current_node_value = frontier_nodes[-1].value
    expected_node_value = r'[^\s]+'
    frontier_nodes, _ = apply_action(frontier_nodes, actions[9], grammar)
    frontier_nodes_values = [n.value for n in frontier_nodes]
    expected_frontier_nodes = ['var_token']
    self.assertEqual(current_node_value, expected_node_value)
    self.assertEqual(frontier_nodes_values, expected_frontier_nodes)
    # Step 10: ApplyRule var_token -> TOKEN
    current_node_value = frontier_nodes[-1].value
    expected_node_value = 'var_token'
    frontier_nodes, _ = apply_action(frontier_nodes, actions[10], grammar)
    frontier_nodes_values = [n.value for n in frontier_nodes]
    expected_frontier_nodes = ['TOKEN']
    self.assertEqual(current_node_value, expected_node_value)
    self.assertEqual(frontier_nodes_values, expected_frontier_nodes)
    # Step 11: ApplyRule TOKEN -> /[^\s]+/
    current_node_value = frontier_nodes[-1].value
    expected_node_value = 'TOKEN'
    frontier_nodes, _ = apply_action(frontier_nodes, actions[11], grammar)
    frontier_nodes_values = [n.value for n in frontier_nodes]
    expected_frontier_nodes = [r'[^\s]+']
    self.assertEqual(current_node_value, expected_node_value)
    self.assertEqual(frontier_nodes_values, expected_frontier_nodes)
    # Step 12: GenerateToken 'people.person'
    current_node_value = frontier_nodes[-1].value
    expected_node_value = r'[^\s]+'
    frontier_nodes, _ = apply_action(frontier_nodes, actions[12], grammar)
    frontier_nodes_values = [n.value for n in frontier_nodes]
    expected_frontier_nodes = []
    self.assertEqual(current_node_value, expected_node_value)
    self.assertEqual(frontier_nodes_values, expected_frontier_nodes)
  # pytype: enable=wrong-arg-types
    

if __name__ == '__main__':
  absltest.main()