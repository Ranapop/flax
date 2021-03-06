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
"""Module with unit tests for grammar.py
Should be run from cfq with:
python -m tests.syntax_based.grammar_test
"""
from absl.testing import absltest
from absl.testing import parameterized
from cfq.syntax_based.grammar import Grammar, RuleBranch, Term, TermType


class GrammarTest(parameterized.TestCase):

  def test_generate_grammar(self):
    grammar_str = r"""
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
    """
    grammar = Grammar(grammar_str)
    expected_branches = [
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
    expected_rules = {
      'query': [0], 'select_query': [1], 'select_clause': [2, 3],
      'where_clause': [4, 5], 'where_entry': [6, 7],
      'triples_block': [9], 'filter_clause': [8],
      'var_token': [10, 11],
      'VAR': [12, 13, 14, 15, 16, 17],
      'TOKEN': [18]
    }
    self.assertEqual(grammar.branches, expected_branches)
    self.assertEqual(grammar.rules, expected_rules)

  def test_Grammar(self):
    grammar_str = """
      a: b
      b: c | d
      c: "some_token"
      d: "some_other_token"
    """
    grammar = Grammar(grammar_str)
    expected_branches = [
      RuleBranch(0, 'b'),
      RuleBranch(1, 'c'),
      RuleBranch(2, 'd'),
      RuleBranch(3, '"some_token"'),
      RuleBranch(4, '"some_other_token"'),
    ]
    expected_rules = {
      'a': [0],
      'b': [1, 2],
      'c': [3],
      'd': [4]
    }
    self.assertEqual(grammar.branches, expected_branches)
    self.assertEqual(grammar.rules, expected_rules)

  def test_regex_rule_branch(self):
    rule_branch = RuleBranch(0, r'/[^\s]+/')
    term = rule_branch.body[0]
    self.assertEqual(term.term_type, TermType.REGEX_TERM)

  def test_collect_node_types(self):
    grammar_str = r"""
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
    """
    grammar = Grammar(grammar_str)
    node_types = grammar.collect_node_types()
    expected_nodes = ['query',
                      'select_query', 'select_clause',
                      'where_clause', 'where_entry', 'triples_block',
                      'filter_clause',
                      'var_token', 'TOKEN', 'VAR',
                      '[^\\s]+']
    expected_action_types = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    self.assertEqual(node_types, (expected_nodes, expected_action_types))

  def test_get_expanded_nodes(self):
    grammar_str = r"""
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
      VAR: "?x0" 
         | "?x1" 
         | "?x2" 
         | "?x3" 
         | "?x4" 
         | "?x5" 
      TOKEN: /[^\s]+/
    """
    grammar = Grammar(grammar_str)
    nodes_vocab = {'query': 0, 'select_query': 1, 'select_clause':2,
                   'where_clause':3, 'where_entry': 4, 'triples_block': 5,
                   'filter_clause': 6,
                   'var_token': 7, 'TOKEN':8, 'VAR':9,
                   '[^\\s]+': 10}
    expected_expanded_nodes = [
      [1],
      [2, 3],
      [],
      [],
      [4],
      [3, 4],
      [5],
      [6],
      [7, 7],
      [7, 8, 7],
      [9],
      [8],
      [],
      [],
      [],
      [],
      [],
      [],
      [10],
      []
    ]
    expanded_nodes = grammar.get_expanded_nodes(nodes_vocab)
    self.assertEqual(expanded_nodes, expected_expanded_nodes)

if __name__ == '__main__':
  absltest.main()
