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
"""Module for unit tests for asg.py
Should be run from cfq with:
python -m tests.syntax_based.grammar_test
"""
from absl.testing import absltest
from absl.testing import parameterized
from examples.cfq.syntax_based.asg import apply_rule_act, generate_act, generate_action_sequence
from examples.cfq.syntax_based.grammar import Grammar, GRAMMAR_STR


class AsgTest(parameterized.TestCase):

  def test_enerate_action_sequence_simple(self):
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
    query = """SELECT DISTINCT ?x0 WHERE {
      ?x0 a people.person }"""
    grammar = Grammar(grammar_str)
    generated_action_sequence = generate_action_sequence(query, grammar)
    expected_action_sequence = [
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
    self.assertEqual(generated_action_sequence, expected_action_sequence)

  def test_enerate_action_sequence_muliple(self):
    grammar_str = r"""
      query: select_query
      select_query: select_clause "WHERE" "{" where_clause "}"
      select_clause: "SELECT" "DISTINCT" "?x0"
                  | "SELECT" "count(*)"     
      where_clause: where_entry 
                  | where_clause "." where_entry
      where_entry: triples_block
      triples_block: var_token TOKEN var_token
      var_token: VAR
              | TOKEN 
      VAR: "?x0" | "?x1" | "?x2" | "?x3" | "?x4" | "?x5"
      TOKEN: /[^\s]+/
    """
    query = """SELECT DISTINCT ?x0 WHERE {
      ?x0 a people.person .
      ?x0 influence.influencenode.influencedby ?x1 .
      ?x1 film.actor.filmnsfilm.performance.character M1 }"""
    grammar = Grammar(grammar_str)
    generated_action_sequence = generate_action_sequence(query, grammar)
    expected_action_sequence = [
      (0, 0), # query -> select_query
      (0, 1), # select_query -> select_clause "WHERE" "{" where_clause "}"
      (0, 2), # select_clause -> "SELECT" "DISTINCT" "?x0"
      (0, 5), # where_clause -> where_clause "." where_entry
      (0, 5), # where_clause -> where_clause "." where_entry
      (0, 4), # where_clause -> where_entry
      (0, 6), # where_entry -> triples_block
      (0, 7), # triples_block -> var_token TOKEN var_token
      (0, 8), # var_token -> VAR
      (0, 10), # VAR -> "?x0"
      (0, 16), # TOKEN -> /[^\s]+/
      (1, 'a'),
      (0, 9), # var_token -> TOKEN
      (0, 16), # TOKEN -> /[^\s]+/
      (1, 'people.person'),
      (0, 6), # where_entry -> triples_block
      (0, 7), # triples_block -> var_token TOKEN var_token
      (0, 8), # var_token -> VAR
      (0, 10), # VAR -> "?x0"
      (0, 16), # TOKEN -> /[^\s]+/
      (1, 'influence.influencenode.influencedby'),
      (0, 8), # var_token -> VAR
      (0, 11), # VAR -> "?x1"
      (0, 6), # where_entry -> triples_block
      (0, 7), # triples_block -> var_token TOKEN var_token
      (0, 8), # var_token -> VAR
      (0, 11), # VAR -> "?x1"
      (0, 16), # TOKEN -> /[^\s]+/
      (1, 'film.actor.filmnsfilm.performance.character'),
      (0, 9), # var_token -> TOKEN
      (0, 16), # TOKEN -> /[^\s]+/
      (1, 'M1')
    ]
    self.assertEqual(generated_action_sequence, expected_action_sequence)

  def test_generate_action_sequence_filter(self):
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
    query = """SELECT count(*)  WHERE {
      ?x0 a film.editor .
      ?x0 influence.influence_node.influenced_by ?x1 .
      ?x1 simplified_token M1 .
      FILTER ( ?x1 != M1 ) 
    }
    """
    grammar = Grammar(grammar_str)
    generated_action_sequence = generate_action_sequence(query, grammar)
    expected_action_sequence = [
      (0, 0), # query -> select_query
      (0, 1), # select_query -> select_clause "WHERE" "{" where_clause "}"
      (0, 3), # select_clause -> "SELECT" "count(*)"
      (0, 5), # where_clause -> where_clause "." where_entry
      (0, 5), # where_clause -> where_clause "." where_entry
      (0, 5), # where_clause -> where_clause "." where_entry
      (0, 4), # where_clause -> where_entry
      (0, 6), # where_entry -> triples_block
      (0, 9), # triples_block -> var_token TOKEN var_token
      (0, 10), # var_token -> VAR
      (0, 12), # VAR -> "?x0"
      (0, 18), # TOKEN -> /[^\s]+/
      (1, 'a'),
      (0, 11), # var_token -> TOKEN
      (0, 18), # TOKEN -> /[^\s]+/
      (1, 'film.editor'),
      (0, 6), # where_entry -> triples_block
      (0, 9), # triples_block -> var_token TOKEN var_token
      (0, 10), # var_token -> VAR
      (0, 12), # VAR -> "?x0"
      (0, 18), # TOKEN -> /[^\s]+/
      (1, 'influence.influence_node.influenced_by'),
      (0, 10), # var_token -> VAR
      (0, 13), # VAR -> "?x1"
      (0, 6), # where_entry -> triples_block
      (0, 9), # triples_block -> var_token TOKEN var_token
      (0, 10), # var_token -> VAR
      (0, 13), # VAR -> "?x1"
      (0, 18), # TOKEN -> /[^\s]+/
      (1, 'simplified_token'),
      (0, 11), # var_token -> TOKEN
      (0, 18), # TOKEN -> /[^\s]+/
      (1, 'M1'),
      (0, 7), # where_entry -> filter_clause
      (0, 8), # filter_clause -> "FILTER" "(" var_token "!=" var_token ")"
      (0, 10), # var_token -> VAR
      (0, 13), # VAR -> "?x1"
      (0, 11), # var_token -> TOKEN
      (0, 18), # TOKEN -> /[^\s]+/
      (1, 'M1')
    ]
    self.assertEqual(generated_action_sequence, expected_action_sequence)


if __name__ == '__main__':
  absltest.main()