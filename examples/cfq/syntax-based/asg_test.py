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
"""Module for unit tests for asg.py."""
from absl.testing import absltest
from absl.testing import parameterized
from asg import apply_rule_act, generate_act, generate_action_sequence
from grammar import Grammar, GRAMMAR_STR


class AsgTest(parameterized.TestCase):

  def test_enerate_action_sequence_muliple(self):
    grammar_str = """
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
      VAR: "?x" DIGIT 
      DIGIT: /\d+/
      TOKEN: /[^\s]+/
    """
    query = """SELECT DISTINCT ?x0 WHERE {
      ?x0 a people.person .
      ?x0 influence.influencenode.influencedby ?x1 .
      ?x1 film.actor.filmnsfilm.performance.character M1 }"""
    grammar = Grammar(grammar_str)
    generated_action_sequence = generate_action_sequence(query, grammar)
    expected_action_sequence = [
      (0, 'r0'),
      (0, 'r1'),
      (0, 'r6'),
      (0, 'r13'),
      (0, 'r13'),
      (0, 'r12'),
      (0, 'r17'),
      (0, 'r18'),
      (0, 'r21'),
      (0, 'r24'),
      (1, '0'),
      (1, 'a'),
      (0, 'r22'),
      (1, 'people.person'),
      (0, 'r17'), (0, 'r18'),
      (0, 'r21'), (0, 'r24'),
      (1, '0'),
      (1, 'influence.influencenode.influencedby'),
      (0, 'r21'),
      (0, 'r24'),
      (1, '1'),
      (0, 'r17'),
      (0, 'r18'),
      (0, 'r21'),
      (0, 'r24'),
      (1, '1'),
      (1, 'film.actor.filmnsfilm.performance.character'),
      (0, 'r22'),
      (1, 'M1')]
    self.assertEqual(generated_action_sequence, expected_action_sequence)

  def test_generate_action_sequence_filter(self):
    grammar_str = """
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
      VAR: "?x" DIGIT 
      DIGIT: /\d+/
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
      (0, 'r0'),
      (0, 'r1'),
      (0, 'r7'),
      (0, 'r13'),
      (0, 'r13'),
      (0, 'r13'),
      (0, 'r12'),
      (0, 'r17'),
      (0, 'r26'),
      (0, 'r29'),
      (0, 'r32'),
      (1, '0'),
      (1, 'a'),
      (0, 'r30'),
      (1, 'film.editor'),
      (0, 'r17'),
      (0, 'r26'),
      (0, 'r29'),
      (0, 'r32'),
      (1, '0'),
      (1, 'influence.influence_node.influenced_by'),
      (0, 'r29'),
      (0, 'r32'),
      (1, '1'),
      (0, 'r17'),
      (0, 'r26'),
      (0, 'r29'),
      (0, 'r32'),
      (1, '1'),
      (1, 'simplified_token'),
      (0, 'r30'),
      (1, 'M1'),
      (0, 'r18'),
      (0, 'r20'),
      (0, 'r29'),
      (0, 'r32'),
      (1, '1'),
      (0, 'r30'),
      (1, 'M1')]
    self.assertEqual(generated_action_sequence, expected_action_sequence)


if __name__ == '__main__':
  absltest.main()