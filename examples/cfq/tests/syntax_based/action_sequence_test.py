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
"""Module with integration tests for generating/applying action sequences.
Should be run from cfq with:
python -m tests.syntax_based.action_sequence_test"""

from absl.testing import absltest
from absl.testing import parameterized
# pytype: disable=import-error
from cfq.syntax_based.grammar import Grammar, GRAMMAR_STR
from cfq.syntax_based.asg import generate_action_sequence
from cfq.syntax_based.node import apply_sequence_of_actions, extract_query, pretty_print_tree
# pytype: enable=import-error
class ActionSequenceTest(parameterized.TestCase):

  def test_action_sequence(self):
    """Test that when going query -> sequence of actions -> tree -> query2
    the input and output query are equal (query, query2)."""
    query = """SELECT count(*) WHERE {
                 ?x0 ns:film.cinematographer.film ?x1 .
                 ?x0 ns:film.writer.film ?x1 .
                 ?x1 a ns:film.film .
                 ?x2 ns:film.film_costumer_designer.costume_design_for_film M1 .
                 M2 ns:film.film.starring/ns:film.performance.actor ?x0 .
                 M2 ns:film.film.starring/ns:film.performance.actor ?x2
               }"""
    no_extra_spaces_query = " ".join(query.split())
    grammar = Grammar(GRAMMAR_STR)
    act_seq = generate_action_sequence(query, grammar)
    root, _ = apply_sequence_of_actions(act_seq, grammar)
    generated_query = extract_query(root, grammar)
    no_extra_spaces_generated_query =  " ".join(generated_query.split())
    self.assertEqual(no_extra_spaces_query, no_extra_spaces_generated_query)
     
  def test_second_action_sequence(self):
    """Test that when going query -> sequence of actions -> tree -> query2
    the input and output query are equal (query, query2)."""
    query = """SELECT count(*) WHERE {
                ?x0 ns:film.director.film M0 .
                ?x0 ns:influence.influence_node.influenced ?x1 .
                ?x0 ns:influence.influence_node.influenced ?x2 .
                ?x0 ns:influence.influence_node.influenced ?x3 .
                ?x1 a ns:film.writer .
                ?x2 ns:organization.organization_founder.organizations_founded ?x4 .
                ?x3 ns:film.writer.film M3 .
                ?x4 a ns:film.production_company
            }"""
    no_extra_spaces_query = " ".join(query.split())
    grammar = Grammar(GRAMMAR_STR)
    act_seq = generate_action_sequence(query, grammar)
    root, _ = apply_sequence_of_actions(act_seq, grammar)
    generated_query = extract_query(root, grammar)
    no_extra_spaces_generated_query =  " ".join(generated_query.split())
    self.assertEqual(no_extra_spaces_query, no_extra_spaces_generated_query)

if __name__ == '__main__':
  absltest.main()