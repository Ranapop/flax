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
"""Module for unit tests for input_pipeline.py."""
from absl.testing import absltest
from absl.testing import parameterized
import tensorflow.compat.v2 as tf
from syntax_based.grammar import Grammar, GRAMMAR_STR
from input_pipeline import Seq2TreeCfqDataSource


class InputPipelineTest(parameterized.TestCase):

  def test_construct_output_fields(self):
    data_source = Seq2TreeCfqDataSource(0, False, load_data=False)
    query = """SELECT count(*) WHERE {
      M0 film.producer.films_executive_produced M1 .
      M0 film.producer.film|ns:film.production_company.films M1 }"""
    data_source.tokens_vocab = {
      b'film.producer.films_executive_produced': 0, b'M0': 1, b'M1': 2,
      b'film.producer.film|ns:film.production_company.films': 3}
    query = tf.constant(query.encode())
    action_types, action_values, node_types, parent_steps =\
      data_source.construct_output_fields(query)
    expected_action_types = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0,
                             0, 0, 1, 0, 1, 0, 0, 1]
    expected_action_values = [0, 1, 3, 5, 4, 6, 9, 11, 18, 1, 18, 0, 11, 18, 2,
                              6, 9, 11, 18, 1, 18, 3, 11, 18, 2]
    expected_node_types = [0, 1, 2, 3, 3, 4, 5, 7, 8, 10, 8, 10, 7, 8, 10, 4, 5,
                           7, 8, 10, 8, 10, 7, 8, 10]
    expected_parent_steps = [-1, 0, 1, 1, 3, 4, 5, 6, 7, 8, 6, 10, 6, 12, 13, 3,
                             15, 16, 17, 18, 16, 20, 16, 22, 23]
    self.assertEqual(action_types, expected_action_types)
    self.assertEqual(action_values, expected_action_values)
    self.assertEqual(node_types,expected_node_types)
    self.assertEqual(parent_steps, expected_parent_steps)


if __name__ == '__main__':
  absltest.main()