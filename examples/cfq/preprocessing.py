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
"""Preprocessing module

The preprocessing here is taken from
https://github.com/google-research/google-research/tree/master/cfq
to ensure comparable results with the CFQ paper
"""

from typing import Dict, Text, Callable
import string
import tensorflow.compat.v2 as tf
import cfq.input_pipeline_constants as inp_constants

ExampleType = Dict[Text, tf.Tensor]


def preprocess_example(example: ExampleType) -> ExampleType:
  """Preprocess question and query"""
  example[inp_constants.QUESTION_KEY] = tf_wrap_seq_fun(
      preprocess_question, example[inp_constants.QUESTION_KEY])
  example[inp_constants.QUERY_KEY] = tf_wrap_seq_fun(preprocess_sparql,
                                                 example[inp_constants.QUERY_KEY])
  return example


def preprocess_question(tensor: tf.Tensor):
  """Preprocess question by wrapping punctuation with spaces
  (helps tokenization)"""
  text = tensor.numpy().decode()
  mapped = map(lambda c: f' {c} ' if c in string.punctuation else c, text)
  preprocessed = ' '.join(''.join(mapped).split())
  return preprocessed


def preprocess_sparql(tensor: tf.Tensor):
  """Do various preprocessing on the SPARQL query."""
  query = tensor.numpy().decode()

  tokens = []
  for token in query.split():
    # Replace 'ns:' prefixes.
    if token.startswith('ns:'):
      token = token[3:]
    # Replace mid prefixes.
    if token.startswith('m.'):
      token = 'm_' + token[2:]
    tokens.append(token)

  preprocessed_query = ' '.join(tokens).replace('\\n', ' ')
  return preprocessed_query


def tf_wrap_seq_fun(fun: Callable[[tf.Tensor], Text], text: tf.Tensor):
  "Apply fun on a string (sequence) with py_function"
  fun_output = tf.py_function(fun, [text], Tout=tf.string)
  fun_output.set_shape(text.get_shape())
  return fun_output
