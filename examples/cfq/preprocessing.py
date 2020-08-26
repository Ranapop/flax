import tensorflow.compat.v2 as tf
from typing import Dict, Text, Any
import string
import constants

ExampleType = Dict[Text, tf.Tensor]


def preprocess_example(example: ExampleType) -> ExampleType:
    example[constants.QUESTION_KEY] = tf_wrap_seq_fun(
      preprocess_question,
      example[constants.QUESTION_KEY])
    example[constants.QUERY_KEY] = tf_wrap_seq_fun(preprocess_sparql, example[constants.QUERY_KEY])
    return example


def preprocess_question(tensor: tf.Tensor):
  text = tensor.numpy().decode()
  mapped = map(lambda c: f' {c} ' if c in string.punctuation else c, text)
  preprocessed = ' '.join(''.join(mapped).split()) 
  return [preprocessed]


def preprocess_sparql(tensor: tf.Tensor):
  """Do various preprocessing on the SPARQL query."""
  query = tensor.numpy().decode()
  # Tokenize braces.
  query = query.replace('count(*)', 'count ( * )')

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


def tf_wrap_seq_fun(fun: Any, text: tf.Tensor):
  "Apply fun on a string (sequence) with py_function"
  fun_output = tf.py_function(fun, [text], Tout=tf.string)
  fun_output.set_shape(text.get_shape())
  return fun_output