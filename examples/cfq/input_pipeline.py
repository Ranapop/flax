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
"""CFQ input pipeline."""

# pylint: disable=too-many-arguments,import-error,too-many-instance-attributes,too-many-locals
from typing import Dict, Text, List, Tuple

from absl import logging
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import tensorflow_text as text
import jax
import jax.numpy as jnp
import numpy as np

import flax

import input_pipeline_utils as inp_utils
import preprocessing
import input_pipeline_constants as inp_constants
from input_pipeline_constants import QUESTION_KEY, QUESTION_LEN_KEY, QUERY_KEY, QUERY_LEN_KEY,\
  ACTION_TYPES_KEY, ACTION_VALUES_KEY, NODE_TYPES_KEY, PARENT_STEPS_KEY,\
  ACTION_SEQ_LEN_KEY
from syntax_based.grammar import Grammar, GRAMMAR_STR
import syntax_based.node as node 
import syntax_based.asg as asg

ExampleType = Dict[Text, tf.Tensor]


class CFQDataSource:
  """Provides the base functionality for the CFQ dataset (extracting the
  vocabulary, questions and queries preprocessing, batching). Some of the more
  specific functionality is left to the class extending this one."""

  # pylint: disable=too-few-public-methods

  def __init__(self,
               seed: int,
               fixed_output_len: bool,
               tokenizer: text.Tokenizer = text.WhitespaceTokenizer(),
               cfq_split: Text = 'mcd1',
               replace_with_dummy: bool = False):
    # Load datasets.
    if replace_with_dummy:
      # Load dummy data.
      data = inp_utils.create_dummy_data()
      vocab_file = 'dummy_vocab'
    else:
      data = tfds.load('cfq/' + cfq_split)
      vocab_file = 'vocab_' + cfq_split

    train_raw = data['train']
    dev_raw = data['validation']
    test_raw = data['test']

    # Print an example.
    logging.info('Data sample: %s', next(iter(tfds.as_numpy(train_raw.skip(4)))))

    self.tokenizer = tokenizer
    self.seed = seed
    self.fixed_output_len = fixed_output_len
    self.tokens_vocab = self.build_tokens_vocab(vocab_file,
                          tokenizer,
                          train_raw,
                          replace_with_dummy)

    self.unk_idx = self.tokens_vocab[inp_constants.UNK]
    self.bos_idx = np.dtype('uint8').type(self.tokens_vocab[inp_constants.BOS])
    self.eos_idx = self.tokens_vocab[inp_constants.EOS]
    self.tf_tokens_vocab = inp_utils.build_tf_hashtable(
                             self.tokens_vocab, self.unk_idx)
    self.tokens_vocab_size = len(self.tokens_vocab)
    self.i2w = list(self.tokens_vocab.keys())

    # Turn data examples into pre-processed examples by turning each sentence
    # into a sequence of token IDs. Also pre-prepend a beginning-of-sequence
    # token <s> and append an end-of-sequence token </s>.

    self.splits = {
      'train': train_raw.map(
        self.prepare_example, num_parallel_calls=inp_constants.AUTOTUNE).cache(),
      'dev': dev_raw.map(
        self.prepare_example, num_parallel_calls=inp_constants.AUTOTUNE).cache(),
      'test': test_raw.map(
        self.prepare_example, num_parallel_calls=inp_constants.AUTOTUNE).cache()
    }

  def build_tokens_vocab(self, vocab_file, tokenizer, dataset, dummy):
    """Build the tokens (input and output) vocabulary."""
    return inp_utils.build_vocabulary(
        file_name=vocab_file,
        input_features={QUESTION_KEY, QUERY_KEY},
        tokenizer=tokenizer,
        datasets=(dataset,),
        preprocessing_fun=preprocessing.preprocess_example,
        force_generation=dummy)

  def get_specific_padded_shapes(self, output_pad):
    """Get padding shapes for processed data (will be used in batching)."""
    raise NotImplementedError()

  def get_padded_shapes(self, output_len):
    """The padded shapes used by batching functions."""
    query_pad = None
    if self.fixed_output_len:
      query_pad = output_len
    return self.get_specific_padded_shapes(query_pad)

  def add_bos_eos(self, sequence: tf.Tensor) -> tf.Tensor:
    """Prepends BOS ID and appends EOS ID to a sequence of token IDs."""
    return tf.concat([[self.bos_idx], sequence, [self.eos_idx]], 0)

  def prepare_sequence(self, sequence: Text):
    """Prepares a sequence(question or query) by tokenizing it, transforming
    it to a list of vocabulary indices, and adding the BOS and EOS tokens."""
    tokenized_seq = self.tokenizer.tokenize(sequence)
    indices = self.tf_tokens_vocab.lookup(tokenized_seq)
    wrapped_seq = self.add_bos_eos(indices)
    return tf.cast(wrapped_seq, tf.uint8)

  def construct_new_fields(self, example: ExampleType) -> ExampleType:
    """Construct new fields (process the question and query to obtain new
    fields for each dataset entry)."""
    raise NotImplementedError()

  def prepare_example(self, example: ExampleType) -> ExampleType:
    """Prepares an example by converting to IDs and wrapping in <s> and </s>."""
    example = preprocessing.preprocess_example(example)
    example = self.construct_new_fields(example)
    return example

  def get_output_length(self, example: ExampleType) -> tf.Tensor:
    """Function that takes a dataset entry and returns the query length."""
    raise NotImplementedError()

  def get_example_length(self, example: ExampleType) -> tf.Tensor:
    """Returns the length of the example for the purpose of the bucketing
    If the output should be of fixed length (self.fixed_output_len=True),
    then the length of the example is given by the the input (question)
    length, otherwise the example length is the maximum length of the input and
    output sequences.
    """
    input_len = example[QUESTION_LEN_KEY]
    output_len = self.get_output_length(example)
    example_len = 0
    if self.fixed_output_len:
      example_len = input_len
    else:
      example_len = tf.math.maximum(input_len, output_len)
    return example_len

  def get_batches(self,
                  split: Text,
                  batch_size: int,
                  drop_remainder: bool = False,
                  shuffle: bool = False):
    """Returns an iterator with padded batches for the provided dataset."""
    dataset = self.splits[split]
    if shuffle:
      buffer_size = inp_utils.cardinality(dataset)  # number of examples.
      dataset = dataset.shuffle(buffer_size,
                                seed=self.seed,
                                reshuffle_each_iteration=True)
    if self.fixed_output_len:
      # max_output_len only needs to be computed if the output is padded
      # to a fixed length across the dataset
      max_output_len = inp_utils.get_max_length(dataset, self.get_output_length)
      padded_shapes = self.get_padded_shapes(max_output_len)
    else:
      padded_shapes = self.get_padded_shapes(None)
    dataset = dataset.padded_batch(batch_size,
                                   padded_shapes=padded_shapes,
                                   drop_remainder=drop_remainder)
    if split=='train':
      dataset = dataset.repeat()
    return dataset.prefetch(inp_constants.AUTOTUNE)

  def get_bucketed_batches(self,
                           split: Text,
                           batch_size: int,
                           bucket_size: int,
                           drop_remainder: bool = False,
                           shuffle: bool = False):
    """Returns an iterator with bucketed batches for the provided dataset."""
    dataset = self.splits[split]
    max_length = inp_utils.get_max_length(dataset, self.get_output_length)
    padded_shapes = self.get_padded_shapes(max_length)
    training = split == 'train'
    return inp_utils.get_bucketed_batches(
        dataset=dataset,
        training=training,
        batch_size=batch_size,
        bucket_size=bucket_size,
        max_length=max_length,
        padded_shapes=padded_shapes,
        example_size_fn=self.get_example_length,
        seed=self.seed,
        shuffle=shuffle,
        drop_remainder=drop_remainder)


class Seq2SeqCfqDataSource(CFQDataSource):
  """Provides the cfq data for a seq2seq model (tokenized input and output
  sequences with their lengths)."""

  def __init__(self,
               seed: int,
               fixed_output_len: bool,
               tokenizer: text.Tokenizer = text.WhitespaceTokenizer(),
               cfq_split: Text = 'mcd1',
               replace_with_dummy: bool = False):
    super().__init__(seed,
                     fixed_output_len,
                     tokenizer,
                     cfq_split,
                     replace_with_dummy)

  def get_specific_padded_shapes(self, output_pad):
    """Get padding shapes for processed data (will be used in batching)."""
    return {
        QUESTION_KEY: [None],
        QUERY_KEY: [output_pad],
        QUESTION_LEN_KEY: [],
        QUERY_LEN_KEY: []
    }
  
  def construct_new_fields(self, example: ExampleType) -> ExampleType:
    """Takes as input an example with QUESTION_KEY and QUERY_KEY and constructs
    new fields from that information."""
    example[QUESTION_KEY] = self.prepare_sequence(example[QUESTION_KEY])
    example[QUERY_KEY] = self.prepare_sequence(example[QUERY_KEY])
    example[QUESTION_LEN_KEY] = tf.size(example[QUESTION_KEY])
    example[QUERY_LEN_KEY] = tf.size(example[QUERY_KEY])
    return example

  def get_output_length(self, example: ExampleType) -> tf.Tensor:
    """Function that takes a dataset entry and returns the query length."""
    return example[QUERY_LEN_KEY]

  def indices_to_sequence_string(self,
                                 indices: jnp.ndarray,
                                 keep_padding: bool = False) -> Text:
    """Transforms a list of vocab indices into a string
    (e.g. from token indices to question/query). When keep_padding is False,
    the padding tokens are filtered out (zero-valued indices)."""
    tokens = [self.i2w[i].decode() for i in indices if keep_padding or i!=0]
    str_seq = ' '.join(tokens)
    return str_seq


class Seq2TreeCfqDataSource(CFQDataSource):
  """Provides the cfq data for a seq2tree model."""

  def __init__(self,
               seed: int,
               fixed_output_len: bool,
               tokenizer: text.Tokenizer = text.WhitespaceTokenizer(),
               cfq_split: Text = 'mcd1',
               grammar: Grammar = Grammar(GRAMMAR_STR),
               load_data: bool = True):
    self.grammar = grammar
    node_types, node_flags = grammar.collect_node_types()
    self.node_vocab = self.construct_vocab(node_types)
    self.node_vocab_size = len(node_types)
    self.rule_vocab_size = len(grammar.branches)
    self.nodes_to_action_types = self.construct_nodes_to_action_types(
      node_types, node_flags)
    expanded_nodes_list = grammar.get_expanded_nodes(self.node_vocab)
    self.expanded_nodes = self.get_expanded_nodes_array(expanded_nodes_list)
    if load_data:
      super().__init__(seed,
                       fixed_output_len,
                       tokenizer,
                       cfq_split)

  def get_expanded_nodes_array(self, expanded_nodes: List[List[int]]):
    """
    Gets as parameter a list of lists, a list of nodes expansion for each
    rule. The last list will be an empty list, found at position rule_vocab_size
    and will be used when generating a token.
    
    This list will be transformed into a 2D jnp array, with padding to not have
    a ragged shape. The length of each list will be also returned in a separate
    array.

    Returns
      (node_expansions, lengths).
    """
    no_lists = len(expanded_nodes)
    lists_lengths = [len(l) for l in expanded_nodes]
    max_list_len = max(lists_lengths)
    node_expansions_array = jnp.zeros((no_lists, max_list_len))
    for i in range(no_lists):
      l = node_expansions_array[i]
      length = lists_lengths[i]
      indexes = jax.ops.index[0:length]
      node_expansions_array = jax.ops.index_update(
        node_expansions_array, indexes, l)
    lengths_array = jnp.array(lists_lengths)
    return node_expansions_array, lengths_array 

  def construct_nodes_to_action_types(self,
                                      node_types: List[str],
                                      node_flags: List[int]):
    """Construct a dictionary node type idx -> action type.

    Args:
      node_types: list of node types.
      node_flags: flag array speciffying if a node is a rule node (1) or not (0).
    Returns:
     dict node idx -> action type.
    """
    nodes_to_action_types = np.zeros((len(node_types)))
    for i in range(len(node_types)):
      node_idx = self.node_vocab[node_types[i]]
      if node_flags[i] == 1:
        action_type = asg.APPLY_RULE
      else:
        action_type = asg.GENERATE_TOKEN
      nodes_to_action_types[i] = action_type
    return nodes_to_action_types

  def construct_vocab(self, items_list: List[str]):
    """Constructs vocabulary from list (word -> index). Assumes list contains no
    duplicates."""
    vocab = {}
    for i in range(len(items_list)):
      vocab[items_list[i]] = i
    return vocab

  def build_tokens_vocab(self, vocab_file, tokenizer, dataset, dummy):
    """Build tokens vocabulary by extracting the tokens from the questions and
    queries in the dataset, then removing the syntax tokens.
    """
    vocab = super().build_tokens_vocab(vocab_file, tokenizer, dataset, dummy)
    syntax_tokens_list = self.grammar.collect_syntax_tokens()
    for syntax_token in syntax_tokens_list:
      byte_token = syntax_token.encode()
      # Update indices in vocab.
      removed_index = vocab[byte_token]
      for word, index in vocab.items():
        if index > removed_index:
          vocab[word] -= 1
      # Delete token from vocab.
      del vocab[byte_token]
    return vocab

  def extract_data_from_act_seq(self, action_sequence: List[asg.Action]):
    """Extract from the sequence of actions a sequence of action types (at each
    position 0 for ApplyRule and 1 for GenerateRule), and a sequence of action
    values (at each position either rule branch id or token id)."""
    action_types = []
    action_values = []
    for action in action_sequence:
      action_type, action_value = action
      action_types.append(action_type)
      if action_type == asg.GENERATE_TOKEN:
        action_value = self.tokens_vocab[action_value.encode()]
      action_values.append(action_value)
    return (action_types, action_values)

  def construct_output_fields(self, query: str
                             ) -> Tuple[List[int],List[int],List[int]]:
    """Construct the output fields (action types, action values and parent
    steps) from the query.
    """
    query = query.numpy()
    query = query.decode()
    act_sequence = asg.generate_action_sequence(query, self.grammar)
    root = node.apply_sequence_of_actions(act_sequence, self.grammar)
    parent_steps = node.get_parent_time_steps(root)
    node_types = node.get_node_types(root)
    node_types = [self.node_vocab[node_type] for node_type in node_types]
    action_types, action_values = self.extract_data_from_act_seq(act_sequence)
    if len(action_types) != len(parent_steps):
      raise Exception(
              'action types and parent time steps should be of the same length,\
              got {0} and {1}'.format(len(action_types), len(parent_steps)))
    return (action_types, action_values, node_types, parent_steps)

  def construct_new_fields(self, example: ExampleType) -> ExampleType:
    """Populate the example with the 'question', 'question_len', 'action_types',
    'action_values', 'parent_steps' and 'action_seq_len' fields."""
    new_example = {}
    new_example[QUESTION_KEY] = self.prepare_sequence(example[QUESTION_KEY])
    new_example[QUESTION_LEN_KEY] = tf.size(new_example[QUESTION_KEY])
    output_fields = tf.py_function(self.construct_output_fields,
                                   [example[QUERY_KEY]],
                                   Tout=(tf.int64, tf.int64, tf.int64, tf.int64))
    new_example[ACTION_TYPES_KEY],\
      new_example[ACTION_VALUES_KEY],\
      new_example[NODE_TYPES_KEY],\
      new_example[PARENT_STEPS_KEY] = output_fields
    new_example[ACTION_SEQ_LEN_KEY] = tf.size(new_example[ACTION_TYPES_KEY])
    return new_example

  def get_output_length(self, example: ExampleType) -> tf.Tensor:
    """Get output length (action sequence length)."""
    return example[ACTION_SEQ_LEN_KEY]

  def get_specific_padded_shapes(self, output_pad):
    """Get padding shapes for processed data (will be used in batching)."""
    return {
        QUESTION_KEY: [None],
        QUESTION_LEN_KEY: [],
        ACTION_TYPES_KEY: [output_pad],
        ACTION_VALUES_KEY: [output_pad],
        NODE_TYPES_KEY: [output_pad],
        PARENT_STEPS_KEY: [output_pad],
        ACTION_SEQ_LEN_KEY: []
    }


if __name__ == '__main__':
  #TODO: remove this and add tests
  data_source = Seq2TreeCfqDataSource(seed=13467, fixed_output_len=True)
  train_batches = data_source.get_batches('train',
                                          batch_size=5,
                                          drop_remainder=False,
                                          shuffle=True)

  # Print queries
  batch_no = 1
  print('vocab size is ', data_source.tokens_vocab_size)
  print('vocab')
  print(data_source.tokens_vocab)
  for batch in tfds.as_numpy(train_batches):
    action_values_batch = batch[inp_constants.ACTION_VALUES_KEY]
    print('Batch no ',batch_no)
    for action_values in action_values_batch:
      print(action_values)
    print()
    if batch_no == 2:
      break
    batch_no+=1
    