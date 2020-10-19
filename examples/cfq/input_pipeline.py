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
from typing import Dict, Text

from absl import logging
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import tensorflow_text as text
import jax.numpy as jnp
import numpy as np

import input_pipeline_utils as inp_utils
import preprocessing
import constants

ExampleType = Dict[Text, tf.Tensor]


class CFQDataSource:
  """Provides CFQ data as pre-processed batches, a vocab, and embeddings."""

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
    logging.info('Data sample: %s', next(tfds.as_numpy(train_raw.skip(4))))

    self.tokenizer = tokenizer
    self.seed = seed
    self.fixed_output_len = fixed_output_len
    self.vocab = inp_utils.build_vocabulary(
        file_name=vocab_file,
        input_features={constants.QUESTION_KEY, constants.QUERY_KEY},
        tokenizer=tokenizer,
        datasets=(train_raw,),
        preprocessing_fun=preprocessing.preprocess_example,
        force_generation=replace_with_dummy)

    self.unk_idx = self.vocab[constants.UNK]
    self.bos_idx = np.dtype('uint8').type(self.vocab[constants.BOS])
    self.eos_idx = self.vocab[constants.EOS]
    self.tf_vocab = inp_utils.build_tf_hashtable(self.vocab, self.unk_idx)
    self.vocab_size = len(self.vocab)
    self.i2w = list(self.vocab.keys())

    # Turn data examples into pre-processed examples by turning each sentence
    # into a sequence of token IDs. Also pre-prepend a beginning-of-sequence
    # token <s> and append an end-of-sequence token </s>.

    self.splits = {
      'train': train_raw.map(
        self.prepare_example, num_parallel_calls=constants.AUTOTUNE).cache(),
      'dev': dev_raw.map(
        self.prepare_example, num_parallel_calls=constants.AUTOTUNE).cache(),
      'test': test_raw.map(
        self.prepare_example, num_parallel_calls=constants.AUTOTUNE).cache()
    }

  def get_padded_shapes(self, output_len):
    """The padded shapes used by batching functions."""
    query_pad = None
    if self.fixed_output_len:
      query_pad = output_len
    return {
        constants.QUESTION_KEY: [None],
        constants.QUERY_KEY: [query_pad],
        constants.QUESTION_LEN_KEY: [],
        constants.QUERY_LEN_KEY: []
    }

  def add_bos_eos(self, sequence: tf.Tensor) -> tf.Tensor:
    """Prepends BOS ID and appends EOS ID to a sequence of token IDs."""
    return tf.concat([[self.bos_idx], sequence, [self.eos_idx]], 0)

  def prepare_sequence(self, sequence: Text):
    """Prepares a sequence(question or query) by tokenizing it, transforming
        it to a list of vocabulary indices, and adding the BOS and EOS tokens"""
    tokenized_seq = self.tokenizer.tokenize(sequence)
    indices = self.tf_vocab.lookup(tokenized_seq)
    wrapped_seq = self.add_bos_eos(indices)
    return tf.cast(wrapped_seq, tf.uint8)

  def prepare_example(self, example: ExampleType) -> ExampleType:
    """Prepares an example by converting to IDs and wrapping in <s> and </s>."""
    example = preprocessing.preprocess_example(example)
    example[constants.QUESTION_KEY] = self.prepare_sequence(
        example[constants.QUESTION_KEY])
    example[constants.QUERY_KEY] = self.prepare_sequence(
        example[constants.QUERY_KEY])
    example[constants.QUESTION_LEN_KEY] = tf.size(
        example[constants.QUESTION_KEY])
    example[constants.QUERY_LEN_KEY] = tf.size(example[constants.QUERY_KEY])
    return example

  def get_output_length(self, example: ExampleType) -> tf.Tensor:
    """Function that takes a dataset entry and returns the query length"""
    return example[constants.QUERY_LEN_KEY]

  def get_example_length(self, example: ExampleType) -> tf.Tensor:
    """Returns the length of the example for the purpose of the bucketing
        If the output should be of fixed length (self.fixed_output_len=True),
        then the length of the example is given by the the input (question)
        length, otherwise the example length is the maximum length of the 2 
        sequences (input and output)
        """
    question_len = example[constants.QUESTION_LEN_KEY]
    query_len = example[constants.QUERY_LEN_KEY]
    example_len = 0
    if self.fixed_output_len:
      example_len = question_len
    else:
      example_len = tf.math.maximum(question_len, query_len)
    return example_len

  def indices_to_sequence_string(self,
                                 indices: jnp.ndarray,
                                 keep_padding: bool = False) -> Text:
    """Transforms a list of vocab indices into a string
        (e.g. from token indices to question/query). When keep_padding is False,
        the padding tokens are filtered out (zero-valued indices)."""
    tokens = [self.i2w[i].decode() for i in indices if keep_padding or i!=0]
    str_seq = ' '.join(tokens)
    return str_seq

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
    return dataset.prefetch(constants.AUTOTUNE)

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


if __name__ == '__main__':
  #TODO: remove this and add tests
  data_source = CFQDataSource(seed=13467, fixed_output_len=True)
  train_batches = data_source.get_batches('train',
                                          batch_size=5,
                                          drop_remainder=False,
                                          shuffle=True)

  # Print queries
  batch_no = 1
  for batch in tfds.as_numpy(train_batches):
    queries = batch[constants.QUERY_KEY]
    print('Batch no ',batch_no)
    for query in queries:
      print(data_source.indices_to_sequence_string(query))
    print()
    if batch_no == 30:
      break
    batch_no+=1
    