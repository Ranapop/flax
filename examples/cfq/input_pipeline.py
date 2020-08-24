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
import collections
from typing import Dict, Sequence, Text
from absl import logging
import os
import pickle
import numpy as np

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import tensorflow_text as text

import utils
import constants

ExampleType = Dict[Text, tf.Tensor]


class CFQDataSource:
  """Provides SST-2 data as pre-processed batches, a vocab, and embeddings."""
  # pylint: disable=too-few-public-methods

  def __init__(self, 
              seed: int = None,
              tokenizer = text.WhitespaceTokenizer(),
              max_output_length = None,
              cfq_split='mcd1',
              min_freq: int = 0):
    # Load datasets.
    data = tfds.load('cfq/'+cfq_split)
    train_raw = data['train']
    valid_raw = data['validation']
    test_raw = data['test']

    # Print an example.
    logging.info('Data sample: %s', next(tfds.as_numpy(train_raw.skip(4))))


    self.tokenizer = tokenizer
    self.seed = seed
    self.max_output_length = max_output_length
    # Get a vocabulary and a corresponding GloVe word embedding matrix.
    # self.vocab = build_vocab((train_raw,), min_freq=min_freq)
    self.vocab = utils.build_vocabulary(input_features = {constants.QUESTION_KEY, constants.QUERY_KEY},
                              tokenizer = tokenizer,
                              datasets = (train_raw,))

    self.unk_idx = self.vocab[constants.UNK]
    self.bos_idx = self.vocab[constants.BOS]
    self.eos_idx = self.vocab[constants.EOS]
    self.tf_vocab = utils.build_tf_hashtable(self.vocab, self.unk_idx)
    self.vocab_size = len(self.vocab)
    self.i2w = list(self.vocab.keys())

    # Turn data examples into pre-processed examples by turning each sentence
    # into a sequence of token IDs. Also pre-prepend a beginning-of-sequence
    # token <s> and append an end-of-sequence token </s>.

    self.train_dataset = train_raw.map(
        self.prepare_example, num_parallel_calls=constants.AUTOTUNE).cache()
    self.valid_dataset = valid_raw.map(
        self.prepare_example, num_parallel_calls=constants.AUTOTUNE).cache()
    self.test_dataset = test_raw.map(
        self.prepare_example, num_parallel_calls=constants.AUTOTUNE).cache()
    
    self.max_length_train = utils.get_max_length(self.train_dataset, self.example_length_fn)
    self.max_length_test = utils.get_max_length(self.test_dataset, self.example_length_fn)
    self.max_length_valid = utils.get_max_length(self.valid_dataset, self.example_length_fn)


  @property
  def padded_shapes(self):
    """The padded shapes used by batching functions."""
    query_pad = self.max_output_length
    return {
              constants.QUESTION_KEY: [None],
              constants.QUERY_KEY: [query_pad],
              constants.QUESTION_LEN_KEY: [],
              constants.QUERY_LEN_KEY: []
          }

  def add_bos_eos(self, sequence: tf.Tensor) -> tf.Tensor:
    """Prepends BOS ID and appends EOS ID to a sequence of token IDs."""
    return tf.concat([[self.bos_idx], sequence, [self.eos_idx]], 0)

  def prepare_sequence(self, sequence: Text, max_length = None):
    tokenized_seq = self.tf_vocab.lookup(
      self.tokenizer.tokenize(sequence))
    # truncate if needed
    if max_length is not None:
      # -2 because of the BOS and EOS tokens
      tokenized_seq = tokenized_seq[0:max_length-2]
    wrapped_seq = self.add_bos_eos(tokenized_seq)
    return wrapped_seq

  def prepare_example(self, example: ExampleType) -> ExampleType:
    """Prepares an example by converting to IDs and wrapping in <s> and </s>."""
    example[constants.QUESTION_KEY] = self.prepare_sequence(example[constants.QUESTION_KEY])
    example[constants.QUERY_KEY] = self.prepare_sequence(example[constants.QUERY_KEY], self.max_output_length)
    example[constants.QUESTION_LEN_KEY] = tf.size(example[constants.QUESTION_KEY])
    example[constants.QUERY_LEN_KEY] = tf.size(example[constants.QUERY_KEY])
    return example

  def example_length_fn(self, example: ExampleType) -> tf.Tensor:
    """Returns the length of the example for the purpose of the bucketing."""
    question_len = example[constants.QUESTION_LEN_KEY]
    query_len = example[constants.QUERY_LEN_KEY]
    # if I have a fixed output length, I only care about the input length when batching
    if self.max_output_length is not None:
      return question_len
    # otherwise I take as the example length the maximum of the two sequence' lengths (input and output)
    else:
      return tf.math.maximum(question_len, query_len)

  def get_batches(self,
                  dataset: tf.data.Dataset,
                  batch_size: int,
                  drop_remainder: bool = False,
                  shuffle: bool = False):
    """Returns an iterator with padded batches for the provided dataset."""
    if shuffle:
      buffer_size = utils.cardinality(dataset)  # The number of examples.
      dataset = dataset.shuffle(
          buffer_size, seed=self.seed, reshuffle_each_iteration=True)
    return dataset.padded_batch(
        batch_size,
        padded_shapes=self.padded_shapes,
        drop_remainder=drop_remainder)

  def get_bucketed_batches(self,
                           dataset: tf.data.Dataset,
                           batch_size: int,
                           bucket_size: int,
                           drop_remainder: bool = False,
                           shuffle: bool = False):
    """Returns an iterator with bucketed batches for the provided dataset."""
    return utils.get_bucketed_batches(
        dataset,
        batch_size,
        bucket_size,
        self.max_length_train,
        self.padded_shapes,
        self.example_length_fn,
        seed=self.seed,
        shuffle=shuffle,
        drop_remainder=drop_remainder)

if __name__ == '__main__':
    #TODO: remove this and add tests
    data_source = CFQDataSource(seed = 13467, max_output_length = 10)
    # train_batches = get_batches(
    #     data_source.train_dataset, batch_size=5)
    # train_batches = data_source.get_batches(data_source.train_dataset,
    #                                         batch_size = 5,
    #                                         drop_remainder = False,
    #                                         shuffle = True)
    train_batches = data_source.get_bucketed_batches(data_source.train_dataset,
                                            batch_size = 20,
                                            bucket_size = 8,
                                            drop_remainder = False,
                                            shuffle = True)
    batch = next(tfds.as_numpy(train_batches.skip(4)))
    questions, queries, lengths = batch[constants.QUESTION_KEY], batch[constants.QUERY_KEY], batch[constants.QUESTION_LEN_KEY]
    questions_strings = []
    for question in questions:
      qt = [data_source.i2w[i] for i in question]
      qt_str = [token.decode() for token in qt]
      qs = ' '.join(qt_str) 
      questions_strings.append(qs)
    print('Questions')
    print(questions)
    print('Questions (tokens)')
    print(questions_strings)
    print('Queries')
    print(queries)
    print('Lengths')
    print(lengths)