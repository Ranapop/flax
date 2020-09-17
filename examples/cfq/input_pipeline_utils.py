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
"""Utils module"""
import os
import pickle
import random
import string
from typing import Iterable, List, Sequence, Dict, Text, Any, Tuple
import collections

import numpy as np
import tensorflow_text as text
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from absl import logging

import constants

tf.config.experimental.set_visible_devices([], "GPU")

def create_reversed_dataset(no_examples: int, min_len: int, max_len: int):
  inputs = []
  outputs = []
  for i in range(no_examples):
    seq_len = random.randint(min_len, max_len)
    input = np.random.choice(list(string.ascii_lowercase),  size=(seq_len))
    output = np.flip(input)
    inputs.append(' '.join(input))
    outputs.append(' '.join(output))
  data = list(zip(inputs, outputs))
  tf_data = tf.data.Dataset.from_tensor_slices(data)
  tf_data = tf_data.map(
    lambda ex: {constants.QUESTION_KEY: ex[0], constants.QUERY_KEY: ex[1]})
  return tf_data

def create_dummy_data(
    no_examples: Tuple[int,int,int] = (8000,1000,1000),
    example_length: Tuple[int, int] = tuple((20,50))
  ):
  """Create a tf dummy dataset with reversed sequences ('a b c' -> 'c b a')
  
  The dataset is created with 3 splits: 'train', 'validation' and 'test'
  Each example has
    a string input sequence (key 'question')
    a string output sequence (key 'query')

  Args:
    no_examples: Tuple of number of examples for (train,validation,test)
    example_length: The interval for the example length [min_length, max_length]
      The examples will be generated with a random length in that interval
  """
  min_len = example_length[0]
  max_len = example_length[1]
  train_data = create_reversed_dataset(no_examples[0], min_len, max_len)
  validation_data = create_reversed_dataset(no_examples[1], min_len, max_len)
  test_data = create_reversed_dataset(no_examples[2], min_len, max_len)
  return {
    'train': train_data,
    'validation': validation_data,
    'test': test_data
  }

def _get_tokens(input_features: Dict,
                tokenizer: text.Tokenizer,
                datasets: Iterable[tf.data.Dataset],
                preprocessing_fun: Any) -> Iterable[List[bytes]]:
    """Returns a list of tokens for all input fields in the given datasets."""

    def _tokenize_input_features(
            example: Dict[Text, tf.Tensor]) -> Dict[Text, tf.Tensor]:
        """Tokenizes all input features in an example."""
        example = preprocessing_fun(example)
        for feature in example:
            if feature in input_features:
                example[feature] = tokenizer.tokenize(example[feature])
        return example

    for dataset in datasets:
        # Apply the tokenizer to all input features.
        tokenized_dataset = dataset.map(_tokenize_input_features,
                                        num_parallel_calls=constants.AUTOTUNE)
        # Yield all tokenized input features (i.e., tokenized input sentences).
        for example in tfds.as_numpy(tokenized_dataset):
            for feature in input_features:
                yield example[feature]


def build_vocabulary(file_name: Text,
                     input_features: Dict,
                     tokenizer: text.Tokenizer,
                     datasets: Iterable[tf.data.Dataset],
                     special_tokens: Sequence[bytes] = (constants.PAD,
                                                        constants.UNK,
                                                        constants.BOS,
                                                        constants.EOS),
<<<<<<< HEAD
                     preprocessing_fun: Any = lambda x: x,
                     force_generation: bool = False) -> Dict[bytes, int]:
  """Returns a vocabulary of tokens with optional minimum frequency.
=======
                     preprocessing_fun: Any = lambda x: x) -> Dict[bytes, int]:
    """Returns a vocabulary of tokens with optional minimum frequency.
>>>>>>> Add multilayer LSTM for decoder + dropout on encoder and decoder
    The vocabulary is saved in the file `./temp/<file_name>`.

    Args:

        special_tokens: Special tokens that will be the start of the vocabulary.
        tokenizer: tf tokenizer (eg. WhitespaceTokenizer)
        datasets: datasets from which the vocabulary should be extracted
        special_tokens: special tokens to be added in the vocabulary; they are
                        added at the begginig with PAD at index 0
        preprocessing_fun: function for applying preprocessing on a dataset
                           entry; by default the identity function
                           (no preprocessing)
        force_generation: if True, do not used vcabulary cached in file
    Returns:
        An ordered dictionary that maps tokens to their IDs in the vocabulary.
    """

<<<<<<< HEAD
  vocab_dir = 'temp'
  vocab_path = os.path.join(vocab_dir, file_name)
  if not os.path.exists(vocab_dir):
    os.makedirs(vocab_dir)
  # Try to read the vocabulary.
  if not force_generation:
    if os.path.isfile(vocab_path):
      with open(vocab_path, 'rb') as vocab_file:
        vocab = pickle.load(vocab_file)
      return vocab

  tokens_from_datasets = _get_tokens(input_features, tokenizer, datasets,
                                     preprocessing_fun)

  # Count all the tokens.
  counter = collections.Counter()
  for tokens in tokens_from_datasets:
    counter.update(tokens)

  # Add special tokens to the start of vocab.
  vocab = collections.OrderedDict()
  for token in special_tokens:
    vocab[token] = len(vocab)
=======
    vocab_dir = 'temp'
    vocab_path = os.path.join(vocab_dir, file_name)
    if not os.path.exists(vocab_dir):
        os.makedirs(vocab_dir)
    # Try to read the vocabulary.
    if os.path.isfile(vocab_path):
        with open(vocab_path, 'rb') as vocab_file:
            vocab = pickle.load(vocab_file)
        return vocab

    tokens_from_datasets = _get_tokens(input_features,
                                       tokenizer,
                                       datasets,
                                       preprocessing_fun)

    # Count all the tokens.
    counter = collections.Counter()
    for tokens in tokens_from_datasets:
        counter.update(tokens)

    # Add special tokens to the start of vocab.
    vocab = collections.OrderedDict()
    for token in special_tokens:
        vocab[token] = len(vocab)

    # Add all other tokens to the vocab
    for token, freq in sorted(
            # Sort by frequency (from high to low), and then by token string.
            # This makes sure high frequency tokens get a low token ID.
            counter.items(),
            key=lambda token_freq: (-token_freq[1], token_freq[0])):
        vocab[token] = len(vocab)

    logging.info('Number of unfiltered tokens: %d', len(counter))
    logging.info('Vocabulary size: %d', len(vocab))

    # Save the vocabulary to disk
    with open(vocab_path, 'wb') as vocab_file:
        pickle.dump(vocab, vocab_file)
>>>>>>> Add multilayer LSTM for decoder + dropout on encoder and decoder

    return vocab


def build_tf_hashtable(vocabulary: Dict[bytes, int],
                       unk_idx: int) -> tf.lookup.StaticHashTable:
    """Returns a TF lookup table from a vocabulary."""
    return tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(
        list(vocabulary.keys()), list(vocabulary.values())),
                                     default_value=unk_idx)


def cardinality(dataset: tf.data.Dataset) -> int:
    """Returns the number of examples in the dataset by iterating over it once."""
    return dataset.reduce(np.int64(0), lambda x, _: x + 1).numpy()


def get_max_length(dataset: tf.data.Dataset, len_fn: Any):
    """Returns the max length in a dataset
  Args:
    dataset: the tensorflow dataset
    len_fn: function that gets as argument a dataset example and returns the length of that example
  """
<<<<<<< HEAD
  return dataset.reduce(
    np.int32(0),
    lambda m_len, ex: m_len if m_len > len_fn(ex) else len_fn(ex)
    ).numpy()
=======
    return dataset.reduce(
        np.int32(0),
        lambda m_len, ex: m_len if m_len > len_fn(ex) else len_fn(ex)).numpy()
>>>>>>> Add multilayer LSTM for decoder + dropout on encoder and decoder


def get_bucket_boundaries(bucket_size: int, max_size: int) -> np.ndarray:
    """Bucket boundaries with `bucket_size` items per bucket, up to `max_size`.
  Example:
  ```
  get_bucket_boundaries(8, 24)
  [9, 17, 25]
  ```
  E.g., the first boundary covers items with sizes 0-8, the next boundary covers
  items with sizes 9-16, and the last bucket covers sizes 17-24. Each bucket
  covers 8 different sizes (e.g., sentence lengths).
  Args:
   bucket_size: The number of different items per bucket.
   max_size: The maximum size to be expected for a bucket.
  Returns:
    A list of (exclusive) bucket boundaries.
  """
    return np.arange(bucket_size, max_size + bucket_size, bucket_size) + 1


def get_bucketed_batches(dataset: tf.data.Dataset,
                         batch_size: int,
                         bucket_size: int,
                         max_length: int,
                         padded_shapes: Any,
                         example_size_fn: Any,
                         seed: int = None,
                         shuffle: bool = False,
                         drop_remainder: bool = False) -> tf.data.Dataset:
    """Returns padded batches of shuffled examples bucketed by length.
  This shuffles the examples randomly each epoch. The random order is
  deterministic and controlled by the seed.
  Batches are padded because sentences have different lengths.
  Sentences that are shorter in a batch will get 0s added at the end, until
  all sentences in the batch have the same length.
  For performance, examples of similar lengths are bucketed together. However,
  the contents of the buckets and their order is random each epoch, and
  controlled by the seed.
  Args:
    dataset: A TF Dataset with examples to be shuffled and batched.
    batch_size: The size of each batch.
    bucket_size: How many different lengths go in each bucket.
    max_length: The maximum length to provide a bucket for.
    padded_shapes: A nested structure representing the shape to which the
      respective component of each input element should be padded prior to
      batching. See `tf.data.Dataset.padded_batch` for examples.
    example_size_fn: A TF function that returns the size of an example to
      determine in which bucket it goes. E.g., the sentence length.
    seed: The seed that determines the shuffling order, with a different order
      each epoch.
    shuffle: Shuffle the dataset each epoch using seed.
    drop_remainder: Drop the last batch if it is not of size batch_size.
  Returns:
    A TF Dataset containing padded bucketed batches.
  """
<<<<<<< HEAD
  if shuffle:
    assert seed is not None, 'When shuffling you must provide a seed.'

  # Multiple of bucket_size.
  max_length = max_length + bucket_size % max_length
  # For bucket_size 8 and max length 24, we get bucket boundaries [9, 17, 25].
  bucket_boundaries = get_bucket_boundaries(bucket_size, max_length)
  logging.info('Batching bucket boundaries: %r', bucket_boundaries)

  # One batch size for each bucket plus one additional one (per requirement).
  bucket_batch_sizes = [batch_size] * (len(bucket_boundaries) + 1)
  bucket_fn = tf.data.experimental.bucket_by_sequence_length(
      example_size_fn,
      bucket_boundaries,
      bucket_batch_sizes,
      padded_shapes=padded_shapes,
      pad_to_bucket_boundary=True,
      drop_remainder=drop_remainder)

  if shuffle:
    # For shuffling we need to know how many training examples we have.
    num_examples = cardinality(dataset)
    num_batches = num_examples // batch_size
    return dataset.shuffle(
        num_examples, seed=seed,
        reshuffle_each_iteration=True
        ).apply(bucket_fn).shuffle(
            num_batches, seed=seed,
            reshuffle_each_iteration=True).prefetch(constants.AUTOTUNE)
  return dataset.apply(bucket_fn).prefetch(constants.AUTOTUNE)
=======
    if shuffle:
        assert seed is not None, 'When shuffling you must provide a seed.'

    # Multiple of bucket_size.
    max_length = max_length + bucket_size % max_length
    # For bucket_size 8 and max length 24, we get bucket boundaries [9, 17, 25].
    bucket_boundaries = get_bucket_boundaries(bucket_size, max_length)
    logging.info('Batching bucket boundaries: %r', bucket_boundaries)

    # One batch size for each bucket plus one additional one (per requirement).
    bucket_batch_sizes = [batch_size] * (len(bucket_boundaries) + 1)
    bucket_fn = tf.data.experimental.bucket_by_sequence_length(
        example_size_fn,
        bucket_boundaries,
        bucket_batch_sizes,
        padded_shapes=padded_shapes,
        pad_to_bucket_boundary=True,
        drop_remainder=drop_remainder)

    if shuffle:
        # For shuffling we need to know how many training examples we have.
        num_examples = cardinality(dataset)
        num_batches = num_examples // batch_size
        return dataset.shuffle(
            num_examples, seed=seed,
            reshuffle_each_iteration=True).apply(bucket_fn).shuffle(
                num_batches, seed=seed,
                reshuffle_each_iteration=True).prefetch(constants.AUTOTUNE)
    return dataset.apply(bucket_fn).prefetch(constants.AUTOTUNE)
>>>>>>> Add multilayer LSTM for decoder + dropout on encoder and decoder
