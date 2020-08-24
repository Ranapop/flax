
import tensorflow_text as text
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from typing import Iterable, List, Sequence, Dict, Text, Any
import numpy as np
import collections
from absl import logging
import os
import pickle
import constants



def _get_tokens(
            input_features,
            tokenizer: text.Tokenizer,
            datasets: Iterable[tf.data.Dataset]) -> Iterable[List[bytes]]:
    """Returns a list of tokens for all input fields in the given datasets."""

    def _tokenize_input_features(example: Dict[Text, tf.Tensor]) -> Dict[Text, tf.Tensor]:
      """Tokenizes all input features in an example."""
      for feature in example:
        if feature in input_features:
          example[feature] = tokenizer.tokenize(example[feature])
      return example

    for dataset in datasets:
      # Apply the tokenizer to all input features.
      tokenized_dataset = dataset.map(
          _tokenize_input_features, num_parallel_calls=constants.AUTOTUNE)
      # Yield all tokenized input features (i.e., tokenized input sentences).
      for example in tfds.as_numpy(tokenized_dataset):
        for feature in input_features:
          yield example[feature]


def build_vocabulary(input_features,
            tokenizer: text.Tokenizer,
            datasets: Iterable[tf.data.Dataset],
            special_tokens: Sequence[bytes] = (constants.PAD, constants.UNK,
                                            constants.BOS, constants.EOS),
            min_freq: int = 1) -> Dict[bytes, int]:
    """Returns a vocabulary of tokens with optional minimum frequency.
    Args:
    TODO: redo comment
        sequences: An iterable with sequences of tokens.
        special_tokens: Special tokens that will be the start of the vocabulary.
        min_freq: The minimum frequency of each token to be included. Default: 1.
    Returns:
        An ordered dictionary that maps tokens to their IDs in the vocabulary.
    """

    vocab_path = 'dumps/vocab'
    if not os.path.exists('dumps'):
      os.makedirs('dumps')
    # try to read the vocab
    if os.path.isfile(vocab_path):
      with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
        f.close()
        return vocab

    sequences = _get_tokens(input_features, tokenizer, datasets)

    # Count all the tokens.
    counter = collections.Counter()
    for tokens in sequences:
      counter.update(tokens)

    # Add special tokens to the start of vocab.
    vocab = collections.OrderedDict()
    for token in special_tokens:
      vocab[token] = len(vocab)

    # Add all other tokens to the vocab if their frequency is >= min_freq.
    for token, freq in sorted(
      # Sort by frequency (from high to low), and then by token string.
      # This makes sure high frequency tokens get a low token ID.
      counter.items(),
      key=lambda token_freq: (-token_freq[1], token_freq[0])):
      if freq >= min_freq:
        vocab[token] = len(vocab)

    logging.info('Number of unfiltered tokens: %d', len(counter))
    logging.info('Vocabulary size: %d', len(vocab))

    # persist the vocab
    f = open(vocab_path, 'wb')
    pickle.dump(vocab,f)
    f.close()

    return vocab


def build_tf_hashtable(vocabulary: Dict[bytes, int],
                       unk_idx: int) -> tf.lookup.StaticHashTable:
  """Returns a TF lookup table from a vocabulary."""
  return tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(
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
  # why is the length field of type int32??
  return dataset.reduce(np.int32(0), lambda old_max, ex: old_max if old_max > len_fn(ex) else len_fn(ex)).numpy()

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

def get_bucketed_batches(
    dataset: tf.data.Dataset,
    batch_size: int,
    bucket_size: int,
    max_length: int,
    padded_shapes: Any,
    example_size_fn: Any,
    seed: int = None,
    shuffle: bool = False,
    drop_remainder: bool = False,
) -> tf.data.Dataset:
  """Returns padded batches of shuffled CFQ examples bucketed by length.
  This shuffles the examples randomly each epoch. The random order is
  deterministic and controlled by the seed.
  Batches are padded because sentences have different lengths.
  Sentences that are shorter in a batch will get 0s added at the end, until
  all sentences in the batch have the same length.
  For performance, examples of similar lengths are bucketed together. However,
  the contents of the buckets and their order is random each epoch, and
  controlled by the seed.
  Args:
    dataset: A TF Dataset with SST examples to be shuffled and batched.
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
  if shuffle:
    assert seed is not None, 'When shuffling you must provide a seed.'

  # For bucket_size 8 and max length 24, we get bucket boundaries [9, 17, 25].
  max_length = max_length + bucket_size % max_length  # Multiple of bucket_size.
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
