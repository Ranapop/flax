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

import tensorflow_datasets as tfds
import tensorflow.compat.v2 as tf

from absl import flags
from absl import app
from absl import logging

from typing import Sequence, Text, Dict

import collections

EOS = '<eos>'
UNK = '<unk>'
PAD = '<pad>'

flags.DEFINE_string(
    'cfq_split',
    default='mcd1',
    help=('The cfq split'))

def build_vocab(datasets: Sequence[tf.data.Dataset],
                special_tokens: Sequence[Text] = (PAD, EOS),
                min_freq: int = 0) -> Dict[Text, int]:
  """Returns a vocabulary of tokens with optional minimum frequency."""
  # Count the tokens in the datasets.
  counter = collections.Counter()
  for dataset in datasets:
    for example in tfds.as_numpy(dataset):
      counter.update(whitespace_tokenize(example['question']))

  # Add special tokens to the start of vocab.
  vocab = collections.OrderedDict()
  for token in special_tokens:
    vocab[token] = len(vocab)

  # Add all other tokens to the vocab if their frequency is >= min_freq.
  for token in sorted(list(counter.keys())):
    if counter[token] >= min_freq:
      vocab[token] = len(vocab)

  logging.info('Number of unfiltered tokens: %d', len(counter))
  logging.info('Vocabulary size: %d', len(vocab))
  return vocab

def whitespace_tokenize(text: Text) -> Sequence[Text]:
  """Splits an input into tokens by whitespace."""
  return text.strip().split()

def process_sequence(seq: Text):
    # tokenize
    print(seq)
    tokenized = whitespace_tokenize(seq.numpy())
    # append end of sequence token
    tokenized_numpy = tokenized.numpy()
    tokenized_with_eos = tf.concat(tokenized_numpy,[EOS])

def process_example(example: Dict[Text, Text]):
    example['question'] = process_sequence(example['question'])

def process_split(ds:tf.data.Dataset, is_train=False, ds_info=None):
    # should I also save one hot encodings??
    #TODO: map processings here
    ds = ds.map(process_example)
    ds = ds.cache()
    if is_train:
        ds.shuffle(ds_info.splits['train'].num_examples)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

def load_data():

    cfq_path = 'cfq/'+flags.FLAGS.cfq_split

    train_split, train_info = tfds.load(cfq_path, split='train',with_info=True)
    #construct vocab
    vocab = build_vocab((train_split,), min_freq=0)
    # validation_split, validation_info = tfds.load(cfq_path, split='validation', with_info=True)

    process_split(train_split, True, train_info)
    # process_split(train_split)

def main(_):
    # tf.enable_eager_execution()
    load_data()
    # print(tfds.load('cfq/mcd1:1.2.0').keys())


if __name__ == '__main__':
  app.run(main)

