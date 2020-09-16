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
"""Module for training/evaluation (metrics, loss, train, evaluate)"""

from typing import Any, Text, Dict
from absl import logging

import tensorflow_datasets as tfds
import tensorflow.compat.v2 as tf
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

import flax
from flax import jax_utils
from flax import nn
from flax.training import checkpoints
from flax.training import common_utils

import input_pipeline as inp
import constants
import models

BatchType = Dict[Text, jnp.array]

ACC_KEY = 'accuracy'
LOSS_KEY = 'loss'
TRAIN_ACCURACIES = 'train acc'
TRAIN_LOSSES = 'train loss'
TEST_ACCURACIES = 'test acc'
TEST_LOSSES = 'test loss'


# vmap?
def indices_to_str(batch_inputs: jnp.ndarray, data_source: inp.CFQDataSource):
  """Decode a batch of one-hot encoding to strings."""
  return np.array(
      [data_source.indices_to_sequence_string(seq) for seq in batch_inputs])


def mask_sequences(sequence_batch: jnp.array, lengths: jnp.array):
  """Set positions beyond the length of each sequence to 0."""
  return sequence_batch * (lengths[:, jnp.newaxis] > jnp.arange(
      sequence_batch.shape[1]))


def create_model(vocab_size: int) -> nn.Module:
  """Creates a seq2seq model."""
  _, initial_params = models.Seq2seq.partial(
      vocab_size=vocab_size
  ).init_by_shape(
      nn.make_rng(),
      #encoder_inputs
      [
          ((1, 1), jnp.uint8),
          #decoder_inputs
          # need to pass 2 for decoder length
          # as the first token is cut off
          ((1, 2), jnp.uint8),
          ((1,), jnp.uint8)
      ])
  model = nn.Model(models.Seq2seq, initial_params)
  return model


def cross_entropy_loss(logits: jnp.array, labels: jnp.array,
                       lengths: jnp.array):
  """Returns cross-entropy loss."""
  log_soft = nn.log_softmax(logits)
  log_sum = jnp.sum(log_soft * labels, axis=-1)
  masked_log_sum = jnp.mean(mask_sequences(log_sum, lengths))
  return -masked_log_sum


def pad_batch_to_max(batch: jnp.array, seq_len: int, max_len: int):
  """Pads the input array on the 2nd dimension to the given length
    (padding is done with 0)

    Args:
      batch: batch array
      seq_len: 2nd dimension current value
      max_len: 2nd dimension desired value
    """
  padding_size = max_len - seq_len
  padding = tf.constant([[0, 0], [0, padding_size], [0, 0]])
  return tf.pad(batch, padding, "CONSTANT").numpy()


def compute_metrics(logits: jnp.array, labels: jnp.array,
                    queries_lengths: jnp.array) -> Dict:
  """Computes metrics for a batch of logist & labels and returns those metrics

    The metrics computed are cross entropy loss and mean batch accuracy. The
    accuracy at sequence level needs perfect matching of the compared sequences
    Args:
      logits: logits (train time) or ohe predictions (test time)
              [batch_size, logits seq_len, vocab_size]
      labels: ohe gold labels, shape [batch_size, labels seq_len, vocab_size]
      queries_lengths: lengths of gold queries (until eos)

    """
  lengths = queries_lengths
  labels_seq_len = labels.shape[1]
  logits_seq_len = logits.shape[1]
  max_seq_len = max(labels_seq_len, logits_seq_len)
  if labels_seq_len != max_seq_len:
    labels = pad_batch_to_max(labels, labels_seq_len, max_seq_len)
  elif logits_seq_len != max_seq_len:
    logits = pad_batch_to_max(logits, logits_seq_len, max_seq_len)

  loss = cross_entropy_loss(logits, labels, lengths)
  token_accuracy = jnp.argmax(logits, -1) == jnp.argmax(labels, -1)
  sequence_accuracy = (jnp.sum(mask_sequences(token_accuracy, lengths),
                               axis=-1) == lengths)
  accuracy = jnp.mean(sequence_accuracy)
  metrics = {
      'loss': loss,
      'accuracy': accuracy,
  }
  return metrics


def log(epoch: int, train_metrics: Dict, dev_metrics: Dict):
  """Logs performance for an epoch.

    Args:
      epoch: The epoch number.
      train_metrics: A dict with the train metrics for this epoch.
      dev_metrics: A dict with the validation metrics for this epoch.
    """
  logging.info(
      'Epoch %02d train loss %.4f dev loss %.4f train acc %.2f dev acc %.2f',
      epoch + 1, train_metrics[LOSS_KEY], dev_metrics[LOSS_KEY],
      train_metrics[ACC_KEY], dev_metrics[ACC_KEY])


@jax.partial(jax.jit, static_argnums=3)
def train_step(optimizer: Any, batch: BatchType, rng: Any, vocab_size: int):
  """Train one step."""

  inputs = batch[constants.QUESTION_KEY]
  input_lengths = batch[constants.QUESTION_LEN_KEY]
  labels = batch[constants.QUERY_KEY]
  labels_no_bos = labels[:, 1:]
  queries_lengths = batch[constants.QUERY_LEN_KEY]

  def loss_fn(model):
    """Compute cross-entropy loss."""
    with nn.stochastic(rng):
      logits, predictions = model(encoder_inputs=inputs,
                                  decoder_inputs=labels,
                                  encoder_inputs_lengths=input_lengths,
                                  vocab_size=vocab_size)
    loss = cross_entropy_loss(logits,
                              common_utils.onehot(labels_no_bos, vocab_size),
                              queries_lengths)
    return loss, logits

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, logits), grad = grad_fn(optimizer.target)
  optimizer = optimizer.apply_gradient(grad)
  metrics = {}
  metrics = compute_metrics(logits,
                            common_utils.onehot(labels_no_bos, vocab_size),
                            queries_lengths)
  return optimizer, metrics


@jax.partial(jax.jit, static_argnums=[4, 6])
def infer(model: nn.Module, inputs: jnp.array, inputs_lengths: jnp.array,
          rng: Any, vocab_size: int, bos_encoding: jnp.array,
          predicted_output_length: int):
  """Apply model on inference flow and return predictions.

    Args:
        model: the seq2seq model applied
        inputs: batch of input sequences
        rng: rng
        vocab_size: size of vocabulary
        bos_encoding: id the BOS token
        predicted_output_length: what length should predict for the output
                                 (should be a static argnum)
    """
  # This simply creates a batch (batch size = inputs.shape[0])
  # filled with sequences of max_output_len of only the bos encoding
  initial_dec_inputs = jnp.tile(bos_encoding,
                                (inputs.shape[0], predicted_output_length))
  with nn.stochastic(rng):
    logits, predictions = model(encoder_inputs=inputs,
                                decoder_inputs=initial_dec_inputs,
                                encoder_inputs_lengths=inputs_lengths,
                                vocab_size=vocab_size,
                                teacher_force=False)
  return logits, predictions


def evaluate_model(model: nn.Module,
                   batches: tf.data.Dataset,
                   data_source: inp.CFQDataSource,
                   predicted_output_length: int,
                   no_logged_examples: int = None):
  """Evaluate the model on the validation/test batches

    Args:
        model: model
        batches: validation batches
        data_source: CFQ data source (needed for vocab size, w2i etc.)
        predicted_output_length: how long the predicted sequence should be
        no_logged_examples: how many examples to log (they will be taken
                            from the first batch, so no_logged_examples
                            should be < batch_size)
                            if None, no logging
    """
  no_batches = 0
  avg_metrics = {ACC_KEY: 0, LOSS_KEY: 0}
  for batch in tfds.as_numpy(batches):
    inputs = batch[constants.QUESTION_KEY]
    input_lengths = batch[constants.QUESTION_LEN_KEY]
    gold_outputs = batch[constants.QUERY_KEY][:, 1:]
    _, inferred_outputs = infer(model, inputs, input_lengths, nn.make_rng(),
                                data_source.vocab_size, data_source.bos_idx,
                                predicted_output_length)
    ohe_predictions = common_utils.onehot(inferred_outputs,
                                          data_source.vocab_size)
    metrics = compute_metrics(
        ohe_predictions,
        common_utils.onehot(gold_outputs, data_source.vocab_size),
        batch[constants.QUERY_LEN_KEY])
    avg_metrics = {key: avg_metrics[key] + metrics[key] for key in avg_metrics}
    if no_logged_examples is not None and no_batches == 0:
      #log the first examples in the batch
      gold_seq = indices_to_str(gold_outputs, data_source)
      inferred_seq = indices_to_str(inferred_outputs, data_source)
      for i in range(0, no_logged_examples):
        logging.info('\nGold seq:\n %s\nInferred seq:\n %s\n', gold_seq[i],
                     inferred_seq[i])
    no_batches += 1
  avg_metrics = {key: avg_metrics[key] / no_batches for key in avg_metrics}
  return avg_metrics


def plot_metrics(metrics: Dict[Text, float], no_epochs):
  """Plot metrics and save figs in temp"""
  x = range(1, no_epochs + 1)

  # plot accuracies
  plt.plot(x, metrics[TRAIN_ACCURACIES], label='train acc')
  plt.plot(x, metrics[TEST_ACCURACIES], label='test acc')
  plt.xlabel('Epoch')
  plt.title("Accuracies")
  plt.legend()
  plt.show()
  plt.savefig('temp/accuracies.png')
  plt.clf()
  # plot losses
  plt.plot(x, metrics[TRAIN_LOSSES], label='train loss')
  plt.plot(x, metrics[TEST_LOSSES], label='test loss')
  plt.xlabel('Epoch')
  plt.title("Losses")
  plt.legend()
  plt.show()
  plt.savefig('temp/losses.png')
  plt.clf()


def train_model(learning_rate: float = None,
                num_epochs: int = None,
                max_out_len: int = None,
                seed: int = None,
                data_source: inp.CFQDataSource = None,
                batch_size: int = None,
                bucketing: bool = False,
                model_dir=None):
  """ Train model on num_epochs

    Do the training on data_source.train_dataset and evaluate on
    data_source.dev_dataset at each epoch and log the results
    """

  with nn.stochastic(jax.random.PRNGKey(seed)):
    model = create_model(data_source.vocab_size)
    optimizer = flax.optim.Adam(learning_rate=learning_rate).create(model)

    if bucketing:
      train_batches = data_source.get_bucketed_batches(
          data_source.train_dataset,
          batch_size=batch_size,
          bucket_size=8,
          drop_remainder=False,
          shuffle=True)
    else:
      train_batches = data_source.get_batches(data_source.train_dataset,
                                              batch_size=batch_size,
                                              shuffle=True)

    dev_batches = data_source.get_batches(data_source.dev_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)

    train_metrics = {ACC_KEY: 0, LOSS_KEY: 0}
    metrics_per_epoch = {
        TRAIN_ACCURACIES: [],
        TRAIN_LOSSES: [],
        TEST_ACCURACIES: [],
        TEST_LOSSES: []
    }
    for epoch in range(num_epochs):
      no_batches = 0
      for batch in tfds.as_numpy(train_batches):
        batch = jax.tree_map(lambda x: x, batch)
        optimizer, metrics = train_step(optimizer, batch, nn.make_rng(),
                                        data_source.vocab_size)
        train_metrics = {
            key: train_metrics[key] + metrics[key] for key in train_metrics
        }
        no_batches += 1
        # only train for 1 batch (for now)
        # if no_batches == 1:
        #     break
      train_metrics = {
          key: train_metrics[key] / no_batches for key in train_metrics
      }
      # evaluate
      dev_metrics = evaluate_model(model=optimizer.target,
                                   batches=dev_batches,
                                   data_source=data_source,
                                   predicted_output_length=max_out_len,
                                   no_logged_examples=3)
      log(epoch, train_metrics, dev_metrics)
      metrics_per_epoch[TRAIN_ACCURACIES].append(train_metrics[ACC_KEY])
      metrics_per_epoch[TRAIN_LOSSES].append(train_metrics[LOSS_KEY])
      metrics_per_epoch[TEST_ACCURACIES].append(dev_metrics[ACC_KEY])
      metrics_per_epoch[TEST_LOSSES].append(dev_metrics[LOSS_KEY])

    plot_metrics(metrics_per_epoch, num_epochs)
    logging.info('Done training')

  #is it ok to only save at the end?
  if model_dir is not None:
    logging.info('Saving model at %s', model_dir)
    checkpoints.save_checkpoint(model_dir, optimizer, num_epochs)

  return optimizer.target


def test_model(model_dir, data_source: inp.CFQDataSource, max_out_len: int,
               seed: int, batch_size: int):
  """Evaluate model at model_dir on dev subset"""
  with nn.stochastic(jax.random.PRNGKey(seed)):
    model = create_model(data_source.vocab_size)
    optimizer = flax.optim.Adam().create(model)
    dev_batches = data_source.get_batches(data_source.dev_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)
    # evaluate
    dev_metrics = evaluate_model(model=optimizer.target,
                                 batches=dev_batches,
                                 data_source=data_source,
                                 predicted_output_length=max_out_len,
                                 no_logged_examples=3)
    logging.info('Loss %.4f, acc %.2f', dev_metrics[LOSS_KEY],
                 dev_metrics[ACC_KEY])
