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

import os
import shutil
from typing import Any, Text, Dict, TextIO
from absl import logging

import tensorflow_datasets as tfds
import tensorflow.compat.v2 as tf
import numpy as np
import matplotlib.pyplot as plt

import functools
import jax
import jax.numpy as jnp
from jax.experimental.optimizers import clip_grads

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
  mask = (lengths[:, jnp.newaxis] > jnp.arange(sequence_batch.shape[1]))
  return sequence_batch * mask


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
                       lengths: jnp.array, vocab_size: int):
  """Returns cross-entropy loss."""
  labels = common_utils.onehot(labels, vocab_size)
  log_soft = nn.log_softmax(logits)
  log_sum = jnp.sum(log_soft * labels, axis=-1)
  masked_log_sums = jnp.sum(mask_sequences(log_sum, lengths))
  mean_losses = jnp.divide(masked_log_sums, lengths)
  mean_loss = jnp.mean(mean_losses)
  return -mean_loss


def pad_along_axis(array: jnp.array,
                   padding_size: int,
                   axis: int) -> jnp.array:
  """Returns array padded with zeroes on given axis with padding_size positions.
  The padding is done at the end of the array at that axis."""
  # The pad function expects a shape of size (n, 2) where n is the number of
  # axes. For each axis, the number of padding positions at the beginning and
  # end of the array is specified.
  pad_shape = jnp.zeros((array.ndim, 2), dtype=jnp.int32)
  pad_shape = jax.ops.index_update(pad_shape,
                                   jax.ops.index[axis, 1],
                                   padding_size)
  padded = jnp.pad(array, pad_shape)
  return padded


def compute_perfect_match_accuracy(predictions: jnp.array,
                                   labels: jnp.array,
                                   lengths: jnp.array) -> jnp.array:
    """Compute perfect match accuracy.
    
    This function computes the mean accuracy at batch level - averaged sequence
    level accuracies. At sequence level the accuracy is the perfect match
    accuracy: 1 if the sequences are equal, 0 otherwise (so 0 for partially
    matching). Also, the sequences may be padded and the lengths of the gold
    sequences are used to only compare sequences until the <eos>.
    Args:
      predictions: predictions [batch_size, predicted seq len]
      labels: ohe gold labels, shape [batch_size, labels seq_len]
    returns:
      accuracy [batch_size]
    """
    token_accuracy = jnp.equal(predictions, labels)
    sequence_accuracy = (jnp.sum(mask_sequences(token_accuracy, lengths),
                               axis=-1) == lengths)
    accuracy = jnp.mean(sequence_accuracy)
    return accuracy                  


def compute_metrics(logits: jnp.array,
                    predictions: jnp.array,
                    labels: jnp.array,
                    queries_lengths: jnp.array,
                    vocab_size: int) -> Dict:
  """Computes metrics for a batch of logist & labels and returns those metrics

    The metrics computed are cross entropy loss and mean batch accuracy. The
    accuracy at sequence level needs perfect matching of the compared sequences
    Args:
      logits: logits (train time) or ohe predictions (test time)
              [batch_size, logits seq_len, vocab_size]
      predictions: predictions [batch_size, predicted seq len]
      labels: ohe gold labels, shape [batch_size, labels seq_len]
      queries_lengths: lengths of gold queries (until eos) [batch_size]
      vocab_size: vocabulary size

    """
  lengths = queries_lengths
  labels_seq_len = labels.shape[1]
  logits_seq_len = logits.shape[1]
  max_seq_len = max(labels_seq_len, logits_seq_len)
  if labels_seq_len != max_seq_len:
    labels = pad_along_axis(labels, max_seq_len - labels_seq_len, 1)
  elif logits_seq_len != max_seq_len:
    padding_size = max_seq_len - logits_seq_len
    logits = pad_along_axis(logits, padding_size, 1)
    predictions = pad_along_axis(predictions, padding_size, 1)

  loss = cross_entropy_loss(logits, labels, lengths, vocab_size)
  sequence_accuracy = compute_perfect_match_accuracy(
    predictions, labels, lengths)
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


def write_examples(file: TextIO, no_logged_examples: int,
                   gold_outputs: jnp.array, inferred_outputs: jnp.array,
                   attention_weights: jnp.array,
                   data_source: inp.CFQDataSource):
  #log the first examples in the batch
  gold_seq = indices_to_str(gold_outputs, data_source)
  inferred_seq = indices_to_str(inferred_outputs, data_source)
  for i in range(0, no_logged_examples):
    file.write('\nGold seq:\n {0} \nInferred seq:\n {1}\n'.format(gold_seq[i],
                  inferred_seq[i]))
    file.write('Attention weights\n')
    np.savetxt(file, attention_weights[i], fmt='%0.2f')


@functools.partial(jax.pmap, axis_name='batch', static_broadcasted_argnums=(3))
def train_step(optimizer: Any, batch: BatchType, rng: Any, vocab_size: int):
  """Train one step."""

  inputs = batch[constants.QUESTION_KEY]
  input_lengths = batch[constants.QUESTION_LEN_KEY]
  labels = batch[constants.QUERY_KEY]
  labels_no_bos = labels[:, 1:]
  queries_lengths = batch[constants.QUERY_LEN_KEY] - 1

  def loss_fn(model):
    """Compute cross-entropy loss."""
    with nn.stochastic(rng):
      logits, predictions, _ = model(encoder_inputs=inputs,
                                     decoder_inputs=labels,
                                     encoder_inputs_lengths=input_lengths,
                                     vocab_size=vocab_size)
    loss = cross_entropy_loss(logits,
                              labels_no_bos,
                              queries_lengths,
                              vocab_size)
    return loss, (logits, predictions)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, output), grad = grad_fn(optimizer.target)
  logits, predictions = output
  grad = jax.lax.pmean(grad, axis_name='batch')
  grad = clip_grads(grad, max_norm=1.0)
  optimizer = optimizer.apply_gradient(grad)
  metrics = {}
  metrics = compute_metrics(logits,
                            predictions,
                            labels_no_bos,
                            queries_lengths,
                            vocab_size)
  metrics = jax.lax.pmean(metrics, axis_name='batch')
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
  # filled with sequences of max_output_len of only the bos encoding. The length
  # is the desired output length + 2 (bos and eos tokens).
  initial_dec_inputs = jnp.tile(bos_encoding,
                                (inputs.shape[0], predicted_output_length + 2))
  with nn.stochastic(rng):
    logits, predictions, attention_weights = model(encoder_inputs=inputs,
      decoder_inputs=initial_dec_inputs,
      encoder_inputs_lengths=inputs_lengths,
      vocab_size=vocab_size,
      train=False)
  return logits, predictions, attention_weights


def evaluate_model(model: nn.Module,
                   batches: tf.data.Dataset,
                   data_source: inp.CFQDataSource,
                   predicted_output_length: int,
                   logging_file: TextIO,
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
    logits, inferred_outputs, attention_weights = infer(model, inputs, input_lengths, nn.make_rng(),
                                data_source.vocab_size, data_source.bos_idx,
                                predicted_output_length)
    metrics = compute_metrics(
        logits,
        inferred_outputs,
        gold_outputs,
        batch[constants.QUERY_LEN_KEY] - 1,
        data_source.vocab_size)
    avg_metrics = {key: avg_metrics[key] + metrics[key] for key in avg_metrics}
    if no_logged_examples is not None and no_batches == 0:
      write_examples(logging_file, no_logged_examples,
                     gold_outputs, inferred_outputs,
                     attention_weights,
                     data_source)
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

def shard(xs):
  return jax.tree_map(
      lambda x: x.reshape((jax.device_count(), -1) + x.shape[1:]), xs)

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
  if os.path.isdir(model_dir):
    shutil.rmtree(model_dir)
  os.makedirs(model_dir)
  logging_file_name = os.path.join(model_dir, 'logged_examples.txt')
  logging_file = open(logging_file_name,'w')

  with nn.stochastic(jax.random.PRNGKey(seed)):
    model = create_model(data_source.vocab_size)
    optimizer = flax.optim.Adam(learning_rate=learning_rate).create(model)
    optimizer = jax_utils.replicate(optimizer)

    if bucketing:
      train_batches = data_source.get_bucketed_batches(
          split = 'train',
          batch_size = batch_size,
          bucket_size = 8,
          drop_remainder = True,
          shuffle = True)
    else:
      train_batches = data_source.get_batches(split = 'train',
                                              batch_size = batch_size,
                                              shuffle = True,
                                              drop_remainder=True)

    dev_batches = data_source.get_batches(split = 'dev',
                                          batch_size = batch_size,
                                          shuffle = True,
                                          drop_remainder = True)

    train_metrics = []
    metrics_per_epoch = {
        TRAIN_ACCURACIES: [],
        TRAIN_LOSSES: [],
        TEST_ACCURACIES: [],
        TEST_LOSSES: []
    }
    for epoch in range(num_epochs):
      no_batches = 0
      for batch in tfds.as_numpy(train_batches):
        if batch_size % jax.device_count() > 0:
          raise ValueError('Batch size must be divisible by the number of devices')
        batch = shard(batch)
        step_key = nn.make_rng()
        # Shard the step PRNG key
        sharded_keys = common_utils.shard_prng_key(step_key)
        optimizer, metrics = train_step(optimizer, batch, sharded_keys,
                                        data_source.vocab_size)
        train_metrics.append(metrics)
        no_batches += 1
      train_metrics = common_utils.get_metrics(train_metrics)
      # Get training epoch summary for logging
      train_summary = jax.tree_map(lambda x: x.mean(), train_metrics)
      train_metrics = []
      # evaluate
      model = jax_utils.unreplicate(optimizer.target)  # Fetch from 1st device
      dev_metrics = evaluate_model(model=model,
                                   batches=dev_batches,
                                   data_source=data_source,
                                   predicted_output_length=max_out_len,
                                   logging_file = logging_file,
                                   no_logged_examples=3)
      log(epoch, train_summary, dev_metrics)
      metrics_per_epoch[TRAIN_ACCURACIES].append(train_summary[ACC_KEY])
      metrics_per_epoch[TRAIN_LOSSES].append(train_summary[LOSS_KEY])
      metrics_per_epoch[TEST_ACCURACIES].append(dev_metrics[ACC_KEY])
      metrics_per_epoch[TEST_LOSSES].append(dev_metrics[LOSS_KEY])

    plot_metrics(metrics_per_epoch, num_epochs)
    logging.info('Done training')

  logging.info('Saving model at %s', model_dir)
  checkpoints.save_checkpoint(model_dir, optimizer, num_epochs)
  logging_file.close()


  return optimizer.target


def test_model(model_dir, data_source: inp.CFQDataSource, max_out_len: int,
               seed: int, batch_size: int):
  """Evaluate model at model_dir on dev subset"""
  with nn.stochastic(jax.random.PRNGKey(seed)):
    logging_file_name = os.path.join(model_dir, 'eval_logged_examples.txt')
    logging_file = open(logging_file_name,'w')
    model = create_model(data_source.vocab_size)
    optimizer = flax.optim.Adam().create(model)
    optimizer = checkpoints.restore_checkpoint(model_dir, optimizer)
    dev_batches = data_source.get_batches(split = 'dev',
                                          batch_size=batch_size,
                                          shuffle=True)
    # evaluate
    dev_metrics = evaluate_model(model=optimizer.target,
                                 batches=dev_batches,
                                 data_source=data_source,
                                 predicted_output_length=max_out_len,
                                 logging_file = logging_file
                                 no_logged_examples=3)
    logging.info('Loss %.4f, acc %.2f', dev_metrics[LOSS_KEY],
                 dev_metrics[ACC_KEY])
    logging_file.close()
