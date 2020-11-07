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

import functools
import jax
import jax.numpy as jnp
from jax.experimental.optimizers import clip_grads

import flax
from flax import jax_utils
from flax import nn
from flax.training import checkpoints
from flax.training import common_utils
from flax.metrics import tensorboard

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


def mask_sequences(sequence_batch: jnp.array, lengths: jnp.array):
  """Set positions beyond the length of each sequence to 0."""
  mask = (lengths[:, jnp.newaxis] > jnp.arange(sequence_batch.shape[1]))
  return sequence_batch * mask


def create_model(token_vocab_size: int,
                 rule_vocab_size: int,
                 node_vocab_size: int) -> nn.Module:
  """Creates a seq2seq model."""
  seq2tree = models.Seq2tree.partial(token_vocab_size=token_vocab_size,
                                     rule_vocab_size=rule_vocab_size,
                                     node_vocab_size=node_vocab_size)
  _, initial_params = seq2tree.init_by_shape(
    nn.make_rng(),
    [((1, 1), jnp.uint8),
    # decoder inputs [batch_size, 4, seq_len]
    ((1, 4, 1), jnp.uint8),
    ((1,), jnp.uint8)])
  model = nn.Model(models.Seq2tree, initial_params)
  return model


def cross_entropy_loss(rules_logits: jnp.array,
                       tokens_logits: jnp.array,
                       labels: jnp.array,
                       lengths: jnp.array,
                       rule_vocab_size: int,
                       token_vocab_size: int):
  """Returns cross-entropy loss."""
  labels_rules = common_utils.onehot(labels, rule_vocab_size)
  labels_tokens = common_utils.onehot(labels, token_vocab_size)
  scores = labels * rules_logits + labels * tokens_logits
  log_scores = jnp.log(scores)
  log_sum = jnp.sum(log_scores, axis=-1)
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

# TODO: on the test flow will need to compare action_types as well
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


def compute_metrics(rules_logits: jnp.array,
                    tokens_logits: jnp.array,
                    predictions: jnp.array,
                    labels: jnp.array,
                    queries_lengths: jnp.array,
                    rule_vocab_size: int,
                    token_vocab_size: int) -> Dict:
  """Computes metrics for a batch of logist & labels and returns those metrics

    The metrics computed are cross entropy loss and mean batch accuracy. The
    accuracy at sequence level needs perfect matching of the compared sequences
    Args:
      logits: logits (train time) or ohe predictions (test time)
              [batch_size, logits seq_len, vocab_size]
      predictions: predictions [batch_size, predicted seq len]
      labels: ohe gold labels, shape [batch_size, labels seq_len]
      queries_lengths: lengths of gold queries (until eos) [batch_size]
      rule_vocab_size: rule vocabulary size (no of rules).
      token_vocab_size: token vocab size (no of tokens).
    """
  lengths = queries_lengths
  labels_seq_len = labels.shape[1]
  logits_seq_len = rules_logits.shape[1]
  max_seq_len = max(labels_seq_len, logits_seq_len)
  if labels_seq_len != max_seq_len:
    labels = pad_along_axis(labels, max_seq_len - labels_seq_len, 1)
  elif logits_seq_len != max_seq_len:
    padding_size = max_seq_len - logits_seq_len
    rules_logits = pad_along_axis(rules_logits, padding_size, 1)
    tokens_logits = pad_along_axis(tokens_logits, padding_size, 1)
    predictions = pad_along_axis(predictions, padding_size, 1)

  loss = cross_entropy_loss(rules_logits, tokens_logits,
                            labels, lengths,
                            rule_vocab_size,
                            token_vocab_size)
  sequence_accuracy = compute_perfect_match_accuracy(
    predictions, labels, lengths)
  accuracy = jnp.mean(sequence_accuracy)
  metrics = {
      'loss': loss,
      'accuracy': accuracy,
  }
  return metrics


def log(step: int, train_metrics: Dict, dev_metrics: Dict):
  """Logs performance at a certain step..

    Args:
      step: The step number.
      train_metrics: A dict with the train metrics for this step.
      dev_metrics: A dict with the validation metrics for this step.
    """
  logging.info(
      'Step %02d train loss %.4f dev loss %.4f train acc %.2f dev acc %.2f',
      step + 1, train_metrics[LOSS_KEY], dev_metrics[LOSS_KEY],
      train_metrics[ACC_KEY], dev_metrics[ACC_KEY])


def write_examples(file: TextIO, no_logged_examples: int,
                   gold_outputs: jnp.array, inferred_outputs: jnp.array,
                   attention_weights: jnp.array,
                   data_source: inp.Seq2TreeCfqDataSource):
  #log the first examples in the batch
  file.write('Dummy log, TODO\n')


def get_decoder_inputs(batch: BatchType):
  action_types = batch[constants.ACTION_TYPES_KEY]
  action_values = batch[constants.ACTION_VALUES_KEY]
  node_types = batch[constants.NODE_TYPES_KEY]
  parent_steps = batch[constants.PARENT_STEPS_KEY]
  output = jnp.array([action_types, action_values, node_types, parent_steps])
  output = jnp.swapaxes(output, 0, 1)
  return output

@functools.partial(jax.pmap, axis_name='batch',
                   static_broadcasted_argnums=(3, 4, 5))
def train_step(optimizer: Any, batch: BatchType, rng: Any,
               token_vocab_size: int,
               rule_vocab_size: int,
               node_vocab_size: int):
  """Train one step."""

  inputs = batch[constants.QUESTION_KEY]
  input_lengths = batch[constants.QUESTION_LEN_KEY]
  labels = batch[constants.ACTION_VALUES_KEY]
  decoder_inputs = get_decoder_inputs(batch)
  queries_lengths = batch[constants.ACTION_SEQ_LEN_KEY]

  def loss_fn(model):
    """Compute cross-entropy loss."""
    with nn.stochastic(rng):
      rules_logits,\
      tokens_logits,\
      predictions, _ = model(encoder_inputs=inputs,
                             decoder_inputs=decoder_inputs,
                             encoder_inputs_lengths=input_lengths,
                             token_vocab_size=token_vocab_size,
                             rule_vocab_size=rule_vocab_size,
                             node_vocab_size=node_vocab_size)
    loss = cross_entropy_loss(rules_logits,
                              tokens_logits,
                              labels,
                              queries_lengths,
                              rule_vocab_size,
                              token_vocab_size)
    return loss, (rules_logits, tokens_logits, predictions)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, output), grad = grad_fn(optimizer.target)
  rules_logits, tokens_logits, predictions = output
  grad = jax.lax.pmean(grad, axis_name='batch')
  grad = clip_grads(grad, max_norm=1.0)
  optimizer = optimizer.apply_gradient(grad)
  metrics = {}
  metrics = compute_metrics(rules_logits,
                            tokens_logits,
                            predictions,
                            labels,
                            queries_lengths,
                            rule_vocab_size,
                            token_vocab_size)
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
                   data_source: inp.Seq2TreeCfqDataSource,
                   predicted_output_length: int,
                   logging_file_name: str,
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
  avg_metrics = {ACC_KEY: 0, LOSS_KEY: 0}
  # TODO: evaluation flow
  return avg_metrics


def shard(xs):
  return jax.tree_map(
      lambda x: x.reshape((jax.device_count(), -1) + x.shape[1:]), xs)


def save_to_tensorboard(summary_writer: tensorboard.SummaryWriter,
                        dict: Dict, step: int):
  if jax.host_id() == 0:
    for key, val in dict.items():
      summary_writer.scalar(key, val, step)
    summary_writer.flush()


def train_model(learning_rate: float = None,
                num_train_steps: int = None,
                max_out_len: int = None,
                seed: int = None,
                data_source: inp.Seq2TreeCfqDataSource = None,
                batch_size: int = None,
                bucketing: bool = False,
                model_dir=None,
                eval_freq: float = None):
  """ Train model for num_train_steps.

    Do the training on data_source.train_dataset and evaluate on
    data_source.dev_dataset every few steps and log the results.
    """
  if os.path.isdir(model_dir):
    # If attemptying to save in a directory where the model was saved before,
    # first remove the directory with its contents. This is done mostly
    # because the checkpoint saving will through an error when saving in the
    # same place twice.
    shutil.rmtree(model_dir)
  os.makedirs(model_dir)
  logging_file_name = os.path.join(model_dir, 'logged_examples.txt')
  if jax.host_id() == 0:
    train_summary_writer = tensorboard.SummaryWriter(
        os.path.join(model_dir, 'train'))
    eval_summary_writer = tensorboard.SummaryWriter(
        os.path.join(model_dir, 'eval'))

  with nn.stochastic(jax.random.PRNGKey(seed)):
    model = create_model(data_source.tokens_vocab_size,
                         data_source.rule_vocab_size,
                         data_source.node_vocab_size)
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

    train_iter = iter(train_batches)
    train_metrics = []
    for step, batch in zip(range(num_train_steps), train_iter):
      if batch_size % jax.device_count() > 0:
        raise ValueError('Batch size must be divisible by the number of devices')
      batch = common_utils.shard(jax.tree_map(lambda x: x._numpy(), batch))
      step_key = nn.make_rng()
      # Shard the step PRNG key
      sharded_keys = common_utils.shard_prng_key(step_key)
      optimizer, metrics = train_step(optimizer, batch, sharded_keys,
                                      data_source.tokens_vocab_size,
                                      data_source.rule_vocab_size,
                                      data_source.node_vocab_size)
      train_metrics.append(metrics)
      if (step + 1) % eval_freq == 0:
        train_metrics = common_utils.get_metrics(train_metrics)
        train_summary = jax.tree_map(lambda x: x.mean(), train_metrics)
        train_metrics = []
        # evaluate
        model = jax_utils.unreplicate(optimizer.target)  # Fetch from 1st device
        dev_metrics = evaluate_model(model=model,
                                    batches=dev_batches,
                                    data_source=data_source,
                                    predicted_output_length=max_out_len,
                                    logging_file_name = logging_file_name,
                                    no_logged_examples=3)
        log(step, train_summary, dev_metrics)
        save_to_tensorboard(train_summary_writer, train_summary, step)
        save_to_tensorboard(eval_summary_writer, dev_metrics, step)

    logging.info('Done training')

  logging.info('Saving model at %s', model_dir)
  checkpoints.save_checkpoint(model_dir, optimizer, num_train_steps)


  return optimizer.target


def test_model(model_dir, data_source: inp.Seq2TreeCfqDataSource,
               max_out_len: int,
               seed: int, batch_size: int):
  """Evaluate model at model_dir on dev subset"""
  print('TODO')
