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
import time
import datetime
import shutil
from typing import Any, Text, Dict, TextIO, List, Tuple
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
from flax import linen as nn
from flax.training import checkpoints
from flax.training import common_utils
from flax.metrics import tensorboard

import input_pipeline as inp
import input_pipeline_constants as inp_constants
import models
from grammar_info import GrammarInfo
import train_util

BatchType = Dict[Text, jnp.array]

ACC_KEY = 'accuracy'
LOSS_KEY = 'loss'
TRAIN_ACCURACIES = 'train acc'
TRAIN_LOSSES = 'train loss'
TEST_ACCURACIES = 'test acc'
TEST_LOSSES = 'test loss'

# The number of evaluation steps for which the accuracy has been decreasing.
# For example, the accuracy in the past 10 steps has been smaller than the
# accuracy before (the 10 steps avg before that).
EARLY_STOPPING_STEPS = 10

# vmap?
def indices_to_str(batch_inputs: jnp.ndarray, data_source: inp.CFQDataSource):
  """Decode a batch of one-hot encoding to strings."""
  #TODO: implement when implementing inference flow.
  return np.array(
      ['' for seq in batch_inputs])


def mask_sequences(sequence_batch: jnp.array, lengths: jnp.array):
  """Set positions beyond the length of each sequence to 0."""
  mask = (lengths[:, jnp.newaxis] > jnp.arange(sequence_batch.shape[1]))
  # Use where to zero out +/-inf if necessary.
  return jnp.where(mask, sequence_batch, 0)

def get_initial_params(rng: jax.random.PRNGKey,
                       grammar_info: GrammarInfo,
                       token_vocab_size: int):
  seq2seq = models.Seq2tree(
    grammar_info,
    token_vocab_size,
    train = False
  )
  initial_batch = [
    jnp.zeros((1, 1), jnp.uint8),
    jnp.zeros((1, 1, 1), jnp.uint8),
    jnp.ones((1,), jnp.uint8)
  ]
  initial_params = seq2seq.init(rng,
    initial_batch[0],
    initial_batch[1],
    initial_batch[2])
  return initial_params['params']


def cross_entropy_loss(rule_logits: jnp.array,
                       token_logits: jnp.array,
                       action_types: jnp.array,
                       action_values: jnp.array,
                       lengths: jnp.array,
                       rule_vocab_size: int,
                       token_vocab_size: int):
  """Returns cross-entropy loss."""
  rule_logits = jax.nn.softmax(rule_logits)
  token_logits = jax.nn.softmax(token_logits)
  action_types = jnp.expand_dims(action_types, -1)
  rule_logits = jnp.where(action_types,
                          jnp.zeros(rule_vocab_size), rule_logits)
  token_logits = jnp.where(action_types,
                           token_logits, jnp.zeros(token_vocab_size))
  labels_tokens = common_utils.onehot(action_values, token_vocab_size)
  labels_rules = common_utils.onehot(action_values, rule_vocab_size)
  # [batch_size, seq_len, no_rules] -> [batch_size, seq_len]
  scores_rules = jnp.sum(labels_rules * rule_logits, axis=-1)
  # [batch_size, seq_len, no_tokens] -> [batch_size, seq_len]
  scores_tokens = jnp.sum(labels_tokens * token_logits, axis=-1)
  scores = scores_rules + scores_tokens
  masked_logged_scores = jnp.sum(mask_sequences(jnp.log(scores), lengths),
                                 axis=-1)
  mean_losses = jnp.divide(masked_logged_scores, lengths)
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


def compute_metrics(rule_logits: jnp.array,
                    token_logits: jnp.array,
                    predictions: jnp.array,
                    action_values: jnp.array,
                    action_types: jnp.array,
                    queries_lengths: jnp.array,
                    rule_vocab_size: int,
                    token_vocab_size: int) -> Dict:
  """Computes metrics for a batch of logist & labels and returns those metrics

    The metrics computed are cross entropy loss and mean batch accuracy. The
    accuracy at sequence level needs perfect matching of the compared sequences
    Args:
      rule_logits: Rule logits [batch_size, predicted seq_len, rule_vocab_size].
      token_logits: Token logits
        [batch_size, predicted seq_len, token_vocab_size].
      predictions: Predictions [batch_size, predicted seq len].
      action_values: Gold action values, shape [batch_size, gold seq_len].
      action_types: Gold action types, shape [batch_size, gold seq_len].
      queries_lengths: lengths of gold queries (until eos) [batch_size].
      rule_vocab_size: rule vocabulary size.
      token_vocab_size: token vocabulary size.
    """
  lengths = queries_lengths
  gold_seq_len = action_values.shape[1]
  predicted_seq_len = rule_logits.shape[1]
  max_seq_len = max(gold_seq_len, predicted_seq_len)
  if gold_seq_len != max_seq_len:
    action_values = pad_along_axis(action_values, max_seq_len - gold_seq_len, 1)
    action_types = pad_along_axis(action_types, max_seq_len - gold_seq_len, 1)
  elif predicted_seq_len != max_seq_len:
    padding_size = max_seq_len - predicted_seq_len
    rule_logits = pad_along_axis(rule_logits, padding_size, 1)
    token_logits = pad_along_axis(token_logits, padding_size, 1)
    predictions = pad_along_axis(predictions, padding_size, 1)

  loss = cross_entropy_loss(rule_logits,
                            token_logits,
                            action_types,
                            action_values,
                            lengths,
                            rule_vocab_size,
                            token_vocab_size)
  sequence_accuracy = compute_perfect_match_accuracy(
    predictions, action_values, lengths)
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

def write_examples(summary_writer: tensorboard.SummaryWriter,
                   step: int,
                   no_logged_examples: int,
                   gold_batch: Dict, inferred_batch: Dict,
                   attention_weights: jnp.array,
                   data_source: inp.CFQDataSource):
  #Log the first examples in the batch.
  for i in range(0, no_logged_examples):
    # Log queries.
    gold_seq, _ = data_source.action_seq_to_query(
      gold_batch[inp_constants.ACTION_TYPES_KEY][i],
      gold_batch[inp_constants.ACTION_VALUES_KEY][i])
    inferred_seq, actions_as_strings = data_source.action_seq_to_query(
      inferred_batch[inp_constants.ACTION_TYPES_KEY][i],
      inferred_batch[inp_constants.ACTION_VALUES_KEY][i])
    logged_text = 'Gold seq:  \n {0}  \nInferred seq:  \n {1}  \n'.format(
      gold_seq, inferred_seq)
    summary_writer.text('Example {}'.format(i), logged_text, step)
    # Log attention scores.
    question = gold_batch[inp_constants.QUESTION_KEY][i]
    question = data_source.indices_to_sequence_string(question).split()
    question_len = len(question)
    act_seq_len = len(actions_as_strings)
    attention_weights_no_pad = attention_weights[i][0:question_len][0:act_seq_len]
    train_util.save_attention_img_to_tensorboard(
      summary_writer, step, question, actions_as_strings, attention_weights_no_pad)


def get_decoder_inputs(batch: BatchType):
  action_types = batch[inp_constants.ACTION_TYPES_KEY]
  action_values = batch[inp_constants.ACTION_VALUES_KEY]
  node_types = batch[inp_constants.NODE_TYPES_KEY]
  output = jnp.array([action_types, action_values, node_types])
  output = jnp.swapaxes(output, 0, 1)
  return output

# Jit the function instead of just pmapping it to make sure error propagation
# works (TODO: check to see when fix is part of a jax version).
# @functools.partial(jax.jit, static_argnums=(3, 4))
@functools.partial(jax.pmap, axis_name='batch',
                   static_broadcasted_argnums=(3, 4))
def train_step(optimizer: Any,
               batch: BatchType,
               rng: jax.random.PRNGKey,
               grammar_info: GrammarInfo,
               token_vocab_size: int):
  """Train one step."""
  step_rng = jax.random.fold_in(rng, optimizer.state.step)

  inputs = batch[inp_constants.QUESTION_KEY]
  input_lengths = batch[inp_constants.QUESTION_LEN_KEY]
  action_types = batch[inp_constants.ACTION_TYPES_KEY]
  action_values = batch[inp_constants.ACTION_VALUES_KEY]
  decoder_inputs = get_decoder_inputs(batch)
  queries_lengths = batch[inp_constants.ACTION_SEQ_LEN_KEY]

  def loss_fn(params):
    """Compute cross-entropy loss."""
    seq2tree = models.Seq2tree(
      grammar_info,
      token_vocab_size,
      train=True)
    nan_error, rule_logits, token_logits, pred_act_types, pred_act_values, _ = \
      seq2tree.apply(
        {'params': params}, 
        encoder_inputs=inputs,
        decoder_inputs=decoder_inputs,
        encoder_inputs_lengths=input_lengths,
        rngs={'dropout': step_rng})
    loss = cross_entropy_loss(rule_logits,
                              token_logits,
                              action_types,
                              action_values,
                              queries_lengths,
                              grammar_info.rule_vocab_size,
                              token_vocab_size)
    return loss, (nan_error, rule_logits, token_logits, pred_act_types, pred_act_values)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, output), grad = grad_fn(optimizer.target)
  nan_error, rule_logits, token_logits, pred_act_types, pred_act_values = output
  grad = jax.lax.pmean(grad, axis_name='batch')
  grad = clip_grads(grad, max_norm=1.0)
  optimizer = optimizer.apply_gradient(grad)
  metrics = {}
  metrics = compute_metrics(rule_logits,
                            token_logits,
                            pred_act_values,
                            action_values,
                            action_types,
                            queries_lengths,
                            grammar_info.rule_vocab_size,
                            token_vocab_size)
  metrics = jax.lax.pmean(metrics, axis_name='batch')
  return nan_error, optimizer, metrics


@jax.partial(jax.jit, static_argnums=[3, 4, 6])
def infer(params, inputs: jnp.array, inputs_lengths: jnp.array,
          grammar_info: GrammarInfo,
          token_vocab_size: int,
          bos_encoding: jnp.array,
          predicted_output_length: int):
  """Apply model on inference flow and return predictions.

    Args:
        model: the seq2seq model applied
        inputs: batch of input sequences
        vocab_size: size of vocabulary
        bos_encoding: id the BOS token
        predicted_output_length: what length should predict for the output
                                 (should be a static argnum)
    """
  #TODO: figure out initial decoder input when implementing inference flow.
  batch_size = inputs.shape[0]
  action_types = jnp.zeros((batch_size, predicted_output_length))
  action_values = jnp.zeros((batch_size, predicted_output_length),
                            dtype=jnp.uint8)
  node_types = jnp.zeros((batch_size, predicted_output_length))
  decoder_inputs = jnp.array([action_types, action_values, node_types])
  # Go from [2, batch_size, seq_len] -> [batch_size, 2, seq_len].
  decoder_inputs = jnp.swapaxes(decoder_inputs, 0, 1)
  seq2tree = models.Seq2tree(
      grammar_info,
      token_vocab_size,
      train=False)
  _, rule_logits, token_logits,\
     pred_act_types, pred_act_values, attention_weights = \
    seq2tree.apply(
      {'params': params},
      encoder_inputs=inputs,
      decoder_inputs=decoder_inputs,
      encoder_inputs_lengths=inputs_lengths)
  return rule_logits, token_logits,\
    pred_act_types, pred_act_values, attention_weights


def evaluate_model(params: Any,
                   batches: tf.data.Dataset,
                   data_source: inp.CFQDataSource,
                   predicted_output_length: int,
                   summary_writer: tensorboard.SummaryWriter,
                   step: int,
                   no_logged_examples: int = None):
  """Evaluate the model on the validation/test batches

  Args:
      model: The model parameters.
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
    inputs = batch[inp_constants.QUESTION_KEY]
    input_lengths = batch[inp_constants.QUESTION_LEN_KEY]
    gold_outputs = batch['action_values']
    gold_action_types = batch['action_types']
    queries_lengths = batch['action_seq_len'] - 1
    rule_logits, token_logits,\
      pred_act_types, pred_act_values, attention_weights = \
        infer(params,
              inputs, input_lengths,
              data_source.grammar_info,
              data_source.tokens_vocab_size,
              data_source.bos_idx,
              predicted_output_length)
    metrics = compute_metrics(
      rule_logits,
      token_logits,
      pred_act_values,
      gold_outputs,
      gold_action_types,
      queries_lengths,
      data_source.grammar_info.rule_vocab_size,
      data_source.tokens_vocab_size)
    avg_metrics = {key: avg_metrics[key] + metrics[key] for key in avg_metrics}
    predicted_batch = {
      inp_constants.ACTION_TYPES_KEY: pred_act_types,
      inp_constants.ACTION_VALUES_KEY: pred_act_values
    }
    if no_logged_examples is not None and no_batches == 0:
      write_examples(summary_writer,
                     step + 1,
                     no_logged_examples,
                     batch, predicted_batch,
                     attention_weights,
                     data_source)
    no_batches += 1
  avg_metrics = {key: avg_metrics[key] / no_batches for key in avg_metrics}
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
                data_source: inp.CFQDataSource = None,
                batch_size: int = None,
                bucketing: bool = False,
                model_dir=None,
                eval_freq: float = None,
                detail_log_freq: float = None,
                early_stopping = False):
  """ Train model for num_train_steps.

  Do the training on data_source.train_dataset and evaluate on
  data_source.dev_dataset every few steps and log the results.
  """
  start_time = time.time()
  if os.path.isdir(model_dir):
    # If attemptying to save in a directory where the model was saved before,
    # first remove the directory with its contents. This is done mostly
    # because the checkpoint saving will through an error when saving in the
    # same place twice.
    shutil.rmtree(model_dir)
  os.makedirs(model_dir)
  if jax.host_id() == 0:
    train_summary_writer = tensorboard.SummaryWriter(
        os.path.join(model_dir, 'train'))
    eval_summary_writer = tensorboard.SummaryWriter(
        os.path.join(model_dir, 'eval'))

  
  rng = jax.random.PRNGKey(seed)
  rng, init_rng = jax.random.split(rng)
  initial_params = get_initial_params(
    init_rng,
    data_source.grammar_info,
    data_source.tokens_vocab_size)
  optimizer = flax.optim.Adam(learning_rate=learning_rate).create(initial_params)
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
  last_dev_accuracies = []
  best_acc = 0
  for step, batch in zip(range(num_train_steps), train_iter):
    if batch_size % jax.device_count() > 0:
      raise ValueError('Batch size must be divisible by the number of devices')
    batch = common_utils.shard(jax.tree_map(lambda x: x._numpy(), batch))
    rng, step_key = jax.random.split(rng)
    # Shard the step PRNG key
    sharded_keys = common_utils.shard_prng_key(step_key)
    nan_error, optimizer, metrics = train_step(optimizer, batch, sharded_keys,
                                    data_source.grammar_info,
                                    data_source.tokens_vocab_size)
    
    train_metrics.append(metrics)
    if (step + 1) % eval_freq == 0:
      train_metrics = common_utils.get_metrics(train_metrics)
      train_summary = jax.tree_map(lambda x: x.mean(), train_metrics)
      train_metrics = []
      # evaluate
      params = jax_utils.unreplicate(optimizer.target)  # Fetch from 1st device
      no_logged_examples=None
      if (step + 1) % detail_log_freq ==0:
        no_logged_examples=3
      dev_metrics = evaluate_model(params=params,
                                  batches=dev_batches,
                                  data_source=data_source,
                                  predicted_output_length=max_out_len,
                                  summary_writer=eval_summary_writer,
                                  step=step,
                                  no_logged_examples=no_logged_examples)
      log(step, train_summary, dev_metrics)
      save_to_tensorboard(train_summary_writer, train_summary, step + 1)
      save_to_tensorboard(eval_summary_writer, dev_metrics, step + 1)

      # Save best model.
      dev_acc = dev_metrics[ACC_KEY]
      if best_acc < dev_acc:
        best_acc = dev_acc
        checkpoints.save_checkpoint(model_dir, optimizer, num_train_steps, keep=1)

      
      if early_stopping:
        # Early stopping (stop when model dev acc avg decreases).
        if len(last_dev_accuracies) == 2 * EARLY_STOPPING_STEPS:
          # remove the oldest stored accuracy.
          del last_dev_accuracies[0]
        # Store the most recent accuracy.
        last_dev_accuracies.append(dev_metrics[ACC_KEY])
        if len(last_dev_accuracies) == 2 * EARLY_STOPPING_STEPS:
          old_avg = sum(last_dev_accuracies[0:EARLY_STOPPING_STEPS])/EARLY_STOPPING_STEPS
          new_avg = sum(last_dev_accuracies[EARLY_STOPPING_STEPS:])/EARLY_STOPPING_STEPS
          if new_avg < old_avg:
            # Stop training
            break

  end_time = time.time()
  time_passed = datetime.timedelta(seconds=end_time - start_time) 
  logging.info('Done training. Training took {}'.format(time_passed))

  return optimizer.target


#TODO: migrate to linen.
def test_model(model_dir, data_source: inp.CFQDataSource, max_out_len: int,
               seed: int, batch_size: int):
  """Evaluate model at model_dir on dev subset"""
  with nn.stochastic(jax.random.PRNGKey(seed)):
    logging_file_name = os.path.join(model_dir, 'eval_logged_examples.txt')
    model = create_model(data_source.tokens_vocab_size)
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
                                 no_logged_examples=3)
    logging.info('Loss %.4f, acc %.2f', dev_metrics[LOSS_KEY],
                 dev_metrics[ACC_KEY])
