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
from jax import random
from jax.experimental.optimizers import clip_grads

import flax
from flax import jax_utils
from flax import linen as nn
from flax.training import checkpoints
from flax.training import common_utils
from flax.metrics import tensorboard

import cfq.input_pipeline as inp
import cfq.input_pipeline_constants as inp_constants
from cfq.models import Seq2seq
import cfq.train_util as train_util

BatchType = Dict[Text, jnp.array]

ACC_KEY = 'accuracy'
LOSS_KEY = 'loss'
TRAIN_ACCURACIES = 'train acc'
TRAIN_LOSSES = 'train loss'
TEST_ACCURACIES = 'test acc'
TEST_LOSSES = 'test loss'

EARLY_STOPPING_STEPS = 10

def mask_sequences(sequence_batch: jnp.array, lengths: jnp.array):
  """Set positions beyond the length of each sequence to 0."""
  mask = (lengths[:, jnp.newaxis] > jnp.arange(sequence_batch.shape[1]))
  return sequence_batch * mask

def get_initial_params(rng: jax.random.PRNGKey, vocab_size: int):
  seq2seq = Seq2seq(vocab_size=vocab_size)
  initial_batch = [
          jnp.array((1, 1), jnp.uint8),
          #decoder_inputs
          # need to pass 2 for decoder length
          # as the first token is cut off
          jnp.array((1, 2), jnp.uint8),
          jnp.array((1,), jnp.uint8)
      ]
  initial_batch = [
      jnp.zeros((1, 1), jnp.uint8),
      jnp.zeros((1, 2), jnp.uint8),
      # Avoid nans in the attention softmax by masking everything out.
      jnp.ones((1,), jnp.uint8)
    ]
  initial_params = seq2seq.init(rng,
      initial_batch[0],
      initial_batch[1],
      initial_batch[2],
      False)
  return initial_params['params']


def cross_entropy_loss(logits: jnp.array, labels: jnp.array,
                       lengths: jnp.array, vocab_size: int):
  """Returns cross-entropy loss."""
  labels = common_utils.onehot(labels, vocab_size)
  log_soft = nn.log_softmax(logits)
  log_sum = jnp.sum(log_soft * labels, axis=-1)
  masked_log_sums = jnp.sum(mask_sequences(log_sum, lengths), axis=-1)
  mean_loss = jnp.mean(masked_log_sums)
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
                   no_logged_examples,
                   inputs: jnp.array,
                   gold_outputs: jnp.array, inferred_outputs: jnp.array,
                   attention_weights: jnp.array,
                   data_source: inp.CFQDataSource):
  #Log the first examples in the batch.
  for i in range(0, no_logged_examples):
    # Log queries.
    gold_seq = data_source.indices_to_sequence_string(
      gold_outputs[i]).split()
    inferred_seq = data_source.indices_to_sequence_string(
      inferred_outputs[i]).split()
    logged_text = 'Gold seq:  \n {0}  \nInferred seq:  \n {1}  \n'.format(
      gold_seq, inferred_seq)
    summary_writer.text('Example {}'.format(i), logged_text, step)
    # Log attention scores.
    question = data_source.indices_to_sequence_string(inputs[i]).split()
    question_len = len(question)
    query_len = len(inferred_seq)
    attention_weights_no_pad = attention_weights[i][0:question_len][0:query_len]
    train_util.save_attention_img_to_tensorboard(
      summary_writer, step, question, inferred_seq, attention_weights_no_pad)


@functools.partial(jax.pmap, axis_name='batch', static_broadcasted_argnums=(3))
def train_step(optimizer: Any,
    batch: BatchType,
    rng: jax.random.PRNGKey,
    vocab_size: int):
  """Train one step."""
  step_rng = jax.random.fold_in(rng, optimizer.state.step)

  inputs = batch[inp_constants.QUESTION_KEY]
  input_lengths = batch[inp_constants.QUESTION_LEN_KEY]
  labels = batch[inp_constants.QUERY_KEY]
  labels_no_bos = labels[:, 1:]
  queries_lengths = batch[inp_constants.QUERY_LEN_KEY] - 1

  def loss_fn(params):
    """Compute cross-entropy loss."""
    seq2seq = Seq2seq(vocab_size=vocab_size)
    logits, predictions, _ = seq2seq.apply(
      {'params': params}, 
      encoder_inputs=inputs,
      decoder_inputs=labels,
      encoder_inputs_lengths=input_lengths,
      train=True,
      rngs={'dropout': step_rng})
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

@jax.partial(jax.jit, static_argnums=[3, 5])
def infer(params: Dict, inputs: jnp.array, inputs_lengths: jnp.array,
          vocab_size: int, bos_encoding: jnp.array,
          predicted_output_length: int):
  """Apply model on inference flow and return predictions.

    Args:
        model: the seq2seq params.
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
  seq2seq = Seq2seq(vocab_size=vocab_size)
  logits, predictions, attention_weights = seq2seq.apply(
    {'params': params},
    encoder_inputs=inputs,
    decoder_inputs=initial_dec_inputs,
    encoder_inputs_lengths=inputs_lengths,
    train=False)
  return logits, predictions, attention_weights


def evaluate_model(model: nn.Module,
                   batches: tf.data.Dataset,
                   data_source: inp.CFQDataSource,
                   predicted_output_length: int,
                   summary_writer: tensorboard.SummaryWriter,
                   step: int,
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
    inputs = batch[inp_constants.QUESTION_KEY]
    input_lengths = batch[inp_constants.QUESTION_LEN_KEY]
    gold_outputs = batch[inp_constants.QUERY_KEY][:, 1:]
    #TODO: do not use rngs on eval.
    logits, inferred_outputs, attention_weights = infer(model,
                                inputs, input_lengths,
                                data_source.tokens_vocab_size,
                                data_source.bos_idx,
                                predicted_output_length)
    metrics = compute_metrics(
        logits,
        inferred_outputs,
        gold_outputs,
        batch[inp_constants.QUERY_LEN_KEY] - 1,
        data_source.tokens_vocab_size)
    avg_metrics = {key: avg_metrics[key] + metrics[key] for key in avg_metrics}
    if no_logged_examples is not None and no_batches == 0:
      write_examples(summary_writer,
                     step + 1,
                     no_logged_examples,
                     inputs,
                     gold_outputs, inferred_outputs,
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
                early_stopping: bool = False):
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
  if jax.host_id() == 0:
    train_summary_writer = tensorboard.SummaryWriter(
        os.path.join(model_dir, 'train'))
    eval_summary_writer = tensorboard.SummaryWriter(
        os.path.join(model_dir, 'eval'))
  
  rng = jax.random.PRNGKey(seed)
  rng, init_rng = jax.random.split(rng)
  initial_params = get_initial_params(init_rng, data_source.tokens_vocab_size)
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
  best_acc = 0
  for step, batch in zip(range(num_train_steps), train_iter):
    if batch_size % jax.device_count() > 0:
      raise ValueError('Batch size must be divisible by the number of devices')
    batch = common_utils.shard(jax.tree_map(lambda x: x._numpy(), batch))
    rng, step_key = jax.random.split(rng)
    # Shard the step PRNG key
    sharded_keys = common_utils.shard_prng_key(step_key)
    optimizer, metrics = train_step(optimizer, batch, sharded_keys,
                                    data_source.tokens_vocab_size)
    train_metrics.append(metrics)
    if (step + 1) % eval_freq == 0:
      train_metrics = common_utils.get_metrics(train_metrics)
      train_summary = jax.tree_map(lambda x: x.mean(), train_metrics)
      train_metrics = []
      # evaluate
      model = jax_utils.unreplicate(optimizer.target)  # Fetch from 1st device
      no_logged_examples=None
      if (step + 1) % detail_log_freq ==0:
        no_logged_examples=3
      dev_metrics = evaluate_model(model=model,
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
        checkpoints.save_checkpoint(model_dir, optimizer, step + 1, keep=1)

      
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

  logging.info('Done training')


  return optimizer.target

def test_model(model_dir, data_source: inp.CFQDataSource, max_out_len: int,
               seed: int, batch_size: int):
  """Evaluate model at model_dir on dev subset"""
  rng = jax.random.PRNGKey(seed)
  rng, init_rng = jax.random.split(rng)
  initial_params = get_initial_params(
    init_rng,
    data_source.tokens_vocab_size)
  optimizer = flax.optim.Adam().create(initial_params)
  optimizer = checkpoints.restore_checkpoint(model_dir, optimizer)
  dev_batches = data_source.get_batches(split = 'dev',
                                        batch_size=batch_size,
                                        shuffle=True)
  tensorboard_dir = os.path.join(model_dir, 'test')
  if os.path.isdir(tensorboard_dir):
    # Remove old tensorboard logs.
    shutil.rmtree(tensorboard_dir)
  summary_writer = tensorboard.SummaryWriter(tensorboard_dir)
  # evaluate
  params = jax_utils.unreplicate(optimizer.target)  # Fetch from 1st device
  dev_metrics = evaluate_model(model=params,
                               batches=dev_batches,
                               data_source=data_source,
                               predicted_output_length=max_out_len,
                               summary_writer=summary_writer,
                               step=1,
                               no_logged_examples=10)
  loss = dev_metrics[LOSS_KEY]
  acc = dev_metrics[ACC_KEY]
  summary_writer.text('Loss',  '{:.4f}'.format(loss), step=1)
  summary_writer.text('Acc',  '{:.4f}'.format(acc), step=1)
  logging.info('Loss %.4f, acc %.4f', loss, acc)