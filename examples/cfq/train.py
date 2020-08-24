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
"""seq2seq addition example."""
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'  # or pass as env var


import random
from absl import app
from absl import flags
from absl import logging
from typing import Any, Text

import tensorflow_datasets as tfds
import tensorflow.compat.v2 as tf
tf.config.experimental.set_visible_devices([], "GPU")

import flax
from flax import jax_utils
from flax import nn
from flax import optim

import jax
import jax.numpy as jnp
from jax.config import config
config.enable_omnistaging()

import numpy as np

import input_pipeline
import constants

FLAGS = flags.FLAGS

flags.DEFINE_float(
    'learning_rate',
    default=0.003,
    help=('The learning rate for the Adam optimizer.'))

flags.DEFINE_integer(
    'batch_size', default=constants.DEFAULT_BATCH_SIZE, help=('Batch size for training.'))

flags.DEFINE_integer(
    'num_epochs', default=constants.NUM_EPOCHS, help=('Number of epochs.'))

flags.DEFINE_integer(
    'max_question_length',
    default=30, #28 max in the train dataset
    help=('Maximum length of a question.'))

flags.DEFINE_integer(
    'max_query_length',
    default=100,#92 max in the train dataset
    help=('Maximum length of a query.'))

flags.DEFINE_integer(
    'seed', default=0,
    help=('Random seed for network initialization.'))

def get_max_input_len():
  """Returns the max length of an input sequence."""
  return FLAGS.max_question_length 


def get_max_output_len():
  """Returns the max length of an output sequence."""
  return FLAGS.max_query_length 


def onehot(sequence, vocab_size):
  """One-hot encode a single sequence of integers."""
  return jnp.array(
      sequence[:, np.newaxis] == jnp.arange(vocab_size), dtype=jnp.float32)


def onehot_batch(batch, vocab_size: int):
  questions = batch[constants.QUESTION_KEY]
  queries = batch[constants.QUERY_KEY]
  questions_ohe = np.array([onehot(q, vocab_size) for q in questions])
  queries_ohe = np.array([onehot(q, vocab_size) for q in queries])
  return {
    constants.QUESTION_KEY: questions_ohe, 
    constants.QUERY_KEY: queries_ohe, 
    constants.QUESTION_LEN_KEY: batch[constants.QUESTION_LEN_KEY],
    constants.QUERY_LEN_KEY: batch[constants.QUERY_LEN_KEY]}


def decode_ohe_seq(ohe_seq, data_source):
  indices = ohe_seq.argmax(axis=-1)
  tokens = [data_source.i2w[i].decode() for i in indices]
  str_seq = ' '.join(tokens)
  return str_seq


def decode_onehot(batch_inputs, data_source):
  """Decode a batch of one-hot encoding to strings."""
  return np.array([decode_ohe_seq(seq, data_source) for seq in batch_inputs])


def mask_sequences(sequence_batch, lengths):
  """Set positions beyond the length of each sequence to 0."""
  return sequence_batch * (
      lengths[:, np.newaxis] > np.arange(sequence_batch.shape[1]))


class Encoder(nn.Module):
  """LSTM encoder, returning state after EOS is input."""

  def apply(self, inputs, eos_id=1, hidden_size=constants.LSTM_HIDDEN_SIZE):
    # inputs.shape = (batch_size, seq_length, vocab_size).
    batch_size = inputs.shape[0]

    lstm_cell = nn.LSTMCell.shared(name='lstm')
    init_lstm_state = nn.LSTMCell.initialize_carry(
        nn.make_rng(),
        (batch_size,),
        hidden_size)

    def encode_step_fn(carry, x):
      lstm_state, is_eos = carry
      new_lstm_state, y = lstm_cell(lstm_state, x)
      # Pass forward the previous state if EOS has already been reached.
      def select_carried_state(new_state, old_state):
        return jnp.where(is_eos[:, np.newaxis], old_state, new_state)
      # LSTM state is a tuple (c, h).
      carried_lstm_state = tuple(
          select_carried_state(*s) for s in zip(new_lstm_state, lstm_state))
      # Update `is_eos`.
      is_eos = jnp.logical_or(is_eos, x[:, eos_id])
      return (carried_lstm_state, is_eos), y

    init_carry = (init_lstm_state, jnp.zeros(batch_size, dtype=np.bool))
    if self.is_initializing():
      # initialize parameters before scan
      encode_step_fn(init_carry, inputs[:, 0])

    (final_state, _), _ = jax_utils.scan_in_dim(
        encode_step_fn,
        init=init_carry,
        xs=inputs,
        axis=1)
    return final_state


class Decoder(nn.Module):
  """LSTM decoder."""

  def apply(self, init_state, inputs, teacher_force=False):
    # inputs.shape = (batch_size, seq_length, vocab_size).
    vocab_size = inputs.shape[2]
    lstm_cell = nn.LSTMCell.shared(name='lstm')
    projection = nn.Dense.shared(features=vocab_size, name='projection')

    def decode_step_fn(carry, x):
      rng, lstm_state, last_prediction = carry
      carry_rng, categorical_rng = jax.random.split(rng, 2)
      if not teacher_force:
        x = last_prediction
      lstm_state, y = lstm_cell(lstm_state, x)
      logits = projection(y)
      predicted_tokens = jax.random.categorical(categorical_rng, logits)
      prediction = onehot(predicted_tokens, vocab_size)
      return (carry_rng, lstm_state, prediction), (logits, prediction)
    init_carry = (nn.make_rng(), init_state, inputs[:, 0])

    if self.is_initializing():
      # initialize parameters before scan
      decode_step_fn(init_carry, inputs[:, 0])

    _, (logits, predictions) = jax_utils.scan_in_dim(
        decode_step_fn,
        init=init_carry,  # rng, lstm_state, last_pred
        xs=inputs,
        axis=1)
    return logits, predictions


class Seq2seq(nn.Module):
  """Sequence-to-sequence class using encoder/decoder architecture."""

  def _create_modules(self, eos_id, hidden_size):
    encoder = Encoder.partial(
        eos_id=eos_id, hidden_size=hidden_size).shared(name='encoder')
    decoder = Decoder.shared(name='decoder')
    return encoder, decoder

  def apply(self,
            encoder_inputs,
            decoder_inputs,
            teacher_force=True,
            eos_id=1,
            hidden_size=constants.LSTM_HIDDEN_SIZE):
    """Run the seq2seq model.

    Args:
      rng_key: key for seeding the random numbers.
      encoder_inputs: padded batch of input sequences to encode, shaped
        `[batch_size, max(encoder_input_lengths), vocab_size]`.
      decoder_inputs: padded batch of expected decoded sequences for teacher
        forcing, shaped `[batch_size, max(decoder_inputs_length), vocab_size]`.
        When sampling (i.e., `teacher_force = False`), the initial time step is
        forced into the model and samples are used for the following inputs. The
        second dimension of this tensor determines how many steps will be
        decoded, regardless of the value of `teacher_force`.
      teacher_force: bool, whether to use `decoder_inputs` as input to the
        decoder at every step. If False, only the first input is used, followed
        by samples taken from the previous output logits.
      eos_id: int, the token signaling when the end of a sequence is reached.
      hidden_size: int, the number of hidden dimensions in the encoder and
        decoder LSTMs.
    Returns:
      Array of decoded logits.
    """
    encoder, decoder = self._create_modules(eos_id, hidden_size)

    # Encode inputs
    init_decoder_state = encoder(encoder_inputs)
    # Decode outputs.
    logits, predictions = decoder(
        init_decoder_state,
        # why remove the last elem??
        decoder_inputs[:, :-1],
        teacher_force=teacher_force)

    return logits, predictions


def create_model(vocab_size: int, eos_id):
  """Creates a seq2seq model."""
  _, initial_params = Seq2seq.partial(eos_id=eos_id).init_by_shape(
      nn.make_rng(),
      [((1, get_max_input_len(), vocab_size), jnp.float32),
       ((1, get_max_output_len(), vocab_size), jnp.float32)])
  model = nn.Model(Seq2seq, initial_params)
  return model

def cross_entropy_loss(logits, labels, lengths):
  """Returns cross-entropy loss."""
  log_soft = nn.log_softmax(logits)
  xe = jnp.sum(log_soft * labels, axis=-1)
  masked_xe = jnp.mean(mask_sequences(xe, lengths))
  return -masked_xe

def compute_metrics(logits, labels, queries_lengths):
  """Computes metrics and returns them."""
  lengths = queries_lengths
  loss = cross_entropy_loss(logits, labels, lengths)
  # Computes sequence accuracy, which is the same as the accuracy during
  # inference, since teacher forcing is irrelevant when all output are correct.
  token_accuracy = jnp.argmax(logits, -1) == jnp.argmax(labels, -1)
  sequence_accuracy = (
      jnp.sum(mask_sequences(token_accuracy, lengths), axis=-1) == lengths
  )
  accuracy = jnp.mean(sequence_accuracy)
  metrics = {
      'loss': loss,
      'accuracy': accuracy,
  }
  return metrics

def log(epoch, train_metrics, valid_metrics):
  """Logs performance for an epoch.
  Args:
    epoch: The epoch number.
    train_metrics: A dict with the train metrics for this epoch.
    valid_metrics: A dict with the validation metrics for this epoch.
  """
  logging.info('Epoch %02d train loss %.4f valid loss %.4f acc %.2f', epoch + 1,
               train_metrics[constants.LOSS_KEY], valid_metrics[constants.LOSS_KEY], valid_metrics[constants.ACC_KEY])

@jax.jit
def train_step(optimizer, batch, rng, eos_id):
  """Train one step."""

  inputs = batch[constants.QUESTION_KEY]
  labels = batch[constants.QUERY_KEY]
  labels_no_bos = labels[:,1:]
  queries_lengths = batch[constants.QUERY_LEN_KEY]

  def loss_fn(model):
    """Compute cross-entropy loss."""
    with nn.stochastic(rng):
      logits, _ = model(inputs, labels, eos_id=eos_id)
    loss = cross_entropy_loss(logits, labels_no_bos, queries_lengths)
    return loss, logits
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, logits), grad = grad_fn(optimizer.target)
  optimizer = optimizer.apply_gradient(grad)
  metrics = compute_metrics(logits, labels_no_bos, queries_lengths)
  return optimizer, metrics


@jax.jit
def infer(model, inputs, rng, bos_encoding):
  """Apply model on inference flow and return predictions."""
  # This simply creates a batch (batch size = inputs.shape[0]) filled with sequences of max_output_len of only the bos encoding
  init_decoder_inputs = jnp.tile(bos_encoding,
                                 (inputs.shape[0], get_max_output_len(), 1))
  with nn.stochastic(rng):
    _, predictions = model(inputs, init_decoder_inputs, teacher_force=False)
  return predictions


def evaluate_model(model, batches, data_source, bos_encoding, logging_step=None):
  # Evaluate the model on the validation/test batches
  no_batches = 0
  avg_metrics = {constants.ACC_KEY: 0, constants.LOSS_KEY: 0}
  for batch in tfds.as_numpy(batches):
    batch_ohe = onehot_batch(batch, data_source.vocab_size)
    inputs = batch_ohe[constants.QUESTION_KEY]
    gold_outputs = batch_ohe[constants.QUERY_KEY][:, 1:]
    inferred_outputs = infer(model, inputs, nn.make_rng(), bos_encoding)
    metrics = compute_metrics(inferred_outputs, gold_outputs, batch_ohe[constants.QUERY_LEN_KEY])
    avg_metrics = {key : avg_metrics[key] + metrics[key] for key in avg_metrics}
    no_batches += 1
    if logging_step is not None and no_batches % logging_step == 0:
      #log the first example in the batch 
      gold_seq = decode_onehot(gold_outputs, data_source)
      inferred_seq = decode_onehot(inferred_outputs, data_source)
      logging.info('Gold seq: %s\nInferred seq: %s\n',gold_seq[0], inferred_seq[0])
  avg_metrics = {key : avg_metrics[key] / no_batches for key in avg_metrics}
  return avg_metrics

def train_model(learning_rate: float = None,
                num_epochs: int = None,
                seed: int = None,
                data_source: Any = None,
                batch_size: int = None,
                bucketing: bool = False):

  with nn.stochastic(jax.random.PRNGKey(seed)):
    model = create_model(data_source.vocab_size, data_source.eos_idx)
    optimizer = flax.optim.Adam(learning_rate=learning_rate).create(model)

    if bucketing:
      train_batches = data_source.get_bucketed_batches(data_source.train_dataset,
                                          batch_size = batch_size,
                                          bucket_size = 8,
                                          drop_remainder = False,
                                          shuffle = True)
    else:
      train_batches = data_source.get_batches(
        data_source.train_dataset, batch_size=batch_size, shuffle=True)

    valid_batches = data_source.get_batches(
      data_source.valid_dataset, batch_size=batch_size, shuffle=True)


    bos_encoding = onehot(np.array([data_source.bos_idx]), data_source.vocab_size)
    train_metrics = {constants.ACC_KEY: 0, constants.LOSS_KEY:0}
    for epoch in range(num_epochs):

        no_batches = 0
        for batch in tfds.as_numpy(train_batches):
          batch_ohe = onehot_batch(batch, data_source.vocab_size)
          optimizer, metrics = train_step(optimizer, batch_ohe, nn.make_rng(), data_source.eos_idx)
          train_metrics = {key : train_metrics[key] + metrics[key] for key in train_metrics}
          no_batches += 1
        train_metrics = {key : train_metrics[key] / no_batches for key in train_metrics}
        # evaluate  
        valid_metrics = evaluate_model(optimizer.target, valid_batches, data_source, bos_encoding, logging_step=None)   
        log(epoch, train_metrics, valid_metrics)
        
    logging.info('Done training')

  return optimizer.target


def main(_):
  # prepare data source
  data_source = input_pipeline.CFQDataSource(seed = FLAGS.seed, max_output_length = FLAGS.max_query_length)

  # train model
  trained_model = train_model(
    learning_rate=FLAGS.learning_rate,
    num_epochs=1,
    # num_epochs=FLAGS.num_epochs,
    seed=FLAGS.seed,
    data_source=data_source,
    batch_size=FLAGS.batch_size,
    bucketing=False)


if __name__ == '__main__':
  app.run(main)
