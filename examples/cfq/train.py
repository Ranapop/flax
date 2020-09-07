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

import input_pipeline as inp
import constants

BatchType = Dict[Text, jnp.array]

ACC_KEY = 'accuracy'
LOSS_KEY = 'loss'
TRAIN_ACCURACIES = 'train acc'
TRAIN_LOSSES = 'train loss'
TEST_ACCURACIES = 'test acc'
TEST_LOSSES = 'test loss'
# hyperparams
LSTM_HIDDEN_SIZE = 512


def onehot(sequence: jnp.array, vocab_size: int) -> jnp.array:
    """One-hot encode a single sequence of integers."""
    return jnp.array(sequence[:, jnp.newaxis] == jnp.arange(vocab_size),
                     dtype=jnp.float32)


def onehot_batch(batch: BatchType, vocab_size: int) -> BatchType:
    """Encode a batch of examples into onehot (questions & queries)"""
    questions = batch[constants.QUESTION_KEY]
    queries = batch[constants.QUERY_KEY]
    questions_ohe = jnp.array([onehot(q, vocab_size) for q in questions])
    queries_ohe = jnp.array([onehot(q, vocab_size) for q in queries])
    return {
        constants.QUESTION_KEY: questions_ohe,
        constants.QUERY_KEY: queries_ohe,
        constants.QUESTION_LEN_KEY: batch[constants.QUESTION_LEN_KEY],
        constants.QUERY_LEN_KEY: batch[constants.QUERY_LEN_KEY]
    }


def decode_ohe_seq(ohe_seq: jnp.ndarray,
                   data_source: inp.CFQDataSource) -> Text:
    """Deocde sequence from ohe (list of ohe vectors) to string"""
    indices = ohe_seq.argmax(axis=-1)
    tokens = [data_source.i2w[i].decode() for i in indices]
    str_seq = ' '.join(tokens)
    return str_seq


def decode_onehot(batch_inputs: jnp.ndarray, data_source: inp.CFQDataSource):
    """Decode a batch of one-hot encoding to strings."""
    return np.array([decode_ohe_seq(seq, data_source) for seq in batch_inputs])


def mask_sequences(sequence_batch: jnp.array, lengths: jnp.array):
    """Set positions beyond the length of each sequence to 0."""
    return sequence_batch * (lengths[:, jnp.newaxis] > jnp.arange(
        sequence_batch.shape[1]))


class Encoder(nn.Module):
    """LSTM encoder, returning state after EOS is input."""

    def apply(self,
              inputs: jnp.array,
              eos_id: int,
              hidden_size: int = LSTM_HIDDEN_SIZE):
        "Apply encoder module"
        # inputs.shape = (batch_size, seq_length, vocab_size).
        batch_size = inputs.shape[0]

        lstm_cell = nn.LSTMCell.shared(name='lstm')
        init_lstm_state = nn.LSTMCell.initialize_carry(nn.make_rng(),
                                                       (batch_size,),
                                                       hidden_size)

        def encode_step_fn(carry, x):
            lstm_state, is_eos = carry
            new_lstm_state, y = lstm_cell(lstm_state, x)

            # Pass forward the previous state if EOS has already been reached.
            def select_carried_state(new_state, old_state):
                return jnp.where(is_eos[:, jnp.newaxis], old_state, new_state)

            # LSTM state is a tuple (c, h).
            carried_lstm_state = tuple(
                select_carried_state(*s)
                for s in zip(new_lstm_state, lstm_state))
            # Update `is_eos`.
            is_eos = jnp.logical_or(is_eos, x[:, eos_id])
            return (carried_lstm_state, is_eos), y

        init_carry = (init_lstm_state, jnp.zeros(batch_size, dtype=np.bool))
        if self.is_initializing():
            # initialize parameters before scan
            encode_step_fn(init_carry, inputs[:, 0])

        (final_state, _), _ = jax_utils.scan_in_dim(encode_step_fn,
                                                    init=init_carry,
                                                    xs=inputs,
                                                    axis=1)
        return final_state


class Decoder(nn.Module):
    """LSTM decoder."""

    def apply(self, init_state, inputs: jnp.array, teacher_force: bool = False):
        """Apply decoder model"""
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
              encoder_inputs: jnp.array,
              decoder_inputs: jnp.array,
              eos_id: int,
              teacher_force: bool = True,
              hidden_size: int = LSTM_HIDDEN_SIZE):
        """Run the seq2seq model.

    Args:
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


def create_model(vocab_size: int, eos_id: int) -> nn.Module:
    """Creates a seq2seq model."""
    _, initial_params = Seq2seq.partial(eos_id=eos_id).init_by_shape(
        # need to pass 2 for decoder length as the first token is cut off
        nn.make_rng(),
        [((1, 1, vocab_size), jnp.float32), ((1, 2, vocab_size), jnp.float32)])
    model = nn.Model(Seq2seq, initial_params)
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
      logits: predictions, shape (batch_size, logits seq_len, embedding_size)
      labels: gold labels, shape (batch_size, labels seq_len, embedding_size)
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


@jax.jit
def train_step(optimizer: Any, batch: BatchType, rng: Any, eos_id: int):
    """Train one step."""

    inputs = batch[constants.QUESTION_KEY]
    labels = batch[constants.QUERY_KEY]
    labels_no_bos = labels[:, 1:]
    queries_lengths = batch[constants.QUERY_LEN_KEY]

    def loss_fn(model):
        """Compute cross-entropy loss."""
        with nn.stochastic(rng):
            logits, _ = model(encoder_inputs=inputs,
                              decoder_inputs=labels,
                              eos_id=eos_id)
        loss = cross_entropy_loss(logits, labels_no_bos, queries_lengths)
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grad = grad_fn(optimizer.target)
    optimizer = optimizer.apply_gradient(grad)
    metrics = compute_metrics(logits, labels_no_bos, queries_lengths)
    return optimizer, metrics


@jax.partial(jax.jit, static_argnums=5)
def infer(model: nn.Module, inputs: jnp.array, rng: Any, eos_id: int,
          bos_encoding: jnp.array, predicted_output_length: int):
    """Apply model on inference flow and return predictions.

    Args:
        model: the seq2seq model applied
        inputs: batch of input sequences
        rng: rng
        eos_id: index of EOS token in the vocabulary
        bos_encoding: encoding of the BOS token
        predicted_output_length: what length should predict for the output
    """
    # This simply creates a batch (batch size = inputs.shape[0])
    # filled with sequences of max_output_len of only the bos encoding
    initial_dec_inputs = jnp.tile(bos_encoding,
                                  (inputs.shape[0], predicted_output_length, 1))
    with nn.stochastic(rng):
        _, predictions = model(encoder_inputs=inputs,
                               decoder_inputs=initial_dec_inputs,
                               eos_id=eos_id,
                               teacher_force=False)
    return predictions


def evaluate_model(model: nn.Module,
                   batches: tf.data.Dataset,
                   data_source: inp.CFQDataSource,
                   bos_encoding: jnp.array,
                   predicted_output_length: int,
                   no_logged_examples: int = None):
    """Evaluate the model on the validation/test batches

    Args:
        model: model
        batches: validation batches
        data_source: CFQ data source (needed for vocab size, w2i etc.)
        bos_encoding: encoding of BOS token
        predicted_output_length: how long the predicted sequence should be
        no_logged_examples: how many examples to log (they will be taken
                            from the first batch, so no_logged_examples
                            should be < batch_size)
                            if None, no logging
    """
    no_batches = 0
    avg_metrics = {ACC_KEY: 0, LOSS_KEY: 0}
    for batch in tfds.as_numpy(batches):
        batch_ohe = onehot_batch(batch, data_source.vocab_size)
        inputs = batch_ohe[constants.QUESTION_KEY]
        gold_outputs = batch_ohe[constants.QUERY_KEY][:, 1:]
        inferred_outputs = infer(model, inputs, nn.make_rng(),
                                 data_source.eos_idx, bos_encoding,
                                 predicted_output_length)
        metrics = compute_metrics(inferred_outputs, gold_outputs,
                                  batch_ohe[constants.QUERY_LEN_KEY])
        avg_metrics = {
            key: avg_metrics[key] + metrics[key] for key in avg_metrics
        }
        if no_logged_examples is not None and no_batches == 0:
            #log the first examples in the batch
            gold_seq = decode_onehot(gold_outputs, data_source)
            inferred_seq = decode_onehot(inferred_outputs, data_source)
            for i in range(0, no_logged_examples):
                logging.info('\nGold seq:\n %s\nInferred seq:\n %s\n',
                             gold_seq[i], inferred_seq[i])
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
        model = create_model(data_source.vocab_size, data_source.eos_idx)
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

        bos_encoding = onehot(np.array([data_source.bos_idx]),
                              data_source.vocab_size)
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
                batch_ohe = onehot_batch(batch, data_source.vocab_size)
                optimizer, metrics = train_step(optimizer, batch_ohe,
                                                nn.make_rng(),
                                                data_source.eos_idx)
                train_metrics = {
                    key: train_metrics[key] + metrics[key]
                    for key in train_metrics
                }
                no_batches += 1
                # only train for 1 batch (for now)
                if no_batches == 1:
                    break
            train_metrics = {
                key: train_metrics[key] / no_batches for key in train_metrics
            }
            # evaluate
            dev_metrics = evaluate_model(model=optimizer.target,
                                         batches=dev_batches,
                                         data_source=data_source,
                                         bos_encoding=bos_encoding,
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


def test_model(model_dir,
               data_source: inp.CFQDataSource,
               max_out_len: int,
               seed: int,
               batch_size: int):
    """Evaluate model at model_dir on dev subset"""
    with nn.stochastic(jax.random.PRNGKey(seed)):
        model = create_model(data_source.vocab_size, data_source.eos_idx)
        #do I need to pass learning_rate?
        optimizer = flax.optim.Adam().create(model)
        dev_batches = data_source.get_batches(data_source.dev_dataset,
                                              batch_size=batch_size,
                                              shuffle=True)
        bos_encoding = onehot(np.array([data_source.bos_idx]),
                              data_source.vocab_size)
        # evaluate
        dev_metrics = evaluate_model(model=optimizer.target,
                                     batches=dev_batches,
                                     data_source=data_source,
                                     bos_encoding=bos_encoding,
                                     predicted_output_length=max_out_len,
                                     no_logged_examples=3)
        logging.info('Loss %.4f, acc %.2f',
                      dev_metrics[LOSS_KEY],dev_metrics[ACC_KEY])
