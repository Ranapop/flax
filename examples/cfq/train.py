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
from flax.training import common_utils

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


# vmap?
def indices_to_str(batch_inputs: jnp.ndarray, data_source: inp.CFQDataSource):
    """Decode a batch of one-hot encoding to strings."""
    return np.array(
        [data_source.indices_to_sequence_string(seq) for seq in batch_inputs])


def mask_sequences(sequence_batch: jnp.array, lengths: jnp.array):
    """Set positions beyond the length of each sequence to 0."""
    return sequence_batch * (lengths[:, jnp.newaxis] > jnp.arange(
        sequence_batch.shape[1]))


class Encoder(nn.Module):
    """LSTM encoder, returning state after EOS is input."""

    def apply(self,
              inputs: jnp.array,
              shared_embedding: nn.Module,
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

        inputs = shared_embedding(inputs)
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

    def apply(self, 
              init_state,
              inputs: jnp.array,
              shared_embedding: nn.Module,
              vocab_size: int,
              teacher_force: bool = False):
        """Apply decoder model"""
        lstm_cell = nn.LSTMCell.shared(name='lstm')
        projection = nn.Dense.shared(features=vocab_size, name='projection')

        def decode_step_fn(carry, x):
            rng, lstm_state, last_prediction = carry
            carry_rng, categorical_rng = jax.random.split(rng, 2)
            if not teacher_force:
                x = last_prediction
            x = shared_embedding(x)
            lstm_state, y = lstm_cell(lstm_state, x)
            logits = projection(y)
            predicted_tokens = jax.random.categorical(categorical_rng, logits)
            return (carry_rng, lstm_state, predicted_tokens), (logits, predicted_tokens)

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
              vocab_size: int,
              emb_dim: int = 512,
              teacher_force: bool = True,
              hidden_size: int = LSTM_HIDDEN_SIZE):
        """Run the seq2seq model.

    Args:
      encoder_inputs: padded batch of input sequences to encode (as vocab 
      indices), shaped `[batch_size, question length]`.
      decoder_inputs: padded batch of expected decoded sequences for teacher
        forcing, shaped `[batch_size, query len]` on the train flow and shaped
        `[batch_size, max query len]` on the inference flow.
        During inference (i.e., `teacher_force = False`), the initial time step
        is forced into the model and samples are used for the following inputs.
        The second dimension of this tensor determines how many steps will be
        decoded, regardless of the value of `teacher_force`.
      vocab_size: size of vocabulary
      emb_dim: embedding dimension
      teacher_force: bool, whether to use `decoder_inputs` as input to the
        decoder at every step. If False, only the first input is used, followed
        by samples taken from the previous output logits.
      eos_id: int, the token signaling when the end of a sequence is reached.
      hidden_size: int, the number of hidden dimensions in the encoder and
        decoder LSTMs.
      Returns:
        Array of decoded logits.
      """

        
        shared_embedding = nn.Embed.shared(
          num_embeddings=vocab_size,
          features=emb_dim,
          embedding_init=nn.initializers.normal(stddev=1.0))
        encoder, decoder = self._create_modules(eos_id, hidden_size)

        # Encode inputs
        init_decoder_state = encoder(
            encoder_inputs,
            shared_embedding)
        # Decode outputs.
        logits, predictions = decoder(
            init_decoder_state,
            # why remove the last elem??
            decoder_inputs[:, :-1],
            shared_embedding,
            vocab_size,
            teacher_force=teacher_force)

        return logits, predictions


def create_model(vocab_size: int, eos_id: int) -> nn.Module:
    """Creates a seq2seq model."""
    _, initial_params = Seq2seq.partial(eos_id=eos_id,
                                        vocab_size=vocab_size).init_by_shape(
                                            nn.make_rng(),
                                            [((1, 1), jnp.int32),
                                            # need to pass 2 for decoder length
                                            # as the first token is cut off
                                            ((1, 2), jnp.int32)])
    model = nn.Model(Seq2seq, initial_params)
    return model


def cross_entropy_loss(logits: jnp.array, labels: jnp.array,
                       lengths: jnp.array, vocab_size: int):
    """Returns cross-entropy loss.
    
    Args:
        logits: [batch_size, sequence length, vocab size] float array
        labels: [batch_size, sequence length] int array
        lengths: [batch_size] int array
    """
    log_soft = nn.log_softmax(logits)
    # multiplying with ohe selects the gold index
    ohe_labels = common_utils.onehot(labels, vocab_size)
    log_sum = jnp.sum(log_soft * ohe_labels, axis=-1)
    masked_log_sum = jnp.mean(mask_sequences(log_sum, lengths))
    return -masked_log_sum


def pad_along_axis(array: jnp.array,
                   curr_seq_len: int,
                   target_seq_len: int,
                   axis: int) -> jnp.array:
    """Returns array padded on axis to target_len"""
    ndim = array.ndim
    pad_shape = jnp.full((ndim, 2), 0)
    padding_size = target_seq_len - curr_seq_len
    pad_shape = jax.ops.index_update(pad_shape,
                                     jax.ops.index[axis,1],
                                     padding_size)
    padded = jnp.pad(array,pad_shape)
    return padded


def compute_metrics(logits: jnp.array,
                    predictions: jnp.array,
                    labels: jnp.array,
                    queries_lengths: jnp.array,
                    vocab_size: int) -> Dict:
    """Computes metrics for a batch of logist & labels and returns those metrics

    The metrics computed are cross entropy loss and mean batch accuracy. The
    accuracy at sequence level needs perfect matching of the compared sequences
    Args:
      logits: predicted probability distributions over the vocab,
              shape (batch_size, seq_len, vocab_size)
      predictions: predicted tokens [batch_size, seq_len]
      labels: gold labels [batch_size, labels seq_len]
      queries_lengths: lengths of gold queries (until eos) [batch_size]

    """
    lengths = queries_lengths
    labels_seq_len = labels.shape[1]
    logits_seq_len = logits.shape[1]
    max_seq_len = max(labels_seq_len, logits_seq_len)
    if labels_seq_len != max_seq_len:
        labels = pad_along_axis(labels, labels_seq_len, max_seq_len, axis=1)
    elif logits_seq_len != max_seq_len:
        logits = pad_along_axis(logits, logits_seq_len, max_seq_len, axis=1)
        predictions = pad_along_axis(predictions, logits_seq_len, max_seq_len,
                                     axis=1)

    loss = cross_entropy_loss(logits, labels, lengths, vocab_size)
    token_accuracy = jnp.equal(predictions, labels)
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


@jax.partial(jax.jit, static_argnums=4)
def train_step(optimizer: Any,
               batch: BatchType,
               rng: Any,
               eos_id: int,
               vocab_size: int):
    """Train one step."""

    inputs = batch[constants.QUESTION_KEY]
    labels = batch[constants.QUERY_KEY]
    labels_no_bos = labels[:, 1:]
    queries_lengths = batch[constants.QUERY_LEN_KEY]

    def loss_fn(model):
        """Compute cross-entropy loss."""
        with nn.stochastic(rng):
            logits, predictions = model(encoder_inputs=inputs,
                                  decoder_inputs=labels,
                                  eos_id=eos_id,
                                  vocab_size=vocab_size)
        loss = cross_entropy_loss(logits,
                                  labels_no_bos,
                                  queries_lengths,
                                  vocab_size)
        return loss, (logits, predictions)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, values), grad = grad_fn(optimizer.target)
    logits, predictions = values
    optimizer = optimizer.apply_gradient(grad)
    metrics = {}
    metrics = compute_metrics(logits,
                              predictions,
                              labels_no_bos,
                              queries_lengths,
                              vocab_size)
    return optimizer, metrics


@jax.partial(jax.jit, static_argnums=[4,6])
def infer(model: nn.Module,
          inputs: jnp.array,
          rng: Any,
          eos_id: int,
          vocab_size: int,
          bos_encoding: jnp.array,
          predicted_output_length: int):
    """Apply model on inference flow and return predictions.

    Args:
        model: the seq2seq model applied
        inputs: batch of input sequences
        rng: rng
        eos_id: index of EOS token in the vocabulary
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
                               eos_id=eos_id,
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
        gold_outputs = batch[constants.QUERY_KEY][:, 1:]
        logits, inferred_outputs = infer(model, inputs, nn.make_rng(),
                                         data_source.eos_idx,
                                         data_source.vocab_size,
                                         data_source.bos_idx,
                                         predicted_output_length)
        metrics = compute_metrics(logits,
                                  inferred_outputs,
                                  gold_outputs,
                                  batch[constants.QUERY_LEN_KEY],
                                  data_source.vocab_size)
        avg_metrics = {
            key: avg_metrics[key] + metrics[key] for key in avg_metrics
        }
        if no_logged_examples is not None and no_batches == 0:
            #log the first examples in the batch
            gold_seq = indices_to_str(gold_outputs, data_source)
            inferred_seq = indices_to_str(inferred_outputs, data_source)
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

        # dummy train on dummy data
        # batch_size = 2
        # eos_id = 1
        # vocab_size = 10
        # questions = jnp.array([ [3, 2, 1 ],[5,7,1] ])
        # queries = jnp.array([ [4, 8, 2, 1 ],[2,4,6,1] ])
        # queries_lengths = jnp.array([2,2])
        # questions_lengths = jnp.array([3,3])
        # batch = {
        #     constants.QUESTION_KEY: questions,
        #     constants.QUERY_KEY: queries,
        #     constants.QUESTION_LEN_KEY: questions_lengths,
        #     constants.QUERY_LEN_KEY: queries_lengths
        # }
        # optimizer, metrics = train_step(optimizer, 
        #                                 batch,
        #                                 nn.make_rng(),
        #                                 eos_id,
        #                                 vocab_size)

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
                optimizer, metrics = train_step(optimizer, 
                                                batch,
                                                nn.make_rng(),
                                                data_source.eos_idx,
                                                data_source.vocab_size)
                train_metrics = {
                    key: train_metrics[key] + metrics[key]
                    for key in train_metrics
                }
                no_batches += 1
                # only train for 1 batch (for now)
                # if no_batches == 2:
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
        # evaluate
        dev_metrics = evaluate_model(model=optimizer.target,
                                     batches=dev_batches,
                                     data_source=data_source,
                                     predicted_output_length=max_out_len,
                                     no_logged_examples=3)
        logging.info('Loss %.4f, acc %.2f',
                      dev_metrics[LOSS_KEY],dev_metrics[ACC_KEY])
