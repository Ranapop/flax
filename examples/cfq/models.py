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
"""Flax modules composing the seq2seq LSTM architecture for CFQ"""
from typing import Tuple, List
import numpy as np
import jax
from jax import random
import jax.numpy as jnp
from flax import jax_utils
from flax import nn

# hyperparams
LSTM_HIDDEN_SIZE = 512
EMBEDDING_DIM = 200
ATTENTION_SIZE = 100
ATTENTION_LAYER_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.4

class Encoder(nn.Module):
  """LSTM encoder, returning state after EOS is input."""

  def apply(self, inputs: jnp.array, lengths: jnp.array,
            shared_embedding: nn.Module, train: bool, hidden_size: int,
            num_layers: int, horizontal_dropout_rate: float,
            vertical_dropout_rate: float):

    inputs = shared_embedding(inputs)
    lstm = nn.LSTM.partial(
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout_rate=vertical_dropout_rate,
        recurrent_dropout_rate=horizontal_dropout_rate).shared(name='lstm')
    outputs, final_states = lstm(inputs, lengths, train=train)
    return outputs, final_states


class MlpAttention(nn.Module):
  """MLP attention module that returns a scalar score for each key."""

  def apply(self,
            query: jnp.ndarray,
            projected_keys: jnp.ndarray,
            values: jnp.ndarray,
            mask: jnp.ndarray,
            hidden_size: int = None) -> jnp.ndarray:
    """Computes MLP-based attention based on keys and a query.

    Attention scores are computed by feeding the keys and query through an MLP.
    This results in a single scalar per key, and for each sequence the attention
    scores are normalized using a softmax so that they sum to 1. Invalid key
    positions are ignored as indicated by the mask. This is also called
    "Bahdanau attention" and was originally proposed in:
    ```
    Bahdanau et al., 2015. Neural Machine Translation by Jointly Learning to
    Align and Translate. ICLR. https://arxiv.org/abs/1409.0473
    ```

    Args:
      query: The query with which to compute attention. Shape:
        <float32>[batch_size, 1, query_size].
      projected_keys: The inputs multiplied with a weight matrix. Shape:
        <float32>[batch_size, seq_length, attention_size].
      values: The values to be weighted [batch_size, seq_length, values_size]
      mask: A mask that determines which values in `projected_keys` are valid.
        Only values for which the mask is True will get non-zero attention
        scores. <bool>[batch_size, seq_length].
      hidden_size: The hidden size of the MLP that computes the attention score.

    Returns:
      The weighted values (context vector) [batch_size, seq_len]
    """
    projected_query = nn.Dense(query, hidden_size, name='query', bias=False)
    # Query broadcasts in the sum below along the time (seq_length) dimension.
    energy = nn.tanh(projected_keys + projected_query)
    scores = nn.Dense(energy, 1, name='energy', bias=False)
    scores = scores.squeeze(-1)  # New shape: <float32>[batch_size, seq_len].
    # TODO: see if I can rewrite this to not have the squeezing and unsqueezing
    scores = jnp.where(mask, scores, -jnp.inf)  # Using exp(-inf) = 0 below.
    scores = nn.softmax(scores, axis=-1)

    attention = jnp.sum(jnp.expand_dims(scores, 2) * values, axis=1)
    return attention  # Shape: <float32>[batch_size, seq_len]


class LuongAttention(nn.Module):
  """Luong attention module that returns attention vector."""

  def apply(self,
            query: jnp.ndarray,
            projected_keys: jnp.ndarray,
            values: jnp.ndarray,
            mask: jnp.ndarray,
            decoder_state: jnp.ndarray,
            attention_layer_size: int = ATTENTION_LAYER_SIZE) -> jnp.ndarray:
    attention_layer = nn.Dense.partial(features=attention_layer_size)
    mlp_attention = MlpAttention.partial(hidden_size=ATTENTION_SIZE)
    context = mlp_attention(query, projected_keys, values, mask)
    context_and_state = jnp.concatenate([context, decoder_state], axis=-1)
    attention = jnp.tanh(attention_layer(context_and_state))
    return attention

class MultilayerLSTM(nn.Module):
  "LSTM cell with multiple layers"

  def apply(self,
            num_layers: int,
            horizontal_dropout_masks: jnp.array,
            vertical_dropout_masks: jnp.array,
            input: jnp.array, previous_states: List):
    """
    Args
      num_layers: number of layers
      horizontal_dropout_masks: dropout masks for each layer (the same dropout
        mask is used at each time step and applied on the hidden state that is
        fed into the cell to the right) [num_layers, batch_size, hidden_size]
      vertical_dropout_masks: dropout masks between layers, same masks reused
        accross time steps, applied on the hidden state that is fed to the upper
        layer (used only between layers, not on final output). The dropout is
        the one used here https://arxiv.org/abs/1512.05287 except it's not
        applied on the first layer inputs and final outputs to be consistent
        with the implementation in `recurrent.py`
        shape [num_layers-1, batch_size, hidden_size]
      input: input given to the first LSTM layer [batch_size, input_size]
      previous_states: list of (c,h) for each layer
        shape [num_layers, batch_size, 2*hidden_size]
    """
    states = []
    final_output = None
    for layer_idx in range(num_layers):
      lstm_name = 'lstm_layer' + str(layer_idx)
      cell = nn.LSTMCell.partial(name=lstm_name)
      c, h = previous_states[layer_idx]
      # Apply dropout to h.
      if horizontal_dropout_masks[layer_idx] is not None:
        h = h * horizontal_dropout_masks[layer_idx]
      # Apply dropout to the hidden state from lower layer.
      if layer_idx != 0 and vertical_dropout_masks[layer_idx - 1] is not None:
        input = input * vertical_dropout_masks[layer_idx - 1]
      state, output = cell((c, h), input)
      states.append(state)
      input = output
      final_output = output
    return states, final_output


class Decoder(nn.Module):
  """LSTM decoder."""

  @staticmethod
  def create_dropout_masks(num_masks: int, shape: Tuple,
                           dropout_rate: float):
    if dropout_rate == 0:
      return [None] * num_masks
    masks = []
    for i in range(0, num_masks):
      dropout_mask = random.bernoulli(nn.make_rng(),
                                      p=1 - dropout_rate,
                                      shape=shape)
      # Convert array of boolean values to probabilty distribution.
      dropout_mask = dropout_mask / (1.0 - dropout_rate)
      masks.append(dropout_mask)
    return masks

  def apply(self,
            init_states,
            encoder_hidden_states,
            attention_mask,
            inputs: jnp.array,
            shared_embedding: nn.Module,
            vocab_size: int,
            num_layers: int,
            horizontal_dropout_rate: int,
            vertical_dropout_rate: int,
            train: bool = False):
    """
    Args
      init_states: states to initialize the decoder hidden states (coming from
        the encoder). This assumes the encoder and decoder have the same number
        of layers [num_layers, batch_size, hidden_size]
      encoder_hidden_states: encoder hidden states
        [batch_size, input_seq_len, hidden_size]
      attention_mask: attention mask [batch_size, input_seq_len]
      inputs: on the train flow (train=True) the gold decoded sequence (as
        teacher forcing is used), on the inference flow an array of desired
        output length where the first token is the BOS token
      shared_embedding: module for computing the embeddings (shared with
        the encoder)
      vocab_size: vocabulary size
      num_layers: number of layers in the LSTM
      train: boolean choosing from train and inference flow
    """
    multilayer_lstm_cell = MultilayerLSTM.partial(num_layers=num_layers).shared(
        name='multilayer_lstm')
    projection = nn.Dense.shared(features=vocab_size, name='projection')
    luong_attention = LuongAttention.shared(name='luong_attention')
    # The keys projection can be calculated once for the whole sequence.
    projected_keys = nn.Dense(encoder_hidden_states,
                              ATTENTION_SIZE,
                              name='keys',
                              bias=False)

    batch_size = encoder_hidden_states.shape[0]
    hidden_size = encoder_hidden_states.shape[-1]
    h_dropout_masks = Decoder.create_dropout_masks(
        num_masks=num_layers,
        shape=(batch_size, hidden_size),
        dropout_rate=horizontal_dropout_rate)
    v_dropout_masks = Decoder.create_dropout_masks(
        num_masks=num_layers - 1,
        shape=(batch_size, hidden_size),
        dropout_rate=vertical_dropout_rate)

    def decode_step_fn(carry, x):
      rng, multilayer_lstm_output, last_prediction, prev_attention = carry
      previous_states, _ = multilayer_lstm_output
      carry_rng, categorical_rng = jax.random.split(rng, 2)
      if not train:
        x = last_prediction
      x = shared_embedding(x)
      lstm_input = jnp.concatenate([x, prev_attention], axis=-1)
      states, h = multilayer_lstm_cell(horizontal_dropout_masks=h_dropout_masks,
                                       vertical_dropout_masks=v_dropout_masks,
                                       input=lstm_input,
                                       previous_states=previous_states)
      attention = luong_attention(jnp.expand_dims(h, 1), projected_keys,
                                  encoder_hidden_states, attention_mask, h)
      logits = projection(attention)
      predicted_tokens = jax.random.categorical(categorical_rng, logits)
      predicted_tokens_uint8 = jnp.asarray(predicted_tokens, dtype=jnp.uint8)
      new_carry = (carry_rng, (states, h), predicted_tokens_uint8, attention)
      new_x = (logits, predicted_tokens_uint8)
      return new_carry, new_x

    # initialisig the LSTM states and final output with the
    # encoder hidden states
    multilayer_lstm_output = (init_states, init_states[-1][1])
    attention = jnp.zeros((batch_size, ATTENTION_LAYER_SIZE))
    init_carry = (nn.make_rng(), multilayer_lstm_output, inputs[:, 0], attention)

    if self.is_initializing():
      # initialize parameters before scan
      decode_step_fn(init_carry, inputs[:, 0])

    _, (logits, predictions) = jax_utils.scan_in_dim(
        decode_step_fn,
        init=init_carry,  # rng, lstm_state, last_pred
        xs=inputs,
        axis=1)
    return logits, predictions


def compute_attention_masks(mask_shape: Tuple, lengths: jnp.array):
  mask = np.ones(mask_shape, dtype=bool)
  mask = mask * (lengths[:, jnp.newaxis] > jnp.arange(mask.shape[1]))
  return mask


class Seq2seq(nn.Module):
  """Sequence-to-sequence class using encoder/decoder architecture."""

  def apply(self,
            encoder_inputs: jnp.array,
            decoder_inputs: jnp.array,
            encoder_inputs_lengths: jnp.array,
            vocab_size: int,
            emb_dim: int = EMBEDDING_DIM,
            train: bool = True,
            hidden_size: int = LSTM_HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT):
    """Run the seq2seq model.

    Args:
      encoder_inputs: padded batch of input sequences to encode (as vocab 
      indices), shaped `[batch_size, question length]`.
      decoder_inputs: padded batch of expected decoded sequences on the train
        flow, shaped `[batch_size, query len]` on the train flow and shaped
        `[batch_size, max query len]` on the inference flow.
        During inference (i.e., `train = False`), the initial time step
        is forced into the model and samples are used for the following inputs.
        The second dimension of this tensor determines how many steps will be
        decoded, regardless of the value of `teacher_force`.
      encoder_inputs_lengths: input sequences lengths [batch_size]
      vocab_size: size of vocabulary
      emb_dim: embedding dimension
      train: bool, differentiating between train and inference flow. This is
        needed for the dropout on both the encoder and decoder, and on the
        decoder it also activates teacher forcing (the gold sequence used).
      hidden_size: int, the number of hidden dimensions in the encoder and
        decoder LSTMs.
      Returns:
        Array of decoded logits.
      """
    shared_embedding = nn.Embed.shared(
        num_embeddings=vocab_size,
        features=emb_dim,
        embedding_init=nn.initializers.normal(stddev=1.0))

    encoder = Encoder.partial(hidden_size=hidden_size,
                              num_layers=num_layers,
                              horizontal_dropout_rate=dropout,
                              vertical_dropout_rate=dropout)
    decoder = Decoder.partial(num_layers=num_layers,
                              horizontal_dropout_rate=dropout,
                              vertical_dropout_rate=dropout)
    # compute attention masks
    mask = compute_attention_masks(encoder_inputs.shape, encoder_inputs_lengths)

    # Encode inputs
    hidden_states, init_decoder_states = encoder(encoder_inputs,
                                                 encoder_inputs_lengths,
                                                 shared_embedding, train)
    # Decode outputs.
    logits, predictions = decoder(init_decoder_states,
                                  hidden_states,
                                  mask,
                                  decoder_inputs[:, :-1],
                                  shared_embedding,
                                  vocab_size,
                                  train=train)

    return logits, predictions
