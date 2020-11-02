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
EMBEDDING_DIM = 512
ATTENTION_SIZE = 512
ATTENTION_LAYER_SIZE = 512
NUM_LAYERS = 2
HORIZONTAL_DROPOUT = 0
VERTICAL_DROPOUT = 0.4
EMBED_DROPOUT = 0
ATTENTION_DROPOUT = 0

"""Common modules"""
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

    context = jnp.sum(jnp.expand_dims(scores, 2) * values, axis=1)
    return context, scores  # Shape: <float32>[batch_size, seq_len]


class MultilayerLSTM(nn.Module):
  "LSTM cell with multiple layers"

  def apply(self,
            num_layers: int,
            horizontal_dropout_masks: jnp.array,
            vertical_dropout_rate: float,
            input: jnp.array, previous_states: List,
            train: bool):
    """
    Args
      num_layers: number of layers
      horizontal_dropout_masks: dropout masks for each layer (the same dropout
        mask is used at each time step and applied on the hidden state that is
        fed into the cell to the right). This is the recurrent dropout used in
        https://arxiv.org/abs/1512.05287. [num_layers, batch_size, hidden_size]
      vertical_dropout_rate: dropout rate between layers, yet the dropout mask
        is not reused across timestamps. It's only applied if the number of
        layers > 1.
      input: input given to the first LSTM layer [batch_size, input_size]
      previous_states: list of (c,h) for each layer
        shape [num_layers, batch_size, 2*hidden_size]
      train: boolean indicating training or inference flow.
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
      if layer_idx != 0 and vertical_dropout_rate > 0:
        input = nn.dropout(input,
                           rate=vertical_dropout_rate,
                           deterministic=train)
      state, output = cell((c, h), input)
      states.append(state)
      input = output
      final_output = output
    return states, final_output


class Encoder(nn.Module):
  """LSTM encoder, returning state after EOS is input."""

  def apply(self, inputs: jnp.array, lengths: jnp.array,
            shared_embedding: nn.Module, train: bool, hidden_size: int,
            num_layers: int, horizontal_dropout_rate: float,
            vertical_dropout_rate: float,
            embed_dropout_rate: float = EMBED_DROPOUT):

    inputs = shared_embedding(inputs)
    inputs = nn.dropout(inputs, rate=embed_dropout_rate, deterministic=train)
    lstm = nn.LSTM.partial(
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout_rate=vertical_dropout_rate,
        recurrent_dropout_rate=horizontal_dropout_rate).shared(name='lstm')
    outputs, final_states = lstm(inputs, lengths, train=train)
    return outputs, final_states


def compute_attention_masks(mask_shape: Tuple, lengths: jnp.array):
  mask = np.ones(mask_shape, dtype=bool)
  mask = mask * (lengths[:, jnp.newaxis] > jnp.arange(mask.shape[1]))
  return mask


"""Baseline modules."""
class Decoder(nn.Module):
  """LSTM decoder."""

  @staticmethod
  def create_dropout_masks(num_masks: int, shape: Tuple,
                           dropout_rate: float, train: bool):
    if not train or dropout_rate == 0:
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
            horizontal_dropout_rate: float,
            vertical_dropout_rate: float,
            embed_dropout_rate: float = EMBED_DROPOUT,
            attention_layer_dropout: float = ATTENTION_DROPOUT,
            train: bool = False):
    """
    The decoder follows Luong's decoder in how attention is used (the current
    decoder state is used for attention computation, and the attention vector is
    used to get the final probability distribution instead of the RNN output).
    The attention mechanism employed is Bahdanau (also called concat/mlp).
    
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
      horizontal_dropout_rate: dropout applied at the same layer (same mask
        across time steps).
      vertical_dropout_rate: dropout applied between layers.
      embed_dropout_rate: dropout applied on the embeddings.
      attention_layer_dropout: dropout applied on the attention layer input (on
        the concatenation of current state and context vector)
      train: boolean choosing from train and inference flow
    """
    multilayer_lstm_cell = MultilayerLSTM.partial(num_layers=num_layers).shared(
        name='multilayer_lstm')
    attention_layer = nn.Dense.shared(features=ATTENTION_LAYER_SIZE,
                                      name='attention_layer')
    projection = nn.Dense.shared(features=vocab_size, name='projection')
    mlp_attention = MlpAttention.partial(hidden_size=ATTENTION_SIZE).shared(
        name='attention')
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
        dropout_rate=horizontal_dropout_rate,
        train=train)

    def decode_step_fn(carry, x):
      rng, previous_states, last_prediction, prev_attention = carry
      carry_rng, categorical_rng = jax.random.split(rng, 2)
      if not train:
        x = last_prediction
      x = shared_embedding(x)
      x = nn.dropout(x, rate=embed_dropout_rate, deterministic=train)
      lstm_input = jnp.concatenate([x, prev_attention], axis=-1)
      states, h = multilayer_lstm_cell(
        horizontal_dropout_masks=h_dropout_masks,
        vertical_dropout_rate=vertical_dropout_rate,
        input=lstm_input,
        previous_states=previous_states,
        train=train)
      context, scores = mlp_attention(jnp.expand_dims(h, 1), projected_keys,
                                      encoder_hidden_states, attention_mask)
      context_and_state = jnp.concatenate([context, h], axis=-1)
      context_and_state = nn.dropout(context_and_state,
                                     rate=attention_layer_dropout,
                                     deterministic=train)
      attention = jnp.tanh(attention_layer(context_and_state))
      logits = projection(attention)
      predicted_tokens = jax.random.categorical(categorical_rng, logits)
      predicted_tokens_uint8 = jnp.asarray(predicted_tokens, dtype=jnp.uint8)
      new_carry = (carry_rng, states, predicted_tokens_uint8, attention)
      new_x = (logits, predicted_tokens_uint8, scores)
      return new_carry, new_x

    # initialisig the LSTM states and final output with the
    # encoder hidden states
    attention = jnp.zeros((batch_size, ATTENTION_LAYER_SIZE))
    init_carry = (nn.make_rng(), init_states, inputs[:, 0], attention)

    if self.is_initializing():
      # initialize parameters before scan
      decode_step_fn(init_carry, inputs[:, 0])

    _, (logits, predictions, scores) = jax_utils.scan_in_dim(
        decode_step_fn,
        init=init_carry,  # rng, lstm_state, last_pred
        xs=inputs,
        axis=1)
    # The attention weights are only examined on the evaluation flow, so this
    # if is used to avoid unnecesary operations.
    if not self.is_initializing() and not train:
      attention_weights = jnp.array(scores)
      # Going from [output_seq_len, batch_size, input_seq_len]
      # to [batch_size, output_seq_len, input_seq_len].
      jnp.swapaxes(attention_weights, 1, 2)
    else:
      attention_weights = None
    return logits, predictions, attention_weights


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
            horizontal_dropout_rate=HORIZONTAL_DROPOUT,
            vertical_dropout_rate=VERTICAL_DROPOUT):
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
                              horizontal_dropout_rate=horizontal_dropout_rate,
                              vertical_dropout_rate=vertical_dropout_rate)
    decoder = Decoder.partial(num_layers=num_layers,
                              horizontal_dropout_rate=horizontal_dropout_rate,
                              vertical_dropout_rate=vertical_dropout_rate)
    # compute attention masks
    mask = compute_attention_masks(encoder_inputs.shape, encoder_inputs_lengths)

    # Encode inputs
    hidden_states, init_decoder_states = encoder(encoder_inputs,
                                                 encoder_inputs_lengths,
                                                 shared_embedding, train)
    # Decode outputs.
    logits, predictions, attention_weights = decoder(init_decoder_states,
                                                     hidden_states,
                                                     mask,
                                                     decoder_inputs[:, :-1],
                                                     shared_embedding,
                                                     vocab_size,
                                                     train=train)

    return logits, predictions, attention_weights


"""Syntax based modules."""

class SyntaxBasedDecoder(nn.Module):
  """LSTM syntax-based decoder."""

  @staticmethod
  def create_dropout_masks(num_masks: int, shape: Tuple,
                           dropout_rate: float, train: bool):
    if not train or dropout_rate == 0:
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
            horizontal_dropout_rate: float,
            vertical_dropout_rate: float,
            embed_dropout_rate: float = EMBED_DROPOUT,
            train: bool = False):
    """TODO
    """
    multilayer_lstm_cell = MultilayerLSTM.partial(num_layers=num_layers).shared(
        name='multilayer_lstm')
    projection = nn.Dense.shared(features=vocab_size, name='projection')
    mlp_attention = MlpAttention.partial(hidden_size=ATTENTION_SIZE).shared(
        name='attention')
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
        dropout_rate=horizontal_dropout_rate,
        train=train)

    def decode_step_fn(carry, x):
      rng, multilayer_lstm_output, last_prediction = carry
      previous_states, h = multilayer_lstm_output
      carry_rng, categorical_rng = jax.random.split(rng, 2)
      if not train:
        x = last_prediction
      x = shared_embedding(x)
      x = nn.dropout(x, rate=embed_dropout_rate, deterministic=train)
      dec_prev_state = jnp.expand_dims(h, 1)
      context, scores = mlp_attention(dec_prev_state, projected_keys,
                                      encoder_hidden_states, attention_mask)
      lstm_input = jnp.concatenate([x, context], axis=-1)
      states, h = multilayer_lstm_cell(
        horizontal_dropout_masks=h_dropout_masks,
        vertical_dropout_rate=vertical_dropout_rate,
        input=lstm_input,
        previous_states=previous_states,
        train=train)
      logits = projection(h)
      predicted_tokens = jax.random.categorical(categorical_rng, logits)
      predicted_tokens_uint8 = jnp.asarray(predicted_tokens, dtype=jnp.uint8)
      new_carry = (carry_rng, (states, h), predicted_tokens_uint8)
      new_x = (logits, predicted_tokens_uint8, scores)
      return new_carry, new_x

    # initialisig the LSTM states and final output with the
    # encoder hidden states
    multilayer_lstm_output = (init_states, init_states[-1][1])
    init_carry = (nn.make_rng(), multilayer_lstm_output, inputs[:, 0])

    if self.is_initializing():
      # initialize parameters before scan
      decode_step_fn(init_carry, inputs[:, 0])

    _, (logits, predictions, scores) = jax_utils.scan_in_dim(
        decode_step_fn,
        init=init_carry,  # rng, lstm_state, last_pred
        xs=inputs,
        axis=1)
    # The attention weights are only examined on the evaluation flow, so this
    # if is used to avoid unnecesary operations.
    if not self.is_initializing() and not train:
      attention_weights = jnp.array(scores)
      # Going from [output_seq_len, batch_size, input_seq_len]
      # to [batch_size, output_seq_len, input_seq_len].
      jnp.swapaxes(attention_weights, 1, 2)
    else:
      attention_weights = None
    return logits, predictions, attention_weights


class Seq2tree(nn.Module):
  """Sequence-to-ast class following Yin and Neubig."""

  def apply(self,
            encoder_inputs: jnp.array,
            decoder_inputs: jnp.array,
            encoder_inputs_lengths: jnp.array,
            vocab_size: int,
            emb_dim: int = EMBEDDING_DIM,
            train: bool = True,
            hidden_size: int = LSTM_HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            horizontal_dropout_rate=HORIZONTAL_DROPOUT,
            vertical_dropout_rate=VERTICAL_DROPOUT):
    """TODO
    """
    shared_embedding = nn.Embed.shared(
        num_embeddings=vocab_size,
        features=emb_dim,
        embedding_init=nn.initializers.normal(stddev=1.0))

    encoder = Encoder.partial(hidden_size=hidden_size,
                              num_layers=num_layers,
                              horizontal_dropout_rate=horizontal_dropout_rate,
                              vertical_dropout_rate=vertical_dropout_rate)
    decoder = SyntaxBasedDecoder.partial(num_layers=num_layers,
                                         horizontal_dropout_rate=horizontal_dropout_rate,
                                         vertical_dropout_rate=vertical_dropout_rate)
    # compute attention masks
    mask = compute_attention_masks(encoder_inputs.shape, encoder_inputs_lengths)

    # Encode inputs
    hidden_states, init_decoder_states = encoder(encoder_inputs,
                                                 encoder_inputs_lengths,
                                                 shared_embedding, train)
    # Decode outputs.
    logits, predictions, attention_weights = decoder(init_decoder_states,
                                                     hidden_states,
                                                     mask,
                                                     decoder_inputs[:, :-1],
                                                     shared_embedding,
                                                     vocab_size,
                                                     train=train)

    return logits, predictions, attention_weights

