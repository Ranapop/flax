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
from flax import linen
import functools

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

class MlpAttention(linen.Module):
  """MLP attention module that returns a scalar score for each key.

  Args:
    hidden_size: The hidden size of the MLP that computes the attention score.
    use_batch_axis: When True the code is executed at batch level, otherwise
      at example level (in that case the arguments don't have the batch_size
      dimensions, for example query would be [1, query_size]).
  """
  hidden_size: int = None
  use_batch_axis: bool = True

  @linen.compact
  def __call__(self,
            query: jnp.ndarray,
            projected_keys: jnp.ndarray,
            values: jnp.ndarray,
            mask: jnp.ndarray) -> jnp.ndarray:
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

    Returns:
      The weighted values (context vector) [batch_size, seq_len]
    """
    projected_query = linen.Dense(self.hidden_size, name='query', use_bias=False)(query)
    # Query broadcasts in the sum below along the time (seq_length) dimension.
    energy = linen.tanh(projected_keys + projected_query)
    scores = linen.Dense(1, name='energy', use_bias=False)(energy)
    scores = scores.squeeze(-1)  # New shape: <float32>[batch_size, seq_len].
    # TODO: see if I can rewrite this to not have the squeezing and unsqueezing
    scores = jnp.where(mask, scores, -jnp.inf)  # Using exp(-inf) = 0 below.
    scores = linen.softmax(scores, axis=-1)

    if self.use_batch_axis:
      # Shape: <float32>[batch_size, seq_len]
      context = jnp.sum(jnp.expand_dims(scores, 2) * values, axis=1)
    else:
      # Shape: <float32>[seq_len]
      context = jnp.sum(jnp.expand_dims(scores, 1) * values, axis=0)
    return context, scores  


class RecurrentDropoutMasks(linen.Module):
  """ Module for creating dropout masks for the recurrent cells.

  Attributes:
    num_masks: Number of masks.
    dropout_rate: Dropput rate.
  """
  num_masks: int
  dropout_rate: float

  @linen.compact
  def __call__(self, shape: Tuple, train: bool):
    if not train or self.dropout_rate == 0:
      return [None] * self.num_masks
    masks = []
    for i in range(0, self.num_masks):
      dropout_mask = random.bernoulli(self.make_rng('dropout'),
                                      p=1 - self.dropout_rate,
                                      shape=shape)
      # Convert array of boolean values to probabilty distribution.
      dropout_mask = dropout_mask / (1.0 - self.dropout_rate)
      masks.append(dropout_mask)
    return masks


class MultilayerLSTMCell(linen.Module):
  """LSTM cell with multiple layers
  
  Args:
    num_layers: number of layers
  """
  num_layers: int

  @linen.compact
  def __call__(self,
    horizontal_dropout_masks: jnp.array,
    vertical_dropout_rate: float,
    input: jnp.array, previous_states: List,
    train: bool):
    """
    Args
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
    for layer_idx in range(self.num_layers):
      lstm_name = 'lstm_layer' + str(layer_idx)
      cell = linen.LSTMCell(name=lstm_name)
      c, h = previous_states[layer_idx]
      # Apply dropout to h.
      if horizontal_dropout_masks[layer_idx] is not None:
        h = h * horizontal_dropout_masks[layer_idx]
      # Apply dropout to the hidden state from lower layer.
      if train and layer_idx != 0 and vertical_dropout_rate > 0:
        dropout = linen.Dropout(rate=vertical_dropout_rate)
        input = dropout(input)
      state, output = cell((c, h), input)
      states.append(state)
      input = output
      final_output = output
    return states, final_output

class MultilayerLSTMScan(linen.Module):
  num_layers: int
  h_dropout_masks: List
  dropout_rate: float
  train: bool

  @functools.partial(
      linen.transforms.scan,
      variable_broadcast='params',
      split_rngs={'params': False},
      in_axes = 1,
      out_axes = 1,
      )
  @linen.compact
  def __call__(self, carry, x):
    multilayer_lstm_cell = MultilayerLSTMCell(
      num_layers=self.num_layers,
      name='multilayer_lstm')
    
    previous_states, lengths, step = carry
    states, h = multilayer_lstm_cell(
      horizontal_dropout_masks=self.h_dropout_masks,
      vertical_dropout_rate=self.dropout_rate,
      input=x,
      previous_states=previous_states,
      train=self.train)

    def get_carried_state(old_state, new_state):
      (old_c,old_h) = old_state
      (new_c, new_h) = new_state
      c = jnp.where(step < jnp.expand_dims(lengths, 1), new_c, old_c)
      h = jnp.where(step < jnp.expand_dims(lengths, 1), new_h, old_h)
      return (c,h)
    
    carried_states = [get_carried_state(*layer_states)\
                        for layer_states in zip(previous_states, states)]
    new_carry = carried_states, lengths, step+1
    return new_carry, h

class MultilayerLSTM(linen.Module):
  """"Multilayer LSTM.

  Attributes:
    hiden_size: LSTM cell hidden size.
    num_layers: Number of LSTMs stacked.
    dropout_rate: Dropout rate (between layers).
    recurrent_dropout_rate: Recurrent dropout rate.
  """
  hidden_size: int
  num_layers: int
  dropout_rate: float
  recurrent_dropout_rate: float

  @linen.compact
  def __call__(self,
               inputs, lengths,
               train):
    """
    Returns:
      (outputs, final_states)
      outputs: array of shape (batch_size, seq_len, hidden_size).
      final_states: list of length num_layers, where each element is an array
        of shape (batch_size, hidden_size)
    """
    batch_size = inputs.shape[0]
    dropout_masks = RecurrentDropoutMasks(self.num_layers,
                                          self.recurrent_dropout_rate)
    h_dropout_masks = dropout_masks(
      shape=(batch_size, self.hidden_size),
      train=train)
    lstm_scan = MultilayerLSTMScan(self.num_layers,
                                   h_dropout_masks,
                                   self.dropout_rate,
                                   train)
    initial_states = [
          linen.LSTMCell.initialize_carry(
            jax.random.PRNGKey(0), (batch_size,), self.hidden_size)
          for _ in range(self.num_layers)]
    init_carry = init=(initial_states, lengths, jnp.array(0))
    (final_states,_, _), outputs = lstm_scan(init_carry, inputs)
    return outputs, final_states


class Encoder(linen.Module):
  """LSTM encoder, returning state after EOS is input.
  
  Attributes:
    shared_embedding: Embedding module shared between encoder & decoder.
    hidden_size: LSTM hidden size.
    num_layers: Number of stacked LSTMs.
    horizontal_dropout_rate: Dropout rate.
    vertical_dropout_rate: Recurrent dropout rate.
    embed_dropout_rate: Embedding dropout rate.
  """
  shared_embedding: linen.Module
  hidden_size: int
  num_layers: int
  horizontal_dropout_rate: float
  vertical_dropout_rate: float
  embed_dropout_rate: float = EMBED_DROPOUT

  @linen.compact
  def __call__(self, inputs: jnp.array, lengths: jnp.array, train: bool):
    """
    Args:
      inputs: Encoder inputs (batch of sequences). Shape [batch_size, seq_len].
      lengths: Input lenghts, shape [batch_size].
      train: Train flow flag.
    """
    inputs = self.shared_embedding(inputs)
    embed_dropout = linen.Dropout(self.embed_dropout_rate)
    inputs = embed_dropout(inputs, deterministic=train)
    lstm = MultilayerLSTM(
      hidden_size=self.hidden_size,
      num_layers=self.num_layers,
      dropout_rate=self.vertical_dropout_rate,
      recurrent_dropout_rate=self.horizontal_dropout_rate,
      name='lstm')
    outputs, final_states = lstm(inputs, lengths, train=train)
    return outputs, final_states


def compute_attention_masks(mask_shape: Tuple, lengths: jnp.array):
  mask = np.ones(mask_shape, dtype=bool)
  mask = mask * (lengths[:, jnp.newaxis] > jnp.arange(mask.shape[1]))
  return mask


class DecoderLSTM(linen.Module):
  """
  Attributes:
    shared_embedding: Embedding module shared between encoder & decoder.
    encoder_hidden_states: Encoder hidden states.
    projected_keys: Keys (enc states) passed through a dense layer.
    attention_mask: Attention mask.
    train: Train flag.
    vocab_size: Vocabulary size.
    num_layers: Number of LSTM layers.
    h_dropout_masks: Dropout masks.
    vertical_dropout_rate: Dropout rate.
    embed_dropout_rate: Embedding dropout rate.
    attention_layer_dropout: Attention dropout rate.
  """
  shared_embedding: linen.Module
  encoder_hidden_states: jnp.array
  projected_keys: jnp.array
  attention_mask: jnp.array
  train: bool
  vocab_size: int
  num_layers: int
  h_dropout_masks: List
  vertical_dropout_rate: float
  embed_dropout_rate: float = EMBED_DROPOUT
  attention_layer_dropout: float = ATTENTION_DROPOUT

  def setup(self):
    self.multilayer_lstm_cell = MultilayerLSTMCell(num_layers=self.num_layers)
    self.attention_layer = linen.Dense(features=ATTENTION_LAYER_SIZE)
    self.projection = linen.Dense(features=self.vocab_size)
    self.mlp_attention = MlpAttention(hidden_size=ATTENTION_SIZE)
    self.embed_dropout = linen.Dropout(self.embed_dropout_rate)
    self.attention_dropout = linen.Dropout(self.attention_layer_dropout)

  @functools.partial(
      linen.transforms.scan,
      variable_broadcast='params',
      split_rngs={'params': False},
      in_axes = 1,
      out_axes = 1)
  def __call__(self, carry, x):
    rng, previous_states, last_prediction, prev_attention = carry
    carry_rng, categorical_rng = jax.random.split(rng, 2)
    if not self.train:
      x = last_prediction
    x = self.shared_embedding(x)
    x = self.embed_dropout(x, deterministic=self.train)
    lstm_input = jnp.concatenate([x, prev_attention], axis=-1)
    states, h = self.multilayer_lstm_cell(
      horizontal_dropout_masks=self.h_dropout_masks,
      vertical_dropout_rate=self.vertical_dropout_rate,
      input=lstm_input,
      previous_states=previous_states,
      train=self.train)
    context, scores = self.mlp_attention(jnp.expand_dims(h, 1), 
      self.projected_keys, self.encoder_hidden_states, self.attention_mask)
    context_and_state = jnp.concatenate([context, h], axis=-1)
    context_and_state = self.attention_dropout(
      context_and_state, deterministic=self.train)
    attention = jnp.tanh(self.attention_layer(context_and_state))
    logits = self.projection(attention)
    predicted_tokens = jax.random.categorical(categorical_rng, logits)
    predicted_tokens_uint8 = jnp.asarray(predicted_tokens, dtype=jnp.uint8)
    new_carry = (carry_rng, states, predicted_tokens_uint8, attention)
    new_x = (logits, predicted_tokens_uint8, scores)
    return new_carry, new_x


"""Baseline modules."""
class Decoder(linen.Module):
  """LSTM decoder.
  
  Attributes:
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
  """
  shared_embedding: linen.Module
  vocab_size: int
  num_layers: int
  horizontal_dropout_rate: float
  vertical_dropout_rate: float
  embed_dropout_rate: float = EMBED_DROPOUT
  attention_layer_dropout: float = ATTENTION_DROPOUT

  @linen.compact
  def __call__(self,
    init_states: jnp.array,
    encoder_hidden_states: jnp.array,
    attention_mask: jnp.array,
    inputs: jnp.array,
    train: bool = False):
    """
    The decoder follows Luong's decoder in how attention is used (the current
    decoder state is used for attention computation, and the attention vector is
    used to get the final probability distribution instead of the RNN output).
    The attention mechanism employed is Bahdanau (also called concat/mlp).
    
    Args
      init_states: states to initialize the decoder hidden states (coming from
        the encoder). This assumes the encoder and decoder have the same number
        of layers. [num_layers, 2, batch_size, hidden_size]
      encoder_hidden_states: encoder hidden states
        [batch_size, input_seq_len, hidden_size]
      attention_mask: attention mask [batch_size, input_seq_len]
      inputs: on the train flow (train=True) the gold decoded sequence (as
        teacher forcing is used), on the inference flow an array of desired
        output length where the first token is the BOS token
      train: boolean choosing from train and inference flow
    """
    batch_size = encoder_hidden_states.shape[0]
    hidden_size = encoder_hidden_states.shape[-1]
    dropout_masks = RecurrentDropoutMasks(self.num_layers,
                                          self.horizontal_dropout_rate)
    h_dropout_masks = dropout_masks(
      shape=(batch_size, hidden_size),
      train=train)
    # The keys projection can be calculated once for the whole sequence.
    projected_keys = linen.Dense(ATTENTION_SIZE, use_bias=False)(
      encoder_hidden_states)
    decoder_lstm = DecoderLSTM(
      self.shared_embedding,
      encoder_hidden_states,
      projected_keys,
      attention_mask,
      train,
      self.vocab_size,
      self.num_layers,
      h_dropout_masks,
      self.vertical_dropout_rate,
      self.embed_dropout_rate,
      self.attention_layer_dropout
    )

    # initialisig the LSTM states and final output with the
    # encoder hidden states
    attention = jnp.zeros((batch_size, ATTENTION_LAYER_SIZE))
    init_carry = (self.make_rng('lstm'), init_states, inputs[:, 0], attention)

    _, (logits, predictions, scores) = decoder_lstm(init_carry, inputs)
    # The attention weights are only examined on the evaluation flow, so this
    # if is used to avoid unnecesary operations.
    if not train:
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

ACTION_EMBEDDING_SIZE = 128
NODE_EMBEDDING_SIZE = 32

class ActionEmbed(nn.Module):

  def apply(self,
            action_type: jnp.array,
            action_value: jnp.array,
            rule_vocab_size,
            token_embedding: nn.Module):
    rule_embedding = nn.Embed.partial(
      num_embeddings=rule_vocab_size,
      features=ACTION_EMBEDDING_SIZE,
      embedding_init=nn.initializers.normal(stddev=1.0))

    rule_emb = rule_embedding(action_value)
    token_emb = token_embedding(action_value)
    # Return the rule_emb in case the action_type is 0, token_emb in case the
    # action type is 1 and a zero-valued vector otherwise.
    action_emb = jnp.equal(action_type, jnp.array(0)) * rule_emb + \
     jnp.equal(action_type, jnp.array(1)) * token_emb
    return action_emb
class SyntaxBasedDecoder(nn.Module):
  """LSTM syntax-based decoder."""

  def apply(self,
            init_states: jnp.array,
            encoder_hidden_states: jnp.array,
            attention_mask: jnp.array,
            inputs: jnp.array,
            shared_embedding: nn.Module,
            rule_vocab_size: int,
            token_vocab_size: int,
            node_vocab_size: int,
            num_layers: int,
            horizontal_dropout_rate: float,
            vertical_dropout_rate: float,
            embed_dropout_rate: float = EMBED_DROPOUT,
            train: bool = False):
    """
    Args:
      init_states: initial state (c,h) for each layer
        [num_layers, 2, hidden_size].
      encoder_hidden_states: h vector for each input token
        [input_seq_len, hidden_size].
      attention_mask: attention mask [input_seq_len].
      inputs: decoder inputs [output_seq_len].
      shared_embedding: token embedding module.
      rule_vocab_size: rule vocab size.
      token_vocab_size: token vocab size.
      num_layers: number of LSTM layers.
      horizontal_dropout_rate: LSTM horizontal dropout rate.
      horizontal_dropout_rate: LSTM vertical dropout rate.
      embed_dropout_rate: embedding dropout rate.
      train: flag distinguishing between train and test flow.
    """
    multilayer_lstm_cell = MultilayerLSTMCell.partial(num_layers=num_layers).shared(
        name='multilayer_lstm')
    rule_projection = nn.Dense.shared(features=rule_vocab_size,
                                      name='rule_projection')
    token_projection = nn.Dense.shared(features=token_vocab_size,
                                       name='token_projection')
    mlp_attention = MlpAttention(
      hidden_size=ATTENTION_SIZE,
      use_batch_axis=False,
      name='attention')
    action_embedding = ActionEmbed.shared(rule_vocab_size=rule_vocab_size,
                                          token_embedding=shared_embedding,
                                          name='action_embedding')
    node_embedding = nn.Embed.shared(
        name='node_embedding',
        num_embeddings=node_vocab_size,
        features=NODE_EMBEDDING_SIZE,
        embedding_init=nn.initializers.normal(stddev=1.0))
    # The keys projection can be calculated once for the whole sequence.
    projected_keys = nn.Dense(encoder_hidden_states,
                              ATTENTION_SIZE,
                              name='keys',
                              bias=False)
    
    hidden_size = encoder_hidden_states.shape[-1]
    h_dropout_masks = create_dropout_masks(
        num_masks=num_layers,
        shape=(hidden_size),
        dropout_rate=horizontal_dropout_rate,
        train=train)

    def decode_step_fn(carry, x):
      action_type = jnp.asarray(x[0], dtype=jnp.uint8)
      action_value = jnp.asarray(x[1], dtype=jnp.uint8)
      node_type = jnp.asarray(x[2], dtype=jnp.uint8)
      rng, multilayer_lstm_output, previous_action = carry
      previous_states, h = multilayer_lstm_output
      carry_rng, categorical_rng = jax.random.split(rng, 2)
      prev_action_emb = action_embedding(action_type=previous_action[0],
                                         action_value=previous_action[1])
      prev_action_emb = nn.dropout(prev_action_emb,
                                   rate=embed_dropout_rate,
                                   deterministic=not train)
      dec_prev_state = jnp.expand_dims(h, 0)
      context, scores = mlp_attention(dec_prev_state, projected_keys,
                                      encoder_hidden_states, attention_mask)
      node_emb = node_embedding(node_type)
      lstm_input = jnp.concatenate([prev_action_emb, node_emb, context], axis=-1)
      states, h = multilayer_lstm_cell(
        horizontal_dropout_masks=h_dropout_masks,
        vertical_dropout_rate=vertical_dropout_rate,
        input=lstm_input,
        previous_states=previous_states,
        train=train)
      rule_logits = rule_projection(h)
      token_logits = token_projection(h)
      predicted_rules = jax.random.categorical(categorical_rng, rule_logits)
      predicted_tokens = jax.random.categorical(categorical_rng, token_logits)
      prediction = jnp.where(action_type, predicted_tokens, predicted_rules)
      prediction_uint8 = jnp.asarray(prediction, dtype=jnp.uint8)
      if not train:
        action_value = prediction_uint8
      current_action = (action_type, action_value)
      new_carry = (carry_rng, (jnp.array(states), h), current_action)
      accumulator = (rule_logits, token_logits, prediction_uint8, scores)
      return new_carry, accumulator

    # initialisig the LSTM states and final output with the
    # encoder hidden states
    # Convention: initial action has type 2.
    initial_action = (jnp.array(2, dtype=jnp.uint8), jnp.array(0, dtype=jnp.uint8))
    multilayer_lstm_output = (init_states, init_states[-1, 1, :])
    init_carry = (nn.make_rng(), multilayer_lstm_output, initial_action)

    if self.is_initializing():
      # initialize parameters before scan
      initial_x = jnp.zeros((3,), dtype=jnp.uint8)
      decode_step_fn(init_carry, initial_x)

    # Go from [2, out_seq_len] -> [out_seq_len, 2].
    scan_inputs = jnp.swapaxes(inputs, 0, 1)

    _, (rule_logits, token_logits, predictions, scores) = jax_utils.scan_in_dim(
        decode_step_fn,
        init=init_carry,  # rng, lstm_state, last_pred
        xs=scan_inputs,
        axis=0)
    # The attention weights are only examined on the evaluation flow, so this
    # if is used to avoid unnecesary operations.
    if not self.is_initializing() and not train:
      attention_weights = jnp.array(scores)
      # Going from [batch_size, output_seq_len, input_seq_len]
      # to [batch_size, input_seq_len, output_seq_len].
      jnp.swapaxes(attention_weights, 0, 1)
    else:
      attention_weights = None
    return rule_logits, token_logits, predictions, attention_weights


class Seq2tree(nn.Module):
  """Sequence-to-ast class following Yin and Neubig."""

  def apply(self,
            encoder_inputs: jnp.array,
            decoder_inputs: jnp.array,
            encoder_inputs_lengths: jnp.array,
            rule_vocab_size: int,
            token_vocab_size: int,
            node_vocab_size: int,
            emb_dim: int = ACTION_EMBEDDING_SIZE,
            train: bool = True,
            hidden_size: int = LSTM_HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            horizontal_dropout_rate=HORIZONTAL_DROPOUT,
            vertical_dropout_rate=VERTICAL_DROPOUT):
    shared_embedding = nn.Embed.shared(
        num_embeddings=token_vocab_size,
        features=emb_dim,
        embedding_init=nn.initializers.normal(stddev=1.0))

    encoder = Encoder.partial(hidden_size=hidden_size,
                              num_layers=num_layers,
                              horizontal_dropout_rate=horizontal_dropout_rate,
                              vertical_dropout_rate=vertical_dropout_rate)
    decoder = SyntaxBasedDecoder.partial(
                                   num_layers=num_layers,
                                   horizontal_dropout_rate=horizontal_dropout_rate,
                                   vertical_dropout_rate=vertical_dropout_rate,
                                   train=train,
                                   rule_vocab_size=rule_vocab_size,
                                   token_vocab_size=token_vocab_size,
                                   node_vocab_size=node_vocab_size,
                                   shared_embedding=shared_embedding)
    vmapped_decoder = jax.vmap(decoder)
    # compute attention masks
    mask = compute_attention_masks(encoder_inputs.shape, encoder_inputs_lengths)

    # Encode inputs
    hidden_states, init_decoder_states = encoder(encoder_inputs,
                                                 encoder_inputs_lengths,
                                                 shared_embedding, train)
    # [no_layers, 2, batch, hidden_size] -> [batch, no_layers, 2, hidden_size]
    init_decoder_states = jnp.array(init_decoder_states)
    init_decoder_states = jnp.swapaxes(init_decoder_states, 0, 2)
    # Decode outputs.
    rule_logits,\
    token_logits,\
    predictions,\
    attention_weights = vmapped_decoder(init_decoder_states,
                                        hidden_states,
                                        mask,
                                        decoder_inputs)

    return rule_logits, token_logits, predictions, attention_weights

