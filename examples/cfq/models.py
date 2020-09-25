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
from typing import Tuple
import numpy as np
import jax
import jax.numpy as jnp
from flax import jax_utils
from flax import nn

# hyperparams
LSTM_HIDDEN_SIZE = 512
EMBEDDING_DIM = 200
ATTENTION_SIZE = 100
DECODER_PROJECTION = 256

class Encoder(nn.Module):
  """LSTM encoder, returning state after EOS is input."""

  def apply(self,
            inputs: jnp.array,
            lengths: jnp.array,
            shared_embedding: nn.Module,
            hidden_size: int = LSTM_HIDDEN_SIZE):

    inputs = shared_embedding(inputs)
    lstm = nn.LSTM.partial(hidden_size=hidden_size,
                           num_layers=1).shared(name='lstm')
    outputs, final_states = lstm(inputs, lengths)
    return outputs, final_states[-1]


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


class Decoder(nn.Module):
  """LSTM decoder."""

  def apply(self,
            init_state,
            encoder_hidden_states,
            attention_mask,
            inputs: jnp.array,
            shared_embedding: nn.Module,
            vocab_size: int,
            teacher_force: bool = False):
    lstm_cell = nn.LSTMCell.shared(name='lstm')
    pre_output_layer = nn.Dense.shared(features=DECODER_PROJECTION,
                                       name='pre_output_layer')
    projection = nn.Dense.shared(features=vocab_size,
                                 name='projection')
    mlp_attention = MlpAttention.partial(hidden_size=ATTENTION_SIZE).shared(
        name='attention')
    # The keys projection can be calculated once for the whole sequence.
    projected_keys = nn.Dense(encoder_hidden_states,
                              ATTENTION_SIZE,
                              name='keys',
                              bias=False)

    def decode_step_fn(carry, x):
      rng, lstm_state, last_prediction = carry
      carry_rng, categorical_rng = jax.random.split(rng, 2)
      if not teacher_force:
        x = last_prediction
      x = shared_embedding(x)
      _,h = lstm_state
      dec_prev_state = jnp.expand_dims(h, 1)
      attention = mlp_attention(dec_prev_state, projected_keys,
                                encoder_hidden_states, attention_mask)
      lstm_input = jnp.concatenate([x,attention], axis=-1)
      lstm_state, h = lstm_cell(lstm_state, lstm_input)
      inner_proj_input = jnp.concatenate([x, attention, h], axis=-1)
      pre_output = pre_output_layer(inner_proj_input)
      logits = projection(pre_output)
      predicted_tokens = jax.random.categorical(categorical_rng, logits)
      predicted_tokens_uint8 = jnp.asarray(predicted_tokens, dtype=jnp.uint8)
      return (carry_rng, lstm_state,
              predicted_tokens_uint8), (logits, predicted_tokens_uint8)

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


def compute_attention_masks(mask_shape: Tuple, lengths: jnp.array):
  mask = np.ones(mask_shape, dtype=bool)
  mask = mask * (lengths[:, jnp.newaxis] > jnp.arange(mask.shape[1]))
  return mask


class Seq2seq(nn.Module):
  """Sequence-to-sequence class using encoder/decoder architecture."""

  def _create_modules(self, hidden_size):
    encoder = Encoder.partial(hidden_size=hidden_size).shared(name='encoder')
    decoder = Decoder.shared(name='decoder')
    return encoder, decoder

  def apply(self,
            encoder_inputs: jnp.array,
            decoder_inputs: jnp.array,
            encoder_inputs_lengths: jnp.array,
            vocab_size: int,
            emb_dim: int = EMBEDDING_DIM,
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
      encoder_inputs_lengths: input sequences lengths [batch_size]
      vocab_size: size of vocabulary
      emb_dim: embedding dimension
      teacher_force: bool, whether to use `decoder_inputs` as input to the
        decoder at every step. If False, only the first input is used, followed
        by samples taken from the previous output logits.
      hidden_size: int, the number of hidden dimensions in the encoder and
        decoder LSTMs.
      Returns:
        Array of decoded logits.
      """
    shared_embedding = nn.Embed.shared(
        num_embeddings=vocab_size,
        features=emb_dim,
        embedding_init=nn.initializers.normal(stddev=1.0))
    encoder, decoder = self._create_modules(hidden_size)

    # compute attention masks
    mask = compute_attention_masks(encoder_inputs.shape, encoder_inputs_lengths)

    # Encode inputs
    hidden_states, init_decoder_state = encoder(encoder_inputs,
                                                encoder_inputs_lengths,
                                                shared_embedding)
    # Decode outputs.
    logits, predictions = decoder(init_decoder_state,
                                  hidden_states,
                                  mask,
                                  decoder_inputs[:, :-1],
                                  shared_embedding,
                                  vocab_size,
                                  teacher_force=teacher_force)

    return logits, predictions
