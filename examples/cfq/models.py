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
"""Module with defined models"""

import numpy as np
import jax
import jax.numpy as jnp
from flax import jax_utils
from flax import nn

# hyperparams
LSTM_HIDDEN_SIZE = 512
EMBEDDING_DIM = 200

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


