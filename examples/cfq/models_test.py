from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp
from flax import nn

import models
from models import Encoder, MlpAttention, Decoder, Seq2seq, MultilayerLSTM


class ModelsTest(parameterized.TestCase):

  def test_encoder(self):
    seq1 = [1, 0, 3]
    seq2 = [1, 0, 3]
    batch_size = 2
    hidden_size = 50
    seq_len = 3
    num_layers = 5
    inputs = jnp.array([seq1, seq2])
    lengths = jnp.array([len(seq1), len(seq2)])
    with nn.stochastic(jax.random.PRNGKey(0)):
      shared_embedding = nn.Embed.partial(
          num_embeddings=10,
          features=20,
          embedding_init=nn.initializers.normal(stddev=1.0))
      encoder = Encoder.partial(hidden_size=hidden_size,
                                num_layers=num_layers,
                                horizontal_dropout_rate=0.4,
                                vertical_dropout_rate=0.4)
      out, initial_params = encoder.init(nn.make_rng(),
                                         inputs=inputs,
                                         lengths=lengths,
                                         shared_embedding=shared_embedding,
                                         train=True)
      outputs, hidden_states = out
      self.assertEqual(outputs.shape, (batch_size, seq_len, hidden_size))
      self.assertEqual(len(hidden_states), num_layers)
      for i in range(num_layers):
        c, h = hidden_states[i]
        self.assertEqual(c.shape, (batch_size, hidden_size))
        self.assertEqual(h.shape, (batch_size, hidden_size))

  def test_mlp_attenntion(self):
    batch_size = 2
    seq_len = 4
    values_size = 3
    attention_size = 20
    q1 = [1, 2, 3, 4, 5, 6]
    q2 = [7, 8, 9, 10, 11, 12]
    query = jnp.array([[q1], [q2]])
    projected_keys = jnp.zeros((batch_size, seq_len, attention_size))
    values = jnp.array([[[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]],
                        [[5, 5, 5], [6, 6, 6], [7, 7, 7], [8, 8, 8]]])
    mask = jnp.array([[True, True, False, False], [True, True, True, False]])
    mlp_attention = MlpAttention.partial(hidden_size=attention_size)
    with nn.stochastic(jax.random.PRNGKey(0)):
      (context, scores), _ = mlp_attention.init(nn.make_rng(),
                                                query=query,
                                                projected_keys=projected_keys,
                                                values=values,
                                                mask=mask)
      self.assertEqual(context.shape, (batch_size, values_size))
      self.assertEqual(scores.shape, (batch_size, seq_len))

  def test_multilayer_LSTM(self):
    num_layers = 3
    batch_size = 2
    input_size = 5
    hidden_size = 20
    dropout_mask_0 = jnp.zeros((batch_size, hidden_size))
    dropout_mask_1 = jnp.zeros((batch_size, hidden_size))
    dropout_mask_2 = jnp.zeros((batch_size, hidden_size))
    dropout_mask_3 = jnp.zeros((batch_size, hidden_size))
    dropout_mask_4 = jnp.zeros((batch_size, hidden_size))
    h_dropout_masks = [dropout_mask_0, dropout_mask_1, dropout_mask_2]
    dropout_rate = 0.2
    input = jnp.zeros((batch_size, input_size))
    prev_state_0 = jnp.zeros((batch_size, hidden_size))
    prev_state_1 = jnp.zeros((batch_size, hidden_size))
    prev_state_2 = jnp.zeros((batch_size, hidden_size))
    previous_states = [prev_state_0, prev_state_1, prev_state_2]
    multilayer_lstm = MultilayerLSTM.partial(num_layers=num_layers)
    with nn.stochastic(jax.random.PRNGKey(0)):
      (states,
       y), _ = multilayer_lstm.init(nn.make_rng(),
                                    horizontal_dropout_masks=h_dropout_masks,
                                    vertical_dropout_rate=dropout_rate,
                                    input=input,
                                    previous_states=previous_states,
                                    train=False)

  def test_compute_attention_masks(self):
    shape = (2, 7)
    lengths = jnp.array([5, 7])
    mask = models.compute_attention_masks(shape, lengths)
    expected_mask = jnp.array([[True, True, True, True, True, False, False],
                               [True, True, True, True, True, True, True]])
    self.assertEqual(True, jnp.array_equal(mask, expected_mask))

  def test_decoder_train(self):
    seq1 = [1, 0, 2, 4]
    seq2 = [1, 4, 2, 3]
    inputs = jnp.array([seq1, seq2], dtype=jnp.uint8)
    batch_size = 2
    hidden_size = 50
    vocab_size = 10
    seq_len = len(seq1)
    input_seq_len = 7
    num_layers = 5
    initial_state = (jnp.zeros(
        (batch_size, hidden_size)), jnp.zeros((batch_size, hidden_size)))
    initial_states = [initial_state] * num_layers
    enc_hidden_states = jnp.zeros((batch_size, input_seq_len, hidden_size))
    mask = jnp.zeros((batch_size, input_seq_len), dtype=bool)
    with nn.stateful() as state:
      with nn.stochastic(jax.random.PRNGKey(0)):
        shared_embedding = nn.Embed.partial(
            num_embeddings=vocab_size,
            features=20,
            embedding_init=nn.initializers.normal(stddev=1.0))
        decoder = Decoder.partial(vocab_size=vocab_size,
                                  num_layers=num_layers,
                                  horizontal_dropout_rate=0.4,
                                  vertical_dropout_rate=0.4)
        out, initial_params = decoder.init(
            nn.make_rng(),
            init_states=initial_states,
            encoder_hidden_states=enc_hidden_states,
            attention_mask=mask,
            inputs=inputs,
            shared_embedding=shared_embedding,
            train=True)
        logits, predictions = out
        self.assertEqual(logits.shape, (batch_size, seq_len, vocab_size))
        self.assertEqual(predictions.shape, (batch_size, seq_len))
    state_dict = state.as_dict()
    self.assertEqual(state_dict, {})

  def test_decoder_inference(self):
    max_len = 4
    input_seq_len = 5
    seq1 = [1, 1, 1, 1]
    seq2 = [1, 1, 1, 1]
    # only the seq len and first token matters on inference flow
    inputs = jnp.array([seq1, seq2], dtype=jnp.uint8)
    batch_size = 2
    hidden_size = 50
    vocab_size = 10
    num_layers = 5
    initial_state = (jnp.zeros(
        (batch_size, hidden_size)), jnp.zeros((batch_size, hidden_size)))
    initial_states = [initial_state] * num_layers
    enc_hidden_states = jnp.zeros((batch_size, input_seq_len, hidden_size))
    mask = jnp.zeros((batch_size, input_seq_len), dtype=bool)
    with nn.stateful() as state:
      with nn.stochastic(jax.random.PRNGKey(0)):
        shared_embedding = nn.Embed.partial(
            num_embeddings=vocab_size,
            features=20,
            embedding_init=nn.initializers.normal(stddev=1.0))
        decoder = Decoder.partial(vocab_size=vocab_size,
                                  num_layers=num_layers,
                                  horizontal_dropout_rate=0.4,
                                  vertical_dropout_rate=0.4)
        (logits,
        predictions), _ = decoder.init(nn.make_rng(),
                                        init_states=initial_states,
                                        encoder_hidden_states=enc_hidden_states,
                                        attention_mask=mask,
                                        inputs=inputs,
                                        shared_embedding=shared_embedding,
                                        train=False)
        self.assertEqual(logits.shape, (batch_size, max_len, vocab_size))
        self.assertEqual(predictions.shape, (batch_size, max_len))
    state_dict = state.as_dict()
    self.assertEqual(state_dict, {})

  def test_seq_2_seq(self):
    vocab_size = 10
    batch_size = 2
    max_len = 5
    enc_inputs = jnp.array([[1, 0, 2], [1, 4, 2]], dtype=jnp.uint8)
    lengths = jnp.array([2, 3])
    dec_inputs = jnp.array([[6, 7, 3, 5, 1], [1, 4, 2, 3, 2]], dtype=jnp.uint8)
    seq2seq = Seq2seq.partial(vocab_size=vocab_size)
    with nn.stateful() as state:
      with nn.stochastic(jax.random.PRNGKey(0)):
        (logits, predictions), _ = seq2seq.init(nn.make_rng(),
                                                encoder_inputs=enc_inputs,
                                                decoder_inputs=dec_inputs,
                                                encoder_inputs_lengths=lengths)
        self.assertEqual(logits.shape, (batch_size, max_len - 1, vocab_size))
        self.assertEqual(predictions.shape, (batch_size, max_len - 1))
    state_dict = state.as_dict()
    self.assertEqual(state_dict, {})


  def test_seq_2_seq_inference_apply(self):
    vocab_size = 10
    batch_size = 2
    max_len = 5
    enc_inputs = jnp.array([[1, 0, 2], [1, 4, 2]], dtype=jnp.uint8)
    lengths = jnp.array([2, 3])
    dec_inputs = jnp.array([[6, 7, 3, 5, 1], [1, 4, 2, 3, 2]], dtype=jnp.uint8)
    input_length = 3
    predicted_length = 4
    seq2seq = Seq2seq.partial(vocab_size=vocab_size)
    with nn.stateful() as state:
      with nn.stochastic(jax.random.PRNGKey(0)):
        _, initial_params = models.Seq2seq.partial(vocab_size=vocab_size
                              ).init_by_shape(nn.make_rng(),
                              [((1, 1), jnp.uint8),
                               ((1, 2), jnp.uint8),
                               ((1,), jnp.uint8)])
        model = nn.Model(models.Seq2seq, initial_params)
        logits, predictions = model(encoder_inputs=enc_inputs,
                                    decoder_inputs=dec_inputs,
                                    encoder_inputs_lengths=lengths,
                                    vocab_size=vocab_size,
                                    train=False)
        self.assertEqual(logits.shape, (batch_size, max_len - 1, vocab_size))
        self.assertEqual(predictions.shape, (batch_size, max_len - 1))
    state_dict = state.as_dict()
    decoder_state = state_dict['/decoder']
    attention_weigts = decoder_state['attention']
    self.assertEqual(
      attention_weigts.shape, (batch_size, predicted_length, input_length))


if __name__ == '__main__':
  absltest.main()
