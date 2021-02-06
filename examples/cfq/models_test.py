from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import random
import jax.numpy as jnp
from flax import nn
from flax import linen

import models
from models import Encoder, MlpAttention, RecurrentDropoutMasks, Decoder,\
  Seq2seq, MultilayerLSTMCell, MultilayerLSTM, Seq2tree


class ModelsTest(parameterized.TestCase):

  def test_encoder(self):
    rng1, rng2 = random.split(random.PRNGKey(0))
    rngs = {'params': rng1, 'dropout': rng2}
    seq1 = [1, 0, 3]
    seq2 = [1, 0, 3]
    batch_size = 2
    hidden_size = 50
    seq_len = 3
    num_layers = 5
    inputs = jnp.array([seq1, seq2])
    lengths = jnp.array([len(seq1), len(seq2)])
    class DummyModule(linen.Module):

      @linen.compact
      def __call__(self, inputs, lengths, train):
        shared_embedding = linen.Embed(
          num_embeddings=10,
          features=20,
          embedding_init=nn.initializers.normal(stddev=1.0))
        encoder = Encoder(
          shared_embedding=shared_embedding,
          hidden_size=hidden_size,
          num_layers=num_layers,
          horizontal_dropout_rate=0.4,
          vertical_dropout_rate=0.4)
        return encoder(inputs, lengths, train)

    dummy_module = DummyModule()
    out, initial_params = dummy_module.init_with_output(
      rngs,
      inputs=inputs,
      lengths=lengths,
      train=True)
    outputs, hidden_states = out
    self.assertEqual(outputs.shape, (batch_size, seq_len, hidden_size))
    self.assertEqual(len(hidden_states), num_layers)
    for i in range(num_layers):
      c, h = hidden_states[i]
      self.assertEqual(c.shape, (batch_size, hidden_size))
      self.assertEqual(h.shape, (batch_size, hidden_size))

  def test_mlp_attenntion(self):
    rng = dict(params=random.PRNGKey(0))

    batch_size = 2
    seq_len = 4
    values_size = 3
    attention_size = 20
    q1 = [1, 2, 3]
    q2 = [4, 5, 6]
    query = jnp.array([[q1], [q2]])
    projected_keys = jnp.zeros((batch_size, seq_len, attention_size))
    values = jnp.array([[[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]],
                        [[5, 5, 5], [6, 6, 6], [7, 7, 7], [8, 8, 8]]])
    mask = jnp.array([[True, True, False, False], [True, True, True, False]])
    mlp_attention = MlpAttention(hidden_size=attention_size)

    (context, scores), _ = mlp_attention.init_with_output(rng,
      query=query,
      projected_keys=projected_keys,
      values=values,
      mask=mask)
    self.assertEqual(context.shape, (batch_size, values_size))
    self.assertEqual(scores.shape, (batch_size, seq_len))

  def test_recurrent_dropout_masks(self):
    rng1, rng2 = random.split(random.PRNGKey(0))
    rngs = {'params': rng1, 'dropout': rng2}
    dropout = RecurrentDropoutMasks(3, 0.3)
    masks, _ = dropout.init_with_output(rngs, (2, 10), True)
    self.assertEqual(len(masks), 3)
    for mask in masks:
      self.assertEqual(mask.shape, (2, 10))

  def test_multilayer_LSTM_cell(self):
    rng = dict(params=random.PRNGKey(0))
    num_layers = 3
    batch_size = 7
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
    prev_state_0 = (jnp.zeros((batch_size, hidden_size)),
      jnp.zeros((batch_size, hidden_size)))
    prev_state_1 = (jnp.zeros((batch_size, hidden_size)),
      jnp.zeros((batch_size, hidden_size)))
    prev_state_2 = (jnp.zeros((batch_size, hidden_size)),
      jnp.zeros((batch_size, hidden_size)))
    previous_states = [prev_state_0, prev_state_1, prev_state_2]
    multilayer_lstm = MultilayerLSTMCell(num_layers=num_layers)
    (states, y), _ = multilayer_lstm.init_with_output(rng,
      horizontal_dropout_masks=h_dropout_masks,
      vertical_dropout_rate=dropout_rate,
      input=input,
      previous_states=previous_states,
      train=False)
    self.assertEqual(len(states), num_layers)
    for state in states:
      c, h = state
      self.assertEqual(c.shape, (batch_size, hidden_size))
      self.assertEqual(h.shape, (batch_size, hidden_size))
    self.assertEqual(y.shape, (batch_size, hidden_size))

  def test_multilayer_LSTM(self):
    rng = dict(params=random.PRNGKey(0))
    num_layers = 5
    batch_size = 10
    input_size = 5
    seq_len = 7
    hidden_size = 20
    dropout_rate = 0.2
    recurrent_dropout_rate = 0.3
    inputs = jnp.zeros((batch_size, seq_len, input_size))
    lengths = jnp.array([5, 6, 7, 7, 6, 7, 5, 3, 5, 6])
    multilayer_lstm = MultilayerLSTM(
      hidden_size, num_layers, dropout_rate, recurrent_dropout_rate)
    (outputs, states), _ = multilayer_lstm.init_with_output(rng,
      inputs, lengths, False)
    self.assertEqual(outputs.shape, (batch_size, seq_len, hidden_size))
    for state in states:
      c, h = state
      self.assertEqual(c.shape, (batch_size, hidden_size))
      self.assertEqual(h.shape, (batch_size, hidden_size))
    rng1, rng2 = random.split(random.PRNGKey(0))
    rngs = {'params': rng1, 'dropout': rng2}
    (outputs, states), _ = multilayer_lstm.init_with_output(rngs,
      inputs, lengths, True)
    self.assertEqual(outputs.shape, (batch_size, seq_len, hidden_size))
    for state in states:
      c, h = state
      self.assertEqual(c.shape, (batch_size, hidden_size))
      self.assertEqual(h.shape, (batch_size, hidden_size))

  def test_compute_attention_masks(self):
    shape = (2, 7)
    lengths = jnp.array([5, 7])
    mask = models.compute_attention_masks(shape, lengths)
    expected_mask = jnp.array([[True, True, True, True, True, False, False],
                               [True, True, True, True, True, True, True]])
    self.assertEqual(True, jnp.array_equal(mask, expected_mask))

  def test_decoder_train(self):
    rng1, rng2, rng3 = random.split(random.PRNGKey(0), num=3)
    rngs = {'params': rng1, 'dropout': rng2, 'lstm': rng3}
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
    class DummyModule(linen.Module):

      @linen.compact
      def __call__(self):
        shared_embedding = linen.Embed(
          num_embeddings=vocab_size,
          features=20,
          embedding_init=nn.initializers.normal(stddev=1.0))
        decoder = Decoder(shared_embedding=shared_embedding,
                          vocab_size=vocab_size,
                          num_layers=num_layers,
                          horizontal_dropout_rate=0.4,
                          vertical_dropout_rate=0.4)
        return decoder(
          init_states=initial_states,
          encoder_hidden_states=enc_hidden_states,
          attention_mask=mask,
          inputs=inputs,
          train=True)
    
    out, initial_params = DummyModule().init_with_output(rngs)
    logits, predictions, scores = out
    self.assertEqual(logits.shape, (batch_size, seq_len, vocab_size))
    self.assertEqual(predictions.shape, (batch_size, seq_len))
    self.assertEqual(scores, None)

  def test_decoder_inference(self):
    rng1, rng2 = random.split(random.PRNGKey(0))
    rngs = {'params': rng1, 'lstm': rng2}
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

    class DummyModule(linen.Module):

      @linen.compact
      def __call__(self):
        shared_embedding = linen.Embed(
          num_embeddings=vocab_size,
          features=20,
          embedding_init=nn.initializers.normal(stddev=1.0))
        decoder = Decoder(shared_embedding=shared_embedding,
                          vocab_size=vocab_size,
                          num_layers=num_layers,
                          horizontal_dropout_rate=0.4,
                          vertical_dropout_rate=0.4)
        return decoder(
          init_states=initial_states,
          encoder_hidden_states=enc_hidden_states,
          attention_mask=mask,
          inputs=inputs,
          train=False)
    
    (logits, predictions, scores), _ = DummyModule().init_with_output(rngs)
    self.assertEqual(logits.shape, (batch_size, max_len, vocab_size))
    self.assertEqual(predictions.shape, (batch_size, max_len))
    self.assertEqual(scores.shape, (batch_size, max_len, input_seq_len))


  def test_seq_2_seq(self):
    rng1, rng2, rng3 = random.split(random.PRNGKey(0), num=3)
    rngs = {'params': rng1, 'dropout': rng2, 'lstm': rng3}
    vocab_size = 10
    batch_size = 2
    max_len = 5
    enc_inputs = jnp.array([[1, 0, 2], [1, 4, 2]], dtype=jnp.uint8)
    lengths = jnp.array([2, 3])
    dec_inputs = jnp.array([[6, 7, 3, 5, 1], [1, 4, 2, 3, 2]], dtype=jnp.uint8)
    seq2seq = Seq2seq(vocab_size=vocab_size)
    (logits, predictions, scores), _ = seq2seq.init_with_output(rngs,
                                          encoder_inputs=enc_inputs,
                                          decoder_inputs=dec_inputs,
                                          encoder_inputs_lengths=lengths,
                                          train=True)
    self.assertEqual(logits.shape, (batch_size, max_len - 1, vocab_size))
    self.assertEqual(predictions.shape, (batch_size, max_len - 1))
    self.assertEqual(scores, None)

  def est_seq_2_seq_inference_apply(self):
    vocab_size = 10
    batch_size = 2
    max_len = 5
    enc_inputs = jnp.array([[1, 0, 2], [1, 4, 2]], dtype=jnp.uint8)
    lengths = jnp.array([2, 3])
    dec_inputs = jnp.array([[6, 7, 3, 5, 1], [1, 4, 2, 3, 2]], dtype=jnp.uint8)
    input_length = 3
    predicted_length = 4
    seq2seq = Seq2seq.partial(vocab_size=vocab_size)
    with nn.stochastic(jax.random.PRNGKey(0)):
      _, initial_params = models.Seq2seq.partial(vocab_size=vocab_size
                            ).init_by_shape(nn.make_rng(),
                            [((1, 1), jnp.uint8),
                              ((1, 2), jnp.uint8),
                              ((1,), jnp.uint8)])
      model = nn.Model(models.Seq2seq, initial_params)
      logits, predictions, attention_weights = model(encoder_inputs=enc_inputs,
                                                 decoder_inputs=dec_inputs,
                                                 encoder_inputs_lengths=lengths,
                                                 vocab_size=vocab_size,
                                                 train=False)
      self.assertEqual(logits.shape, (batch_size, max_len - 1, vocab_size))
      self.assertEqual(predictions.shape, (batch_size, max_len - 1))
      self.assertEqual(
        attention_weights.shape, (batch_size, predicted_length, input_length))

  
  def est_seq_2_tree_train_apply(self):
    rule_vocab_size = 10
    token_vocab_size = 100
    node_vocab_size = 15
    batch_size = 2
    enc_inputs = jnp.array([[1, 0, 2], [1, 4, 2]], dtype=jnp.uint8)
    lengths = jnp.array([2, 3])
    dec_inputs = [
      # example 1
      [
        [0, 0, 1, 0], # action types
        [2, 3, 10, 0], # action_values
        [1, 12, 1, 0] # node types
      ],
      # example 2
      [
        [0, 1, 1, 0], # action types
        [5, 7, 13, 0], # action_values
        [2, 7, 3, 5] # node types
      ]
    ]
    dec_inputs = jnp.array(dec_inputs)
    input_length = 3
    predicted_length = 4
    with nn.stochastic(jax.random.PRNGKey(0)):
      _, initial_params = models.Seq2tree.partial(
                            rule_vocab_size=rule_vocab_size,
                            token_vocab_size=token_vocab_size,
                            node_vocab_size=node_vocab_size
                            ).init_by_shape(nn.make_rng(),
                            [((1, 1), jnp.uint8),
                              ((1, 3, 1), jnp.uint8),
                              ((1,), jnp.uint8)])
      model = nn.Model(models.Seq2tree, initial_params)
      rule_logits,\
      token_logits,\
      predictions,\
      attention_weights = model(encoder_inputs=enc_inputs,
                                decoder_inputs=dec_inputs,
                                encoder_inputs_lengths=lengths,
                                rule_vocab_size=rule_vocab_size,
                                token_vocab_size=token_vocab_size,
                                node_vocab_size=node_vocab_size,
                                train=True)
      self.assertEqual(rule_logits.shape,
                       (batch_size, predicted_length, rule_vocab_size))
      self.assertEqual(token_logits.shape,
                       (batch_size, predicted_length, token_vocab_size))
      self.assertEqual(predictions.shape, (batch_size, predicted_length))
      self.assertEqual(attention_weights, None)
  
  def est_seq_2_tree_inference_apply(self):
    rule_vocab_size = 20
    token_vocab_size = 100
    node_vocab_size = 15
    batch_size = 2
    max_len = 4
    enc_inputs = jnp.array([[1, 0, 2], [1, 4, 2]], dtype=jnp.uint8)
    lengths = jnp.array([2, 3])
    dec_inputs = jnp.zeros((batch_size, 3, max_len), dtype=jnp.uint8)
    input_length = 3
    predicted_length = 4
    with nn.stochastic(jax.random.PRNGKey(0)):
      _, initial_params = \
        models.Seq2tree.partial(rule_vocab_size=rule_vocab_size,
                                token_vocab_size=token_vocab_size,
                                node_vocab_size=node_vocab_size
                                ).init_by_shape(nn.make_rng(),
                                [((1, 1), jnp.uint8),
                                  ((1, 3, 1), jnp.uint8),
                                  ((1,), jnp.uint8)])
      model = nn.Model(models.Seq2tree, initial_params)
      rule_logits,\
      token_logits,\
      predictions,\
      attention_weights = model(encoder_inputs=enc_inputs,
                                decoder_inputs=dec_inputs,
                                encoder_inputs_lengths=lengths,
                                rule_vocab_size=rule_vocab_size,
                                token_vocab_size=token_vocab_size,
                                node_vocab_size=node_vocab_size,
                                train=False)
      self.assertEqual(rule_logits.shape,
                       (batch_size, max_len, rule_vocab_size))
      self.assertEqual(token_logits.shape,
                       (batch_size, max_len, token_vocab_size))
      self.assertEqual(predictions.shape, (batch_size, max_len))
      self.assertEqual(
        attention_weights.shape, (batch_size, predicted_length, input_length))


if __name__ == '__main__':
  absltest.main()
