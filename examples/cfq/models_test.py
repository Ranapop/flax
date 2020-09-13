from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp
from flax import nn

from models import Encoder, Decoder, Seq2seq

class ModelsTest(parameterized.TestCase):


    def test_encoder(self):
      seq1 = [1,0,3]
      seq2 = [1,0,3]
      batch_size = 2
      hidden_size = 50
      inputs = jnp.array([seq1, seq2])
      lengths = jnp.array([len(seq1), len(seq2)])
      with nn.stochastic(jax.random.PRNGKey(0)):
        shared_embedding = nn.Embed.partial(
            num_embeddings=10,
            features=20,
            embedding_init=nn.initializers.normal(stddev=1.0))
        encoder = Encoder.partial(hidden_size=hidden_size)
        out, initial_params = encoder.init(nn.make_rng(),
                            inputs=inputs,
                            lengths=lengths,
                            shared_embedding=shared_embedding)
        c, h = out
        self.assertEqual(c.shape, (batch_size, hidden_size))
        self.assertEqual(h.shape, (batch_size, hidden_size))
       
    def test_decoder_train(self):
      seq1 = [1,0,2,4]
      seq2 = [1,4,2,3]
      inputs = jnp.array([seq1, seq2])
      batch_size = 2
      hidden_size = 50
      vocab_size = 10
      seq_len = len(seq1)
      initial_state = (jnp.zeros((batch_size, hidden_size)),
                       jnp.zeros((batch_size, hidden_size)))
      with nn.stochastic(jax.random.PRNGKey(0)):
        shared_embedding = nn.Embed.partial(
            num_embeddings=vocab_size,
            features=20,
            embedding_init=nn.initializers.normal(stddev=1.0))
        decoder = Decoder.partial(vocab_size=vocab_size)
        out, initial_params = decoder.init(nn.make_rng(),
                              init_state = initial_state,
                              inputs=inputs,
                              shared_embedding=shared_embedding)
        logits, predictions = out
        self.assertEqual(logits.shape, (batch_size, seq_len, vocab_size))
        self.assertEqual(predictions.shape, (batch_size, seq_len))

    def test_decoder_inference(self):
      max_len = 4
      seq1 = [1,1,1,1]
      seq2 = [1,1,1,1]
      # only the seq len and first token matters on inference flow
      inputs = jnp.array([seq1, seq2])
      batch_size = 2
      hidden_size = 50
      vocab_size = 10
      initial_state = (jnp.zeros((batch_size, hidden_size)),
                       jnp.zeros((batch_size, hidden_size)))
      with nn.stochastic(jax.random.PRNGKey(0)):
        shared_embedding = nn.Embed.partial(
            num_embeddings=vocab_size,
            features=20,
            embedding_init=nn.initializers.normal(stddev=1.0))
        decoder = Decoder.partial(vocab_size=vocab_size)
        (logits, predictions), _ = decoder.init(nn.make_rng(),
                                     init_state = initial_state,
                                     inputs=inputs,
                                     shared_embedding=shared_embedding,
                                     teacher_force=False)
        self.assertEqual(logits.shape, (batch_size, max_len, vocab_size))
        self.assertEqual(predictions.shape, (batch_size, max_len))

    def test_seq_2_seq(self):
      vocab_size = 10
      batch_size = 2
      max_len = 5
      enc_inputs = jnp.array([[1,0,2], [1,4,2]])
      lengths = jnp.array([4, 4])
      dec_inputs = jnp.array([[6,7,3,5,1], [1,4,2,3,2]])
      seq2seq = Seq2seq.partial(vocab_size=vocab_size)
      with nn.stochastic(jax.random.PRNGKey(0)):
        (logits, predictions), _ = seq2seq.init(nn.make_rng(),
                                     encoder_inputs = enc_inputs,
                                     decoder_inputs=dec_inputs,
                                     encoder_inputs_lengths=lengths)
        self.assertEqual(logits.shape, (batch_size, max_len-1, vocab_size))
        self.assertEqual(predictions.shape, (batch_size, max_len-1))

if __name__ == '__main__':
  absltest.main()