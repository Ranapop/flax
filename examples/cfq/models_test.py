from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp
from flax import nn

from models import Encoder

class ModelsTest(parameterized.TestCase):


    def test_encoder(self):
      seq1 = [1,0,3]
      seq2 = [1,0,3]
      BATCH_SIZE = 2
      HIDDEN_SIZE = 50
      inputs = jnp.array([seq1, seq2])
      lengths = jnp.array([len(seq1), len(seq2)])
      with nn.stochastic(jax.random.PRNGKey(0)):
        shared_embedding = nn.Embed.partial(
            num_embeddings=10,
            features=20,
            embedding_init=nn.initializers.normal(stddev=1.0))
        encoder = Encoder.partial(hidden_size=HIDDEN_SIZE)
        out, initial_params = encoder.init(nn.make_rng(),
                            inputs=inputs,
                            lengths=lengths,
                            shared_embedding=shared_embedding)
        c, h = out
        self.assertEqual(c.shape, (BATCH_SIZE, HIDDEN_SIZE))
        self.assertEqual(h.shape, (BATCH_SIZE, HIDDEN_SIZE))
       

if __name__ == '__main__':
  absltest.main()