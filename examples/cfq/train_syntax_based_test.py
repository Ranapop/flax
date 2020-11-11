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
"""Module with test for train_syntax_based.py"""
from absl.testing import absltest
from absl.testing import parameterized
from jax import numpy as jnp
from train_syntax_based import cross_entropy_loss

class TrainTest(parameterized.TestCase):

  def test_cross_entropy_loss_padded(self):
    length = jnp.array([3])
    rule_vocab_size = 3
    token_vocab_size = 3
    # not padded data
    gold_seq = [1, 0, 2]
    rules_logits = [[1 ,2, 3], [0, 0, 0], [4, 2, 1]]
    tokens_logits = [[0, 0, 0], [2, 1, 3], [0, 0, 0]]
    gold_seq = jnp.array([gold_seq])
    rules_logits = jnp.array([rules_logits])
    tokens_logits = jnp.array([tokens_logits])
    # padded data
    padded_gold_seq = [1, 0, 2, 0, 0]
    padded_rules_logits = [[1, 2, 3], [0, 0, 0], [4, 2, 1], [0, 0, 0], [0, 0, 0]]
    padded_tokens_logits = [[0, 0, 0], [2, 1, 3], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    padded_gold_seq = jnp.array([padded_gold_seq])
    padded_rules_logits = jnp.array([padded_rules_logits])
    padded_tokens_logits = jnp.array([padded_tokens_logits])
    # compute losses
    loss = cross_entropy_loss(rules_logits, tokens_logits, gold_seq, length, rule_vocab_size, token_vocab_size)
    loss_for_padded = cross_entropy_loss(padded_rules_logits,
                                        padded_tokens_logits,
                                        padded_gold_seq,
                                        length, rule_vocab_size, token_vocab_size)
    print(loss_for_padded)
    self.assertEqual(loss, loss_for_padded)

if __name__ == '__main__':
  absltest.main()