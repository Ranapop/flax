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
"""Module with test for train.py"""
from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp


from examples.cfq.train import get_initial_params
from examples.cfq.train import mask_sequences
from examples.cfq.train import cross_entropy_loss
from examples.cfq.train import compute_perfect_match_accuracy
from examples.cfq.train import compute_metrics
from examples.cfq.models import Seq2seq

class TrainTest(parameterized.TestCase):


  def test_get_initial_params(self):
    vocab_size = 10
    rng = jax.random.PRNGKey(0)
    initial_params = get_initial_params(rng, vocab_size)
    self.assertNotEqual(initial_params, None)
    
  def test_mask_sequences(self):
    eos_id = 10
    sequences = jnp.array([
      [1, 2, eos_id, 3, 4]
    ])
    lengths = jnp.array([3])
    masked_sequences = mask_sequences(sequences, lengths)
    expected_sequences = jnp.array([
      [1, 2, eos_id, 0, 0]
    ])
    self.assertEqual((expected_sequences == masked_sequences).all(), True)

  def test_cross_entropy_loss(self):
    length = jnp.array([3])
    vocab_size = 3
    # not padded data
    gold_seq = [1, 0, 2]
    logits = [[1 ,2, 3], [2, 1, 3], [4, 2, 1]]
    gold_seq = jnp.array([gold_seq])
    logits = jnp.array([logits])
    # padded data
    padded_gold_seq = [1, 0, 2, 0, 0]
    padded_logits = [[1, 2, 3], [2, 1, 3], [4, 2, 1], [0, 0, 0], [0, 0, 0]]
    padded_gold_seq = jnp.array([padded_gold_seq])
    padded_logits = jnp.array([padded_logits])
    # compute losses
    loss = cross_entropy_loss(logits, gold_seq, length, vocab_size)
    loss_for_padded = cross_entropy_loss(padded_logits,
                                         padded_gold_seq,
                                         length, vocab_size)
    self.assertEqual(loss, loss_for_padded)

  def test_compute_perfect_match_accuracy_simple(self):
    eos_id = 10
    gold_seq = jnp.array([[1, 2 ,eos_id]])
    predicted_seq = jnp.array([[1, 2 ,eos_id]])
    length = jnp.array([3])
    acc = compute_perfect_match_accuracy(predicted_seq, gold_seq, length)
    expected_acc = jnp.array([1])
    self.assertEqual(acc, expected_acc)

  def test_compute_perfect_match_accuracy_larger_predicted(self):
    eos_id = 10
    gold_seq = jnp.array([[1, 2, eos_id, 0]])
    predicted_seq = jnp.array([[1, 2, eos_id, 10]])
    length = jnp.array([3])
    acc = compute_perfect_match_accuracy(predicted_seq, gold_seq, length)
    expected_acc = jnp.array([1])
    self.assertEqual(acc, expected_acc)

  def test_compute_metrics_accuracy_larger_predicted(self):
    eos_id = 10
    gold_seq = jnp.array([[1, 2, eos_id]])
    predicted_seq = jnp.array([[1, 2, eos_id, 10]])
    length = jnp.array([3])
    vocab_size = 100
    logits = jnp.zeros((1,4,vocab_size))
    metrics = compute_metrics(logits,
                              predicted_seq,
                              gold_seq,
                              length,
                              vocab_size)
    acc = metrics['accuracy']
    self.assertEqual(acc, 1)

  def test_compute_metrics_accuracy_larger_predicted_no_eos(self):
    eos_id = 10
    gold_seq = jnp.array([[1, 2, eos_id, 0]])
    predicted_seq = jnp.array([[1, 2, 7, 10]])
    length = jnp.array([3])
    vocab_size = 100
    logits = jnp.zeros((1,4,vocab_size))
    metrics = compute_metrics(logits,
                              predicted_seq,
                              gold_seq,
                              length,
                              vocab_size)
    acc = metrics['accuracy']
    self.assertEqual(acc, 0)


if __name__ == '__main__':
  absltest.main()