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
"""Module with unit tests for node.py
Should be run from cfq with:
python -m syntax_based.tests.node_test
"""
from absl.testing import absltest
from absl.testing import parameterized
from ..node import Node

def add_child_to_parent(parent: Node, child: Node):
    child.parent = parent
    parent.add_child(child)

class GrammarTest(parameterized.TestCase):

  def test_get_parent_time_steps(self):
    a = Node(None, 'a', 0)
    b = Node(None, 'b', 1)
    c = Node(None, 'c', 2)
    d = Node(None, 'd', 3)
    e = Node(None, 'e', 4)
    f = Node(None, 'f', 5)
    add_child_to_parent(a, b)
    add_child_to_parent(b, c)
    add_child_to_parent(c, d)
    add_child_to_parent(c, f)
    add_child_to_parent(d, e)
    generated_time_steps = a.get_parent_time_steps()
    expected_time_steps = [-1, 0, 1, 2, 3, 2]
    self.assertEqual(generated_time_steps, expected_time_steps)


if __name__ == '__main__':
  absltest.main()