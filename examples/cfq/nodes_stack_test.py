from absl.testing import absltest
from absl.testing import parameterized

import jax.numpy as jnp

from nodes_stack import push_to_stack, push_elements_to_stack, \
  pop_element_from_stack

class NodesStackTest(parameterized.TestCase):

  def test_push_to_stack(self):

    stack_array = jnp.array([1, 2, 3, 0, 0, 0])
    stack_pointer = 3
    new_element = 4
    new_stack = push_to_stack((stack_array, stack_pointer), new_element)
    expected_stack_array = jnp.array([1, 2, 3, 4, 0, 0])
    expected_stack_pointer = 4
    self.assertEqual(jnp.array_equal(new_stack[0], expected_stack_array), True)
    self.assertEqual(new_stack[1], expected_stack_pointer)

  
  def test_push_to_empty_stack(self):

    stack_array = jnp.array([0, 0, 0, 0, 0, 0])
    stack_pointer = 0
    new_element = 4
    new_stack = push_to_stack((stack_array, stack_pointer), new_element)
    expected_stack_array = jnp.array([4, 0, 0, 0, 0, 0])
    expected_stack_pointer = 1
    self.assertEqual(jnp.array_equal(new_stack[0], expected_stack_array), True)
    self.assertEqual(new_stack[1], expected_stack_pointer)

  def test_push_to_full_stack(self):

    stack_array = jnp.array([1, 2, 3, 4, 5, 6])
    stack_pointer = 6
    new_element = 7
    error = None
    try:
      new_stack = push_to_stack((stack_array, stack_pointer), new_element)
    except AssertionError as e:
      error = e
    self.assertNotEqual(error, None)

  def test_push_elements_to_stack(self):

    stack_array = jnp.array([1, 2, 3, 0, 0, 0])
    stack_pointer = 3
    new_elements_arr = jnp.array([4, 5, 0])
    new_elements_length = jnp.array(2)
    new_elements = (new_elements_arr, new_elements_length)
    new_stack = push_elements_to_stack((stack_array, stack_pointer), new_elements)
    expected_stack_array = jnp.array([1, 2, 3, 4, 5, 0])
    expected_stack_pointer = 5
    self.assertEqual(jnp.array_equal(new_stack[0], expected_stack_array), True)
    self.assertEqual(new_stack[1], expected_stack_pointer)

  #TODO: think of the behaviour for full stack.
  def est_push_elements_to_full_stack(self):

    stack_array = jnp.array([1, 2, 3, 4, 5, 0])
    stack_pointer = 5
    new_elements_arr = jnp.array([6, 7])
    new_elements_length = jnp.array(2)
    new_elements = (new_elements_arr, new_elements_length)
    error = None
    try:
      new_stack = push_elements_to_stack(
        (stack_array, stack_pointer), new_elements)
    except AssertionError as e:
      error = e
    self.assertNotEqual(error, None)

  def test_pop_element_from_stack(self):
    stack_array = jnp.array([1, 2, 3, 4, 5, 0])
    stack_pointer = 5
    popped_element, new_stack = pop_element_from_stack((stack_array, stack_pointer))
    expected_stack_pointer = 4
    expected_popped_element = 5
    self.assertEqual(popped_element, expected_popped_element)
    self.assertEqual(jnp.array_equal(new_stack[0], stack_array), True)
    self.assertEqual(new_stack[1], expected_stack_pointer)

if __name__ == '__main__':
  absltest.main()