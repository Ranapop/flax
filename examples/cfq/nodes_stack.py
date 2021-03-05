from typing import Tuple, List
import jax
import jax.numpy as jnp

def push_to_stack(stack: Tuple[jnp.array, int], element: int):
  stack_array, stack_pointer = stack
  stack_capacity = stack_array.shape[0]
  # print('Stack pointer + 1 {} <= stack capacity {}'.format(stack_pointer+1, stack_capacity))
  assert stack_pointer + 1 <= stack_capacity
  stack_array = jax.ops.index_update(stack_array, stack_pointer, element)
  stack_pointer += 1
  return (stack_array, stack_pointer)

def push_elements_to_stack(stack: Tuple[jnp.array, int], elements: List[int]):
  no_elements = len(elements)
  stack_array, stack_pointer = stack
  stack_capacity = stack_array.shape[0]
  assert stack_pointer + no_elements <= stack_capacity
  indexes = jax.ops.index[stack_pointer:stack_pointer+no_elements]
  stack_array = jax.ops.index_update(stack_array, indexes, elements)
  stack_pointer += no_elements
  return (stack_array, stack_pointer)

def pop_element_from_stack(stack: Tuple[jnp.array, int]):
  """
  Returns popped_element, new stack.
  """
  stack_array, stack_pointer = stack
  assert stack_pointer != 0
  popped_element = stack_array[stack_pointer-1]
  stack_pointer -= 1
  new_stack = (stack_array, stack_pointer)
  return popped_element, new_stack
