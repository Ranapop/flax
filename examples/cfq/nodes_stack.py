from typing import Tuple, List
import jax
import jax.numpy as jnp

def create_empty_stack(stack_capacity):
  stack_array = jnp.zeros((stack_capacity), dtype=jnp.int32)
  stack_pointer = 0
  return (stack_array, stack_pointer)

def push_to_stack(stack: Tuple[jnp.array, int], element: int):
  stack_array, stack_pointer = stack
  stack_capacity = stack_array.shape[0]
  assert stack_pointer + 1 <= stack_capacity
  stack_array = jax.ops.index_update(stack_array, stack_pointer, element)
  stack_pointer += 1
  return (stack_array, stack_pointer)

def push_elements_to_stack(stack: Tuple[jnp.array, int],
                           elements: Tuple[jnp.array, jnp.array]):
  elements_array, no_elements = elements
  no_elements_with_padding = elements_array.shape[0]
  stack_array, stack_pointer = stack
  stack_capacity = stack_array.shape[0]
  stack_array = jax.lax.dynamic_update_slice(
    stack_array, elements_array, [stack_pointer])
  stack_pointer += no_elements
  return (stack_array, stack_pointer)

def pop_element_from_stack(stack: Tuple[jnp.array, int]):
  """
  Returns popped_element, new stack. If the stack is empty, it returns
  zero (as this is what the sequence would have been padded it while training).
  """
  stack_array, stack_pointer = stack
  popped_element = jnp.where(stack_pointer <= 0,
                             jnp.array(0),
                             stack_array[stack_pointer-1])
  stack_pointer -= 1
  new_stack = (stack_array, stack_pointer)
  return popped_element, new_stack
