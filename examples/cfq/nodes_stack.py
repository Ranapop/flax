from typing import Tuple, List
import jax
import jax.numpy as jnp
from grammar_info import GrammarInfo

def create_empty_stack(stack_capacity):
  stack_array = jnp.zeros((stack_capacity), dtype=jnp.int32)
  stack_pointer = jnp.array(0, dtype=jnp.int32)
  return (stack_array, stack_pointer)

def push_to_stack(stack: Tuple[jnp.array, int], element: int):
  stack_array, stack_pointer = stack
  stack_capacity = stack_array.shape[0]
  #TODO: error handling with nans.
  # assert stack_pointer + 1 <= stack_capacity
  stack_array = jax.ops.index_update(stack_array, stack_pointer, element)
  stack_pointer += jnp.array(1, dtype=jnp.int32)
  return (stack_array, stack_pointer)

def push_elements_to_stack(stack: Tuple[jnp.array, int],
                           elements: Tuple[jnp.array, jnp.array]):
  elements_array, no_elements = elements
  no_elements_with_padding = elements_array.shape[0]
  stack_array, stack_pointer = stack
  stack_capacity = stack_array.shape[0]
  stack_array = jax.lax.dynamic_update_slice(
    stack_array, elements_array, [stack_pointer])
  stack_pointer += jnp.array(no_elements, dtype=jnp.int32)
  return (stack_array, stack_pointer)

def pop_element_from_stack(stack: Tuple[jnp.array, int]):
  """
  Returns popped_element, new stack. If the stack is empty, it returns
  zero (as this is what the sequence would have been padded it while training).
  """
  stack_array, stack_pointer = stack
  stack_top = stack_array[stack_pointer-1]
  popped_element = jnp.where(stack_pointer <= 0,
                             jnp.array(0),
                             stack_top)
  stack_pointer -= 1
  new_stack = (stack_array, stack_pointer)
  return popped_element, new_stack

def apply_action_to_stack(stack: Tuple[jnp.array, int],
                          action: Tuple[jnp.array, jnp.array],
                          grammar_info: GrammarInfo):
  action_type, action_value = action
  # ApplyRule -> 0, GenToken -> 1.
  idx = jnp.where(
      action_type, jnp.array(grammar_info.rule_vocab_size), action_value)
  expanded_nodes_arr, expanded_nodes_lengths = grammar_info.expanded_nodes
  new_nodes = (expanded_nodes_arr[idx], expanded_nodes_lengths[idx])
  new_stack = push_elements_to_stack(stack, new_nodes)
  return new_stack