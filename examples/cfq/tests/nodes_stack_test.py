from absl.testing import absltest
from absl.testing import parameterized

import jax.numpy as jnp

from cfq.nodes_stack import push_to_stack, push_elements_to_stack, \
  pop_element_from_stack, create_empty_stack, apply_action_to_stack

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

  #TODO: Add error propagation with nans and fix this test.
  def test_push_to_full_stack(self):

    stack_array = jnp.array([1, 2, 3, 4, 5, 6])
    stack_pointer = 6
    new_element = 7
    error = None
    try:
      new_stack = push_to_stack((stack_array, stack_pointer), new_element)
    except AssertionError as e:
      error = e
    self.assertEqual(error, None)

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

  class MockGrammarInfo():

    def __init__(self):
      self.node_vocab = {
        'query': 0, 'select_query': 1, 'select_clause':2,
        'where_clause':3, 'where_entry': 4, 'triples_block': 5,
        'filter_clause': 6,
        'var_token': 7, 'TOKEN':8, 'VAR':9,
        '[^\\s]+': 10}
      self.nodes_to_action_types = jnp.array([
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.])
      self.grammar_entry = 0
      expanded_nodes_array = jnp.array([
        [1, 0, 0],
        [3, 2, 0],
        [0, 0, 0],
        [0, 0, 0],
        [4, 0, 0],
        [4, 3, 0],
        [5, 0, 0],
        [6, 0, 0],
        [7, 7, 0],
        [7, 8, 7],
        [9, 0, 0],
        [8, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [10, 0, 0],
        [0, 0, 0]])
      expanded_nodes_lengths = jnp.array([
        1, 2, 0, 0, 1, 2, 1, 1, 2, 3, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0])
      self.expanded_nodes = (expanded_nodes_array, expanded_nodes_lengths)
      self.rule_vocab_size = 19

  def test_stack_for_action_sequence(self):
    tokens_vocab = {'a': 0, 'people.person': 1}
    actions = [
      (0, 0), # query -> select_query
      (0, 1), # select_query -> select_clause "WHERE" "{" where_clause "}"
      (0, 2), # select_clause -> "SELECT" "DISTINCT" "?x0"
      (0, 4), # where_clause -> where_entry
      (0, 6), # where_entry -> triples_block
      (0, 9), # triples_block -> var_token TOKEN var_token
      (0, 10), # var_token -> VAR
      (0, 12), # VAR -> "?x0"
      (0, 18), # TOKEN -> /[^\s]+/
      (1, 0), # GenerateToken 'a'
      (0, 11), # var_token -> TOKEN
      (0, 18), # TOKEN -> /[^\s]+/
      (1, 1) # GenerateToken 'people.person'
    ]
    actions = jnp.array(actions)
    action_seq_len = len(actions)
    max_node_expansion_size = 3
    grammar_info = self.MockGrammarInfo()
    
    # Initial stack: grammar_entry
    frontier = create_empty_stack(action_seq_len * max_node_expansion_size)
    frontier = push_to_stack(frontier, grammar_info.grammar_entry)
    expected_frontier_list = [0 for i in range(3 * action_seq_len)]
    expected_frontier_list[0] = 0 # grammar entry.
    expected_frontier_nodes = jnp.array(expected_frontier_list)
    self.assertTrue(jnp.array_equal(frontier[0], expected_frontier_nodes))
    self.assertTrue(jnp.array_equal(frontier[1], jnp.array(1)))

    # Step 0: ApplyRule query -> select_query
    current_node, frontier = pop_element_from_stack(frontier)
    expected_node = jnp.array(0) # 'query'
    self.assertTrue(jnp.array_equal(current_node, expected_node))
    frontier = apply_action_to_stack(frontier, actions[0], grammar_info)
    # expected_frontier_nodes = ['select_query'] -> [1]
    expected_frontier_list[0] = 1 # select_query
    expected_frontier_nodes = jnp.array(expected_frontier_list)
    self.assertTrue(jnp.array_equal(frontier[0], expected_frontier_nodes))
    self.assertTrue(jnp.array_equal(frontier[1], jnp.array(1)))

    # Step 1: ApplyRule select_query -> select_clause "WHERE" "{" where_clause "}"
    current_node, frontier = pop_element_from_stack(frontier)
    expected_node = jnp.array(1) # 'select_query'
    self.assertTrue(jnp.array_equal(current_node, expected_node))
    frontier = apply_action_to_stack(frontier, actions[1], grammar_info)
    # expected_frontier_nodes = ['where_clause', 'select_clause'] -> [3, 2]
    expected_frontier_list[0] = 3 # where_clause
    expected_frontier_list[1] = 2 # select_clause
    expected_frontier_nodes = jnp.array(expected_frontier_list)
    self.assertTrue(jnp.array_equal(frontier[0], expected_frontier_nodes))
    self.assertTrue(jnp.array_equal(frontier[1], jnp.array(2)))

    # Step 2: ApplyRule select_clause -> "SELECT" "DISTINCT" "?x0"
    current_node, frontier = pop_element_from_stack(frontier)
    expected_node = jnp.array(2) # 'select_clause'
    self.assertTrue(jnp.array_equal(current_node, expected_node))
    frontier = apply_action_to_stack(frontier, actions[2], grammar_info)
    # expected_frontier_nodes = ['where_clause'] -> [3]
    expected_frontier_list[0] = 3 # where_clause
    expected_frontier_list[1] = 0 # no element
    expected_frontier_nodes = jnp.array(expected_frontier_list)
    self.assertTrue(jnp.array_equal(frontier[0], expected_frontier_nodes))
    self.assertTrue(jnp.array_equal(frontier[1], jnp.array(1)))

    # Step 3: ApplyRule where_clause -> where_entry
    current_node, frontier = pop_element_from_stack(frontier)
    expected_node = jnp.array(3) # 'where_clause'
    self.assertTrue(jnp.array_equal(current_node, expected_node))
    frontier = apply_action_to_stack(frontier, actions[3], grammar_info)
    # expected_frontier_nodes = ['where_entry'] -> [4]
    expected_frontier_list[0] = 4 # where_entry
    expected_frontier_nodes = jnp.array(expected_frontier_list)
    self.assertTrue(jnp.array_equal(frontier[0], expected_frontier_nodes))
    self.assertTrue(jnp.array_equal(frontier[1], jnp.array(1)))

    # Step 4: ApplyRule where_entry -> triples_block
    current_node, frontier = pop_element_from_stack(frontier)
    expected_node = jnp.array(4) # 'where_entry'
    self.assertTrue(jnp.array_equal(current_node, expected_node))
    frontier = apply_action_to_stack(frontier, actions[4], grammar_info)
    # expected_frontier_nodes = ['triples_block'] -> [5]
    expected_frontier_list[0] = 5 # triples_block
    expected_frontier_nodes = jnp.array(expected_frontier_list)
    self.assertTrue(jnp.array_equal(frontier[0], expected_frontier_nodes))
    self.assertTrue(jnp.array_equal(frontier[1], jnp.array(1)))

    # Step 5: ApplyRule triples_block -> var_token TOKEN var_token
    current_node, frontier = pop_element_from_stack(frontier)
    expected_node = jnp.array(5) # 'triples_block'
    self.assertTrue(jnp.array_equal(current_node, expected_node))
    frontier = apply_action_to_stack(frontier, actions[5], grammar_info)
    # expected_frontier_nodes = ['var_token', 'TOKEN', 'var_token'] -> [7, 8, 7]
    expected_frontier_list[0] = 7 # var_token
    expected_frontier_list[1] = 8 # TOKEN
    expected_frontier_list[2] = 7 # var_token
    expected_frontier_nodes = jnp.array(expected_frontier_list)
    self.assertTrue(jnp.array_equal(frontier[0], expected_frontier_nodes))
    self.assertTrue(jnp.array_equal(frontier[1], jnp.array(3)))

    # Step 6: ApplyRule var_token -> VAR
    current_node, frontier = pop_element_from_stack(frontier)
    expected_node = jnp.array(7) # 'var_token'
    self.assertTrue(jnp.array_equal(current_node, expected_node))
    frontier = apply_action_to_stack(frontier, actions[6], grammar_info)
    # expected_frontier_nodes = ['var_token', 'TOKEN', VAR] -> [7, 8, 9]
    expected_frontier_list[0] = 7 # var_token
    expected_frontier_list[1] = 8 # TOKEN
    expected_frontier_list[2] = 9 # VAR
    expected_frontier_nodes = jnp.array(expected_frontier_list)
    self.assertTrue(jnp.array_equal(frontier[0], expected_frontier_nodes))
    self.assertTrue(jnp.array_equal(frontier[1], jnp.array(3)))

    # Step 7: ApplyRule VAR -> "?x0"
    current_node, frontier = pop_element_from_stack(frontier)
    expected_node = jnp.array(9) # 'VAR'
    self.assertTrue(jnp.array_equal(current_node, expected_node))
    frontier = apply_action_to_stack(frontier, actions[7], grammar_info)
    # expected_frontier_nodes = ['var_token', 'TOKEN'] -> [7, 8]
    expected_frontier_list[0] = 7 # var_token
    expected_frontier_list[1] = 8 # TOKEN
    expected_frontier_list[2] = 0 # no element
    expected_frontier_nodes = jnp.array(expected_frontier_list)
    self.assertTrue(jnp.array_equal(frontier[0], expected_frontier_nodes))
    self.assertTrue(jnp.array_equal(frontier[1], jnp.array(2)))

    # Step 8: ApplyRule TOKEN -> /[^\s]+/
    current_node, frontier = pop_element_from_stack(frontier)
    expected_node = jnp.array(8) # 'TOKEN'
    self.assertTrue(jnp.array_equal(current_node, expected_node))
    frontier = apply_action_to_stack(frontier, actions[8], grammar_info)
    # expected_frontier_nodes = ['var_token', '[^\s]+'] -> [7, 10]
    expected_frontier_list[0] = 7 # var_token
    expected_frontier_list[1] = 10 # [^\s]+
    expected_frontier_nodes = jnp.array(expected_frontier_list)
    self.assertTrue(jnp.array_equal(frontier[0], expected_frontier_nodes))
    self.assertTrue(jnp.array_equal(frontier[1], jnp.array(2)))

    # Step 9: GenerateToken 'a'
    current_node, frontier = pop_element_from_stack(frontier)
    expected_node = jnp.array(10) # '[^\s]+'
    self.assertTrue(jnp.array_equal(current_node, expected_node))
    frontier = apply_action_to_stack(frontier, actions[9], grammar_info)
    # expected_frontier_nodes = ['var_token'] -> [7]
    expected_frontier_list[0] = 7 # var_token
    expected_frontier_list[1] = 0 # no element
    expected_frontier_nodes = jnp.array(expected_frontier_list)
    self.assertTrue(jnp.array_equal(frontier[0], expected_frontier_nodes))
    self.assertTrue(jnp.array_equal(frontier[1], jnp.array(1)))

    # Step 10: ApplyRule var_token -> TOKEN
    current_node, frontier = pop_element_from_stack(frontier)
    expected_node = jnp.array(7) # 'var_token'
    self.assertTrue(jnp.array_equal(current_node, expected_node))
    frontier = apply_action_to_stack(frontier, actions[10], grammar_info)
    # expected_frontier_nodes = ['TOKEN'] -> [8]
    expected_frontier_list[0] = 8 # TOKEN
    expected_frontier_nodes = jnp.array(expected_frontier_list)
    self.assertTrue(jnp.array_equal(frontier[0], expected_frontier_nodes))
    self.assertTrue(jnp.array_equal(frontier[1], jnp.array(1)))

    # Step 11: ApplyRule TOKEN -> /[^\s]+/
    current_node, frontier = pop_element_from_stack(frontier)
    expected_node = jnp.array(8) # 'TOKEN'
    self.assertTrue(jnp.array_equal(current_node, expected_node))
    frontier = apply_action_to_stack(frontier, actions[11], grammar_info)
    # expected_frontier_nodes = ['[^\s]+'] -> [10]
    expected_frontier_list[0] = 10 # [^\s]+
    expected_frontier_nodes = jnp.array(expected_frontier_list)
    self.assertTrue(jnp.array_equal(frontier[0], expected_frontier_nodes))
    self.assertTrue(jnp.array_equal(frontier[1], jnp.array(1)))

    # Step 12: GenerateToken 'people.person'
    current_node, frontier = pop_element_from_stack(frontier)
    expected_node = jnp.array(10) # '[^\s]+'
    self.assertTrue(jnp.array_equal(current_node, expected_node))
    frontier = apply_action_to_stack(frontier, actions[12], grammar_info)
    # expected_frontier_nodes = [] -> []
    expected_frontier_list[0] = 0 # no element
    expected_frontier_nodes = jnp.array(expected_frontier_list)
    self.assertTrue(jnp.array_equal(frontier[0], expected_frontier_nodes))
    self.assertTrue(jnp.array_equal(frontier[1], jnp.array(0)))


if __name__ == '__main__':
  absltest.main()