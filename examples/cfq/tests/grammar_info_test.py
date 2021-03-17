from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp

from syntax_based.grammar import Grammar
from grammar_info import GrammarInfo

class GrammarInfoTest(parameterized.TestCase):

  def test_grammar_info_nodes_to_action_types(self):
    grammar_str = """
      query: select_query
      select_query: select_clause "WHERE" "{" where_clause "}"
      select_clause: "SELECT" "DISTINCT" "?x0"
                    | "SELECT" "count(*)"     
      where_clause: where_entry 
                    | where_clause "." where_entry
      where_entry: triples_block
                    | filter_clause
      filter_clause: "FILTER" "(" var_token "!=" var_token ")"
      triples_block: var_token TOKEN var_token
      var_token: VAR
               | TOKEN 
      VAR: "?x0" 
         | "?x1" 
         | "?x2" 
         | "?x3" 
         | "?x4" 
         | "?x5" 
      TOKEN: /[^\s]+/
    """
    grammar = Grammar(grammar_str)
    grammar_info = GrammarInfo(grammar)
    expected_node_vocab = {
      'query': 0, 'select_query': 1, 'select_clause':2,
      'where_clause':3, 'where_entry': 4, 'triples_block': 5,
      'filter_clause': 6,
      'var_token': 7, 'TOKEN':8, 'VAR':9,
      '[^\\s]+': 10}
    expected_nodes_to_action_types = [
      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.
    ]
    expected_grammar_entry = 0
    expected_nodes_to_action_types = jnp.array(expected_nodes_to_action_types)
    self.assertEqual(grammar_info.node_vocab, expected_node_vocab)
    self.assertTrue(jnp.array_equal(grammar_info.nodes_to_action_types,
                                    expected_nodes_to_action_types))
    self.assertEqual(grammar_info.grammar_entry, expected_grammar_entry)

  def test_get_expanded_nodes_array(self):
    grammar_str = """
      query: select_query
      select_query: select_clause "WHERE" "{" where_clause "}"
      select_clause: "SELECT" "DISTINCT" "?x0"
                    | "SELECT" "count(*)"     
      where_clause: where_entry 
                    | where_clause "." where_entry
      where_entry: triples_block
                    | filter_clause
      filter_clause: "FILTER" "(" var_token "!=" var_token ")"
      triples_block: var_token TOKEN var_token
      var_token: VAR
               | TOKEN 
      VAR: "?x0" 
         | "?x1" 
         | "?x2" 
         | "?x3" 
         | "?x4" 
         | "?x5" 
      TOKEN: /[^\s]+/
    """
    grammar = Grammar(grammar_str)
    grammar_info = GrammarInfo(grammar)
    expanded_nodes_list = [
      [1],
      [2, 3],
      [],
      [],
      [4],
      [3, 4],
      [5],
      [6],
      [7, 7],
      [7, 8, 7],
      [9],
      [8],
      [],
      [],
      [],
      [],
      [],
      [],
      [10],
      []
    ]
    node_expansions, lengths = grammar_info.get_expanded_nodes_array(
      expanded_nodes_list)
    expected_expanded_nodes = [
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
      [0, 0, 0]]
    expected_expanded_nodes = jnp.array(expected_expanded_nodes)
    expected_lengths = [
      1, 2, 0, 0, 1, 2, 1, 1, 2, 3, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0
    ]
    expected_lengths = jnp.array(expected_lengths)
    self.assertTrue(jnp.array_equal(node_expansions, expected_expanded_nodes))
    self.assertTrue(jnp.array_equal(lengths, expected_lengths))

if __name__ == '__main__':
  absltest.main()