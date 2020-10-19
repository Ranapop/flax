from absl.testing import absltest
from absl.testing import parameterized
from asg import apply_rule_act, generate_act, generate_action_sequence
from grammar import Grammar, GRAMMAR_STR

class AsgTest(parameterized.TestCase):


  def test_generate_action_sequence_muliple(self):
    grammar = Grammar(GRAMMAR_STR)
    expected_action_sequence = [
      apply_rule_act(grammar, 'query', 0),
      apply_rule_act(grammar, 'select_query', 0),
      apply_rule_act(grammar, 'select_clause', 0),
      apply_rule_act(grammar, 'VAR', 0),
      generate_act('0'),
      apply_rule_act(grammar, 'where_clause', 0),
      apply_rule_act(grammar, 'where_multiple', 0),
      apply_rule_act(grammar, 'where_entries', 1),
      apply_rule_act(grammar, 'where_entries', 0),
      apply_rule_act(grammar, 'where_entry', 0),
      apply_rule_act(grammar, 'triples_block', 0),
      apply_rule_act(grammar, 'var_token', 0),
      apply_rule_act(grammar, 'VAR', 0),
      generate_act('0'),
      generate_act('a'),
      apply_rule_act(grammar, 'var_token', 1),
      generate_act('people.person'),
      apply_rule_act(grammar, 'where_entry', 0),
      apply_rule_act(grammar, 'triples_block', 0),
      apply_rule_act(grammar, 'var_token', 0),
      apply_rule_act(grammar, 'VAR', 0),
      generate_act('0'),
      generate_act('influence.influencenode.influencedby'),
      apply_rule_act(grammar, 'var_token', 0),
      apply_rule_act(grammar, 'VAR', 0),
      generate_act('1'),
      apply_rule_act(grammar, 'where_entry', 0),
      apply_rule_act(grammar, 'triples_block', 0),
      apply_rule_act(grammar, 'var_token', 0),
      apply_rule_act(grammar, 'VAR', 0),
      generate_act('1'),
      generate_act('film.actor.filmnsfilm.performance.character'),
      apply_rule_act(grammar, 'var_token', 1),
      generate_act('M1'),
    ]
    query = """SELECT DISTINCT ?x0 WHERE {
      ?x0 a people.person .
      ?x0 influence.influencenode.influencedby ?x1 .
      ?x1 film.actor.filmnsfilm.performance.character M1 }"""
    generated_action_sequence = generate_action_sequence(query, grammar)
    self.assertEqual(generated_action_sequence, expected_action_sequence)


if __name__ == '__main__':
  absltest.main()