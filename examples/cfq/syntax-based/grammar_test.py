from absl.testing import absltest
from absl.testing import parameterized
from grammar import generate_grammar


class GrammarTest(parameterized.TestCase):

  def test_generate_grammar(self):
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
      VAR: "?x" DIGIT 
      DIGIT: /\d+/
      TOKEN: /[^\s]+/
    """
    grammar_dict = generate_grammar(grammar_str)
    expected_grammar_dict = {
      'r0': ('query', 'select_query'),
      'r1': ('select_query', 'select_clause "WHERE" "{" where_clause "}"'),
      'r2': ('select_clause', '"SELECT" "DISTINCT" "?x0"'),
      'r3': ('select_clause', '"SELECT" "count(*)"'),
      'r4': ('where_clause', 'where_entry'),
      'r5': ('where_clause', 'where_clause "." where_entry'),
      'r6': ('where_entry', 'triples_block'),
      'r7': ('where_entry', 'filter_clause'),
      'r8': ('filter_clause', '"FILTER" "(" var_token "!=" var_token ")"'),
      'r9': ('triples_block', 'var_token TOKEN var_token'),
      'r10': ('var_token', 'VAR'),
      'r11': ('var_token', 'TOKEN'),
      'r12': ('VAR', '"?x" DIGIT'),
      'r13': ('DIGIT', '/\\d+/'),
      'r14': ('TOKEN', '/[^\\s]+/')
    }
    self.assertEqual(grammar_dict, expected_grammar_dict)

if __name__ == '__main__':
  absltest.main()
