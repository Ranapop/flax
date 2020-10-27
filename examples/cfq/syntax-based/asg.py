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
import re
from typing import Tuple

Action = Tuple[int, str]
APPLY_RULE = 0
GENERATE_TOKEN = 1


def apply_rule_act(grammar, head, index):
  return (APPLY_RULE, grammar.get_rule_name_by_head(head, index))


def generate_act(token):
    return (GENERATE_TOKEN, token)


def select_clause_rule(substring, grammar):
  """
  Rules:
    select_clause: "SELECT" "DISTINCT" "?x0"
                 | "SELECT" "count(*)"
  Args:
    substring: query substring to be matched by select_clause (without "SELECT")
    grammar: grammar object
  """
  match = re.match(r'DISTINCT \?x0', substring)
  if match:
    action_sequence = [apply_rule_act(grammar, 'select_clause', 0)]
  elif re.match(r'count\(\*\)', substring):
    action_sequence = [apply_rule_act(grammar, 'select_clause', 1)]
  else:
    raise Exception('select_clause rule not matched')
  return action_sequence


def var_token_rule(substring, grammar):
  """
  Rules:
    var_token: VAR
             | TOKEN
    TOKEN: /[^\s]+/
  Args:
    substring: query substring to be matched by select_clause (without "SELECT")
    grammar: grammar object
  """
  match = re.match(r"\?x(\d+)", substring)
  if match:
    # VAR branch
    digit = match.groups()[0]
    action_sequence = [apply_rule_act(grammar, 'var_token', 0)]
    action_sequence += [apply_rule_act(grammar, 'VAR', int(digit))]
  else:
    # TOKEN brancg
    action_sequence = [apply_rule_act(grammar, 'var_token', 1),
                       generate_act(substring)]
  return action_sequence
  

def where_entry_rule(substring, grammar):
  """
  Rules:
    where_entry: triples_block
               | filter_clause
    filter_clause: "FILTER" "(" var_token "!=" var_token ")"
    triples_block: var_token TOKEN var_token
  Args:
    substring: query substring to be matched by where_entry
    grammar: grammar object
  """
  substring = substring.strip()
  match = re.match(r'FILTER \( (.*) \!= (.*) \)', substring)
  if match:
    # filter_clause branch
    action_sequence = [apply_rule_act(grammar, 'where_entry', 1),
                       apply_rule_act(grammar, 'filter_clause', 0)]
    (term1, term2) = match.groups()
    action_sequence += var_token_rule(term1, grammar)
    action_sequence += var_token_rule(term2, grammar)
  else:
    # triples_block branch
    action_sequence = [apply_rule_act(grammar, 'where_entry', 0),
                       apply_rule_act(grammar, 'triples_block', 0)]
    terms = re.split('\s', substring)
    if len(terms) != 3:
      raise Exception('triples_block rule not matched', substring)
    action_sequence += var_token_rule(terms[0], grammar)
    action_sequence.append(generate_act(terms[1]))
    action_sequence += var_token_rule(terms[2], grammar)
  return action_sequence


def where_clause_rule_rec(substrings, grammar):
  """
  Rules:
    where_clause: where_entry
                | where_clause "." where_entry
  Args:
    substring: list of query substrings to be matched by the where_clause
    grammar: grammar object
  """
  if len(substrings) == 1:
    # where_entry branch
    action_sequence = [apply_rule_act(grammar, 'where_clause', 0)]
    action_sequence += where_entry_rule(substrings[0], grammar)
  else:
    # where_clause "." where_entry branch
    action_sequence = [apply_rule_act(grammar, 'where_clause', 1)]
    action_sequence += where_clause_rule_rec(substrings[0:-1], grammar)
    action_sequence += where_entry_rule(substrings[-1], grammar)
  return action_sequence


def where_clause_rule(substring, grammar):
  """
  Rules:
    where_clause: where_entry
                | where_clause "." where_entry
  Args:
    substring: query substring to be matched by one or more where_clause
    grammar: grammar object
  """
  where_clauses = re.split(r' \. ', substring)
  action_sequence = where_clause_rule_rec(where_clauses, grammar)
  return action_sequence


def query_rule(query, grammar):
  """
  Rules:
    query: select_query
    select_query: select_clause "WHERE" "{" where_clause "}"
  Args:
    query: query
    grammar: grammar object
  """
  action_sequence = [apply_rule_act(grammar, 'query', 0),
                     apply_rule_act(grammar,'select_query', 0)]
  # Replace multiple spaces/new lines with simple space.
  query = re.sub(r'\n|\s+',' ', query)
  match = re.match(r'SELECT (.*) WHERE \{ (.*) \}', query)
  (select_clause_input, where_body) = match.groups()
  action_sequence += select_clause_rule(select_clause_input, grammar)
  action_sequence += where_clause_rule(where_body, grammar)
  return action_sequence


def generate_action_sequence(query, grammar):
  action_sequence = query_rule(query, grammar)
  return action_sequence
