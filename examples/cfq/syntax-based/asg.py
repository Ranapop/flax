import re
from typing import Tuple

Action = Tuple[int, str]
APPLY_RULE = 0
GENERATE_TOKEN = 1

def apply_rule_act(grammar, head, index):
  return (APPLY_RULE, grammar.get_rule_name_by_head(head,index))

def generate_act(token):
    return (GENERATE_TOKEN, token)

def var_rule(substring, grammar):
  "Receives something of the form <?x0>"
  action_sequence = [apply_rule_act(grammar, 'VAR',0)]
  match = re.match(r"\?x(\d+)", substring)
  if match:
      digit = match.groups()[0]
      action_sequence.append(generate_act(digit))
  else:
    raise Exception('var rule not matched')
  return action_sequence

def select_clause_rule(substring, grammar):
  "Receives either <DISTINCT var> or <count(*)> "
  match = re.match(r'DISTINCT (.*)', substring)
  if match:
    action_sequence = [apply_rule_act(grammar, 'select_clause', 0)]
    var = match.groups()[0]
    action_sequence += var_rule(var, grammar)
  elif re.match(r'count\(\*\)', substring):
    action_sequence = [apply_rule_act(grammar, 'select_clause',1)]
  else:
    raise Exception('select_clause rule not matched')
  return action_sequence

def var_token_rule(substring, grammar):
  if re.match(r"\?x(\d+)", substring):
    action_sequence = [apply_rule_act(grammar, 'var_token', 0)]
    action_sequence += var_rule(substring, grammar)
  else:
    action_sequence = [apply_rule_act(grammar, 'var_token', 1),
                       generate_act(substring)]
  return action_sequence
  
def where_entry_rule(substring, grammar):
  substring = substring.strip()
  match = re.match(r'FILTER \( (.*) \!= (.*) \)', substring)
  if match:
    action_sequence = [apply_rule_act(grammar, 'where_entry', 1),
                       apply_rule_act(grammar, 'filter_clause', 0)]
    (term1, term2) = match.groups()
    action_sequence += var_token_rule(term1, grammar)
    action_sequence += var_token_rule(term2, grammar)
  else:
    action_sequence = [apply_rule_act(grammar, 'where_entry', 0),
                       apply_rule_act(grammar, 'triples_block', 0)]
    terms = re.split('\s', substring)
    if len(terms)!=3:
      raise Exception('triples_block rule not matched', substring)
    action_sequence += var_token_rule(terms[0], grammar)
    action_sequence.append(generate_act(terms[1]))
    action_sequence += var_token_rule(terms[2], grammar)
  return action_sequence

def where_entries_rule(substrings, grammar):
  "Receives the where clauses (already separated)"
  no_clauses = len(substrings)
  if no_clauses==1:
    action_sequence = [apply_rule_act(grammar, 'where_entries', 0)]
    action_sequence += where_entry_rule(substrings[0], grammar)
  else:
    # Apply the 2nd branch of where_entries (the recursive branch).
    action_sequence = [apply_rule_act(grammar, 'where_entries',1)]
    action_sequence += where_entries_rule(substrings[0:-1], grammar)
    action_sequence += where_entry_rule(substrings[-1], grammar)
  return action_sequence

def where_multiple_rule(substrings, grammar):
  action_sequence = [apply_rule_act(grammar, 'where_clause', 0),
                     apply_rule_act(grammar, 'where_multiple', 0)]
  action_sequence += where_entries_rule(substrings[0:-1], grammar)
  action_sequence += where_entry_rule(substrings[-1], grammar)
  return action_sequence

def where_clause_rule(substring, grammar):
  "Receives the substring inside the brackets {}"
  where_clauses = re.split(r' \. ', substring)
  no_clauses = len(where_clauses)
  if no_clauses == 1:
    action_sequence = [apply_rule_act(grammar, 'where_clause', 1)]
    action_sequence += where_entry_rule(where_clauses[0], grammar)
  else:
    # multiple where clauses
    action_sequence = where_multiple_rule(where_clauses, grammar)
  return action_sequence


def query_rule(query, grammar):
  action_sequence = [apply_rule_act(grammar, 'query',0),
                     apply_rule_act(grammar,'select_query',0)]
  # Replce multiple spaces/new lines with simple space.
  query = re.sub(r'\n|\s+',' ', query)
  match = re.match(r'SELECT (.*) WHERE \{ (.*) \}', query)
  (select_clause_input, where_body) = match.groups()
  action_sequence += select_clause_rule(select_clause_input, grammar)
  action_sequence += where_clause_rule(where_body, grammar)
  return action_sequence


def generate_action_sequence(query, grammar):
  action_sequence = query_rule(query, grammar)
  return action_sequence
