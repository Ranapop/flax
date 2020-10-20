import re

APPLY_RULE = 0
GENERATE_TOKEN = 1


def apply_rule_act(grammar, head, index):
  return (APPLY_RULE, grammar.get_rule_name_by_head(head, index))


def generate_act(token):
    return (GENERATE_TOKEN, token)


def var_rule(substring, grammar):
  "Receives something of the form <?x0>"
  action_sequence = [apply_rule_act(grammar, 'VAR', 0)]
  match = re.match(r"\?x(\d+)", substring)
  if match:
      digit = match.groups()[0]
      action_sequence.append(generate_act(digit))
  else:
    raise Exception('var rule not matched')
  return action_sequence


def select_clause_rule(substring, grammar):
  "Receives either <DISTINCT var> or <count(*)> "
  match = re.match(r'DISTINCT \?x0', substring)
  if match:
    action_sequence = [apply_rule_act(grammar, 'select_clause', 0)]
  elif re.match(r'count\(\*\)', substring):
    action_sequence = [apply_rule_act(grammar, 'select_clause', 1)]
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


def where_clause_rule_rec(substrings, grammar):
  "Receives the list of where clauses."
  if len(substrings) == 1:
    action_sequence = [apply_rule_act(grammar, 'where_clause', 0)]
    action_sequence += where_entry_rule(substrings[0], grammar)
  else:
    action_sequence = [apply_rule_act(grammar, 'where_clause', 1)]
    action_sequence += where_clause_rule_rec(substrings[0:-1], grammar)
    action_sequence += where_entry_rule(substrings[-1], grammar)
  return action_sequence


def where_clause_rule(substring, grammar):
  "Receives the substring inside the brackets {}"
  where_clauses = re.split(r' \. ', substring)
  action_sequence = where_clause_rule_rec(where_clauses, grammar)
  return action_sequence


def query_rule(query, grammar):
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
