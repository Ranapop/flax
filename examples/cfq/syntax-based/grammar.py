import re
import collections

GRAMMAR_STR = """
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

def generate_sub_rules(rule):
  sub_rules = []
  match = re.match(r'\s*(.*):\s*(.*)\s*', rule)
  if match:
    (head, body) = match.groups()
    branches = re.split(r'\s*\|\s*', body)
    for branch in branches:
      sub_rules.append((head, branch.strip()))
  return sub_rules


def generate_grammar(grammar_str):
  grammar_str = grammar_str.strip()
  split_lines = re.split('\n', grammar_str)
  rules = []
  current_rule = ""
  for line in split_lines:
    if re.match('\s+\|', line):
      current_rule += line
    else:
      rules.append(current_rule)
      current_rule = line
  if current_rule not in rules:
    rules.append(current_rule)
  sub_rules = []
  for rule in rules:
    sub_rules += generate_sub_rules(rule)

  sub_rules_dict = {f'r{i}': sub_rules[i] for i in range(len(sub_rules))}
  return sub_rules_dict

def generate_rules_by_head(sub_rules_dict):
  result = collections.defaultdict(list)
  for rule_name, (head, body) in sub_rules_dict.items():
    result[head].append((rule_name, body))
  return result

class Grammar:

  def __init__(self, grammar_str):
    self.sub_rules = generate_grammar(grammar_str)
    self.rules_by_head = generate_rules_by_head(self.sub_rules)

  def get_rule_by_head(self, head, index):
    head_rules = self.rules_by_head[head]
    return head_rules[index]

  def get_rule_name_by_head(self, head, index):
    return self.get_rule_by_head(head, index)[0]

if __name__ == "__main__":
  grammar = Grammar(GRAMMAR_STR)
  sub_rules = grammar.sub_rules
  print(sub_rules)
  for rule, branches in grammar.rules_by_head.items():
    print(rule)
    print(branches)