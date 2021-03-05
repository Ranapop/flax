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
"""Module for grammar definition and parsing."""
import re
import collections
from typing import Dict, List
from enum import Enum

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
  VAR: "?x0" | "?x1" | "?x2" | "?x3" | "?x4" | "?x5"
  TOKEN: /[^\s]+/
"""

def remove_multiple_spaces(s: str):
  return ' '.join(s.strip().split())

def extract_sub_rules_from_rules(rule: str):
  sub_rules = []
  rule = remove_multiple_spaces(rule)
  match = re.match(r'(.*): (.*)', rule)
  if match:
    (head, body) = match.groups()
    branches = re.split(r' \| ', body)
    for branch in branches:
      sub_rules.append((head, branch.strip()))
  return sub_rules


def extract_sub_rules(grammar_str: str):
  """
  Method for extracting the sub rules (head -> branch) from a grammar string.
  The current logic assumes that each rule branch is on a separate line.
  For the following grammar:
  a -> b | c
  b -> d
  c -> "C"
  d -> "D"
  it would generate a list of tuples (head,branch):
  [('a','b'), ('a','c'), ('b', 'd'), ('c','"C"'), ('d','"D"')]
  """
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
    sub_rules += extract_sub_rules_from_rules(rule)
  return sub_rules

class TermType(Enum):
  SYNTAX_TERM = 0
  REGEX_TERM = 1
  RULE_TERM = 2

class Term:
  """
  Class describing a term (token on the right side of a rule). It contains a
  type and a value. If the term is a syntax token, eg. "SELECT", the value
  contains the string "SELECT", if the term is a regex term, it also contains a
  string, for eg. "/[^\s]+/", and if the term is a rule head in another rule,
  the type is RULE_TERM and the token is stored in the value field of Term.
  """

  def __init__(self, term_type: TermType, value: str):
    self.term_type = term_type
    self.value = value

  def __repr__(self):
    return self.value

  def __eq__(self, other):
    """Overrides the default implementation."""
    return self.term_type == other.term_type and self.value == other.value

class RuleBranch:
  """Class describing a branch of a rule. For example, for the rule
   a -> b c "smth" | d e, there are two branches, 'b c "smth"' and 'd e'. Each
   branch can be interpreted as having the body formed of a list of terms, in
   this example [b, c, "smth"] and [d, e] respectively."""

  def __init__(self, id: int, body_str: str):
    """
    Args:
      id: the rule id (index in the list of rule branches of the grammar).
      body_str: the branch body (eg. 'b c "smth"', will be tokenized and
                transformed in a list of Term instances in this constructor).
    """
    self.branch_id = id
    self.body = RuleBranch.construct_from_string(body_str)

  @staticmethod
  def construct_from_string(body: str):
    rule_terms = []
    body_tokens = body.split()
    for body_token in body_tokens:
      match = re.match(r'\"(.*)\"', body_token)
      if match:
        # Syntax token matched.
        syntax_token = match.groups()[0]
        term = Term(TermType.SYNTAX_TERM, syntax_token)
      else:
        match = re.match(r'\/(.*)\/', body_token)
        if match:
          # Regex token matched.
          regex_token = match.groups()[0]
          term = Term(TermType.REGEX_TERM, regex_token)
        else:
          # Rule token => store body token.
          term = Term(TermType.RULE_TERM, body_token)
      rule_terms.append(term)
    return rule_terms

  def __eq__(self, other):
    """Overrides the default implementation."""
    return self.branch_id == other.branch_id and self.body == other.body

  def __repr__(self):
    string_terms = [str(term) for term in self.body]
    string_body = ' '.join(string_terms)
    return string_body

class Grammar:

  def __init__(self, grammar_str: str):
    sub_rules = extract_sub_rules(grammar_str)
    self.branches = []
    self.rules = collections.defaultdict(list)
    for branch_id in range(len(sub_rules)):
      sub_rule = sub_rules[branch_id]
      head = sub_rule[0]
      body = sub_rule[1]
      rule_branch = RuleBranch(id=branch_id, body_str=body)
      self.branches.append(rule_branch)
      self.rules[head].append(branch_id)
    first_rule = sub_rules[0]
    self.grammar_entry = first_rule[0]

  def collect_syntax_tokens(self):
    syntax_tokens = set()
    for branch in self.branches:
       for term in branch.body:
         if term.term_type == TermType.SYNTAX_TERM:
           syntax_tokens.add(term.value)
    return list(syntax_tokens)
  
  def collect_node_types(self):
    """Collects the node types (the values that will be stored in the frontier
    nodes. These can be RULE_TERMs or REGEX_TERMs.
    
    The method also returns a list of flags, speciffying for each node type
    if it's a rule node (1) or not (0)."""
    node_types = [self.grammar_entry]
    node_flags = [1]
    for branch in self.branches:
      for term in branch.body:
        if term.term_type in [TermType.RULE_TERM, TermType.REGEX_TERM]:
          if term.value not in node_types:
            node_types.append(term.value)
            if term.term_type == TermType.RULE_TERM:
              node_flags.append(1)
            else:
              node_flags.append(0)
    return list(node_types), node_flags

  def get_expanded_nodes(self, nodes_vocab: List[str]):
    """
    Returns a list of lists showing how nodes can be expanded once a branch is
    predicted, that is for each branch a list of nodes the node it would be
    expanded with. The function gets a vocab for the nodes so the lists are
    already numericalized.
    """
    expanded_nodes = []
    for branch in self.branches:
      body = branch.body
      branch_nodes = []
      for term in body:
        if term.term_type in [TermType.RULE_TERM, TermType.REGEX_TERM]:
          node_idx = nodes_vocab[term.value]
          branch_nodes.append(node_idx)
      expanded_nodes.append(branch_nodes)
    return expanded_nodes

  def get_branch_id_by_head_and_index(self, head: str, index: int):
    """Returns the branch id given the head and index."""
    head_rules = self.rules[head]
    return head_rules[index]

  def get_head_for_branch(self, branch_id):
    """Returns head for branch. Not optimal, but should only be needed for
    debugging purposes."""
    for head, branches in self.rules.items():
      for branch in branches:
        if branch == branch_id:
          return head
    return None

  def print_grammar(self):
    for head, branch_ids in self.rules.items():
      for branch_id in branch_ids:
        right_side = ''
        branch = self.branches[branch_id]
        for term in branch.body:
          term_str = ' (' + term.value + ', ' + str(term.term_type) + ') '
          right_side += term_str
        print(head, ' -> ',term_str)