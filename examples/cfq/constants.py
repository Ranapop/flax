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
"""Constants module"""
import tensorflow.compat.v2 as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

BOS = b'<bos>'
EOS = b'<eos>'
UNK = b'<unk>'
PAD = b'<pad>'

QUESTION_KEY = 'question'
QUERY_KEY = 'query'
QUESTION_LEN_KEY = 'question_len'
QUERY_LEN_KEY = 'query_len'
# Constants for Seq2Tree
ACTION_TYPES_KEY = 'action_types'
ACTION_VALUES_KEY = 'action_values'
PARENT_STEPS_KEY = 'parent_steps'
ACTION_SEQ_LEN_KEY = 'action_seq_len'
