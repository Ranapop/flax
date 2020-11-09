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
"""Flax modules composing the seq2seq LSTM architecture for CFQ"""
from typing import Tuple, List, Dict
import numpy as np
import jax
from jax import random
import jax.numpy as jnp
from flax import jax_utils
from flax import nn
from syntax_based.node import Node, construct_root, apply_action
from syntax_based.grammar import Grammar

# hyperparams
LSTM_HIDDEN_SIZE = 512
EMBEDDING_DIM = 512
ATTENTION_SIZE = 512
ATTENTION_LAYER_SIZE = 512
NUM_LAYERS = 2
HORIZONTAL_DROPOUT = 0
VERTICAL_DROPOUT = 0.4
EMBED_DROPOUT = 0
ATTENTION_DROPOUT = 0

"""Common modules"""
class MlpAttention(nn.Module):
  """MLP attention module that returns a scalar score for each key."""

  def apply(self,
            query: jnp.ndarray,
            projected_keys: jnp.ndarray,
            values: jnp.ndarray,
            mask: jnp.ndarray,
            hidden_size: int = None,
            use_batch_axis: bool = True) -> jnp.ndarray:
    """Computes MLP-based attention based on keys and a query.

    Attention scores are computed by feeding the keys and query through an MLP.
    This results in a single scalar per key, and for each sequence the attention
    scores are normalized using a softmax so that they sum to 1. Invalid key
    positions are ignored as indicated by the mask. This is also called
    "Bahdanau attention" and was originally proposed in:
    ```
    Bahdanau et al., 2015. Neural Machine Translation by Jointly Learning to
    Align and Translate. ICLR. https://arxiv.org/abs/1409.0473
    ```

    Args:
      query: The query with which to compute attention. Shape:
        <float32>[batch_size, 1, query_size].
      projected_keys: The inputs multiplied with a weight matrix. Shape:
        <float32>[batch_size, seq_length, attention_size].
      values: The values to be weighted [batch_size, seq_length, values_size]
      mask: A mask that determines which values in `projected_keys` are valid.
        Only values for which the mask is True will get non-zero attention
        scores. <bool>[batch_size, seq_length].
      hidden_size: The hidden size of the MLP that computes the attention score.
      use_batch_axis: When True the code is executed at batch level, otherwise
        at example level (in that case the arguments don't have the batch_size
        dimensions, for example query would be [1, query_size]).

    Returns:
      The weighted values (context vector) [batch_size, seq_len]
    """
    projected_query = nn.Dense(query, hidden_size, name='query', bias=False)
    # Query broadcasts in the sum below along the time (seq_length) dimension.
    energy = nn.tanh(projected_keys + projected_query)
    scores = nn.Dense(energy, 1, name='energy', bias=False)
    scores = scores.squeeze(-1)  # New shape: <float32>[batch_size, seq_len].
    # TODO: see if I can rewrite this to not have the squeezing and unsqueezing
    scores = jnp.where(mask, scores, -jnp.inf)  # Using exp(-inf) = 0 below.
    scores = nn.softmax(scores, axis=-1)

    if use_batch_axis:
      # Shape: <float32>[batch_size, seq_len]
      context = jnp.sum(jnp.expand_dims(scores, 2) * values, axis=1)
    else:
      # Shape: <float32>[seq_len]
      context = jnp.sum(jnp.expand_dims(scores, 1) * values, axis=0)
    return context, scores  


class MultilayerLSTM(nn.Module):
  "LSTM cell with multiple layers"

  def apply(self,
            num_layers: int,
            horizontal_dropout_masks: jnp.array,
            vertical_dropout_rate: float,
            input: jnp.array, previous_states: List,
            train: bool):
    """
    Args
      num_layers: number of layers
      horizontal_dropout_masks: dropout masks for each layer (the same dropout
        mask is used at each time step and applied on the hidden state that is
        fed into the cell to the right). This is the recurrent dropout used in
        https://arxiv.org/abs/1512.05287. [num_layers, batch_size, hidden_size]
      vertical_dropout_rate: dropout rate between layers, yet the dropout mask
        is not reused across timestamps. It's only applied if the number of
        layers > 1.
      input: input given to the first LSTM layer [batch_size, input_size]
      previous_states: list of (c,h) for each layer
        shape [num_layers, batch_size, 2*hidden_size]
      train: boolean indicating training or inference flow.
    """
    states = []
    final_output = None
    for layer_idx in range(num_layers):
      lstm_name = 'lstm_layer' + str(layer_idx)
      cell = nn.LSTMCell.partial(name=lstm_name)
      c, h = previous_states[layer_idx]
      # Apply dropout to h.
      if horizontal_dropout_masks[layer_idx] is not None:
        h = h * horizontal_dropout_masks[layer_idx]
      # Apply dropout to the hidden state from lower layer.
      if layer_idx != 0 and vertical_dropout_rate > 0:
        input = nn.dropout(input,
                           rate=vertical_dropout_rate,
                           deterministic=train)
      state, output = cell((c, h), input)
      states.append(state)
      input = output
      final_output = output
    return states, final_output


class Encoder(nn.Module):
  """LSTM encoder, returning state after EOS is input."""

  def apply(self, inputs: jnp.array, lengths: jnp.array,
            shared_embedding: nn.Module, train: bool, hidden_size: int,
            num_layers: int, horizontal_dropout_rate: float,
            vertical_dropout_rate: float,
            embed_dropout_rate: float = EMBED_DROPOUT):

    inputs = shared_embedding(inputs)
    inputs = nn.dropout(inputs, rate=embed_dropout_rate, deterministic=train)
    lstm = nn.LSTM.partial(
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout_rate=vertical_dropout_rate,
        recurrent_dropout_rate=horizontal_dropout_rate).shared(name='lstm')
    outputs, final_states = lstm(inputs, lengths, train=train)
    return outputs, final_states


def compute_attention_masks(mask_shape: Tuple, lengths: jnp.array):
  mask = np.ones(mask_shape, dtype=bool)
  mask = mask * (lengths[:, jnp.newaxis] > jnp.arange(mask.shape[1]))
  return mask


"""Baseline modules."""
class Decoder(nn.Module):
  """LSTM decoder."""

  @staticmethod
  def create_dropout_masks(num_masks: int, shape: Tuple,
                           dropout_rate: float, train: bool):
    if not train or dropout_rate == 0:
      return [None] * num_masks
    masks = []
    for i in range(0, num_masks):
      dropout_mask = random.bernoulli(nn.make_rng(),
                                      p=1 - dropout_rate,
                                      shape=shape)
      # Convert array of boolean values to probabilty distribution.
      dropout_mask = dropout_mask / (1.0 - dropout_rate)
      masks.append(dropout_mask)
    return masks

  def apply(self,
            init_states,
            encoder_hidden_states,
            attention_mask,
            inputs: jnp.array,
            shared_embedding: nn.Module,
            vocab_size: int,
            num_layers: int,
            horizontal_dropout_rate: float,
            vertical_dropout_rate: float,
            embed_dropout_rate: float = EMBED_DROPOUT,
            attention_layer_dropout: float = ATTENTION_DROPOUT,
            train: bool = False):
    """
    The decoder follows Luong's decoder in how attention is used (the current
    decoder state is used for attention computation, and the attention vector is
    used to get the final probability distribution instead of the RNN output).
    The attention mechanism employed is Bahdanau (also called concat/mlp).
    
    Args
      init_states: states to initialize the decoder hidden states (coming from
        the encoder). This assumes the encoder and decoder have the same number
        of layers. [num_layers, 2, batch_size, hidden_size]
      encoder_hidden_states: encoder hidden states
        [batch_size, input_seq_len, hidden_size]
      attention_mask: attention mask [batch_size, input_seq_len]
      inputs: on the train flow (train=True) the gold decoded sequence (as
        teacher forcing is used), on the inference flow an array of desired
        output length where the first token is the BOS token
      shared_embedding: module for computing the embeddings (shared with
        the encoder)
      vocab_size: vocabulary size
      num_layers: number of layers in the LSTM
      horizontal_dropout_rate: dropout applied at the same layer (same mask
        across time steps).
      vertical_dropout_rate: dropout applied between layers.
      embed_dropout_rate: dropout applied on the embeddings.
      attention_layer_dropout: dropout applied on the attention layer input (on
        the concatenation of current state and context vector)
      train: boolean choosing from train and inference flow
    """
    multilayer_lstm_cell = MultilayerLSTM.partial(num_layers=num_layers).shared(
        name='multilayer_lstm')
    attention_layer = nn.Dense.shared(features=ATTENTION_LAYER_SIZE,
                                      name='attention_layer')
    projection = nn.Dense.shared(features=vocab_size, name='projection')
    mlp_attention = MlpAttention.partial(hidden_size=ATTENTION_SIZE).shared(
        name='attention')
    # The keys projection can be calculated once for the whole sequence.
    projected_keys = nn.Dense(encoder_hidden_states,
                              ATTENTION_SIZE,
                              name='keys',
                              bias=False)

    batch_size = encoder_hidden_states.shape[0]
    hidden_size = encoder_hidden_states.shape[-1]
    h_dropout_masks = Decoder.create_dropout_masks(
        num_masks=num_layers,
        shape=(batch_size, hidden_size),
        dropout_rate=horizontal_dropout_rate,
        train=train)

    def decode_step_fn(carry, x):
      rng, previous_states, last_prediction, prev_attention = carry
      carry_rng, categorical_rng = jax.random.split(rng, 2)
      if not train:
        x = last_prediction
      x = shared_embedding(x)
      x = nn.dropout(x, rate=embed_dropout_rate, deterministic=train)
      lstm_input = jnp.concatenate([x, prev_attention], axis=-1)
      states, h = multilayer_lstm_cell(
        horizontal_dropout_masks=h_dropout_masks,
        vertical_dropout_rate=vertical_dropout_rate,
        input=lstm_input,
        previous_states=previous_states,
        train=train)
      context, scores = mlp_attention(jnp.expand_dims(h, 1), projected_keys,
                                      encoder_hidden_states, attention_mask)
      context_and_state = jnp.concatenate([context, h], axis=-1)
      context_and_state = nn.dropout(context_and_state,
                                     rate=attention_layer_dropout,
                                     deterministic=train)
      attention = jnp.tanh(attention_layer(context_and_state))
      logits = projection(attention)
      predicted_tokens = jax.random.categorical(categorical_rng, logits)
      predicted_tokens_uint8 = jnp.asarray(predicted_tokens, dtype=jnp.uint8)
      new_carry = (carry_rng, states, predicted_tokens_uint8, attention)
      new_x = (logits, predicted_tokens_uint8, scores)
      return new_carry, new_x

    # initialisig the LSTM states and final output with the
    # encoder hidden states
    attention = jnp.zeros((batch_size, ATTENTION_LAYER_SIZE))
    init_carry = (nn.make_rng(), init_states, inputs[:, 0], attention)

    if self.is_initializing():
      # initialize parameters before scan
      decode_step_fn(init_carry, inputs[:, 0])

    _, (logits, predictions, scores) = jax_utils.scan_in_dim(
        decode_step_fn,
        init=init_carry,  # rng, lstm_state, last_pred
        xs=inputs,
        axis=1)
    # The attention weights are only examined on the evaluation flow, so this
    # if is used to avoid unnecesary operations.
    if not self.is_initializing() and not train:
      attention_weights = jnp.array(scores)
      # Going from [output_seq_len, batch_size, input_seq_len]
      # to [batch_size, output_seq_len, input_seq_len].
      jnp.swapaxes(attention_weights, 1, 2)
    else:
      attention_weights = None
    return logits, predictions, attention_weights


class Seq2seq(nn.Module):
  """Sequence-to-sequence class using encoder/decoder architecture."""

  def apply(self,
            encoder_inputs: jnp.array,
            decoder_inputs: jnp.array,
            encoder_inputs_lengths: jnp.array,
            vocab_size: int,
            emb_dim: int = EMBEDDING_DIM,
            train: bool = True,
            hidden_size: int = LSTM_HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            horizontal_dropout_rate=HORIZONTAL_DROPOUT,
            vertical_dropout_rate=VERTICAL_DROPOUT):
    """Run the seq2seq model.

    Args:
      encoder_inputs: padded batch of input sequences to encode (as vocab 
      indices), shaped `[batch_size, question length]`.
      decoder_inputs: padded batch of expected decoded sequences on the train
        flow, shaped `[batch_size, query len]` on the train flow and shaped
        `[batch_size, max query len]` on the inference flow.
        During inference (i.e., `train = False`), the initial time step
        is forced into the model and samples are used for the following inputs.
        The second dimension of this tensor determines how many steps will be
        decoded, regardless of the value of `teacher_force`.
      encoder_inputs_lengths: input sequences lengths [batch_size]
      vocab_size: size of vocabulary
      emb_dim: embedding dimension
      train: bool, differentiating between train and inference flow. This is
        needed for the dropout on both the encoder and decoder, and on the
        decoder it also activates teacher forcing (the gold sequence used).
      hidden_size: int, the number of hidden dimensions in the encoder and
        decoder LSTMs.
      Returns:
        Array of decoded logits.
      """
    shared_embedding = nn.Embed.shared(
        num_embeddings=vocab_size,
        features=emb_dim,
        embedding_init=nn.initializers.normal(stddev=1.0))

    encoder = Encoder.partial(hidden_size=hidden_size,
                              num_layers=num_layers,
                              horizontal_dropout_rate=horizontal_dropout_rate,
                              vertical_dropout_rate=vertical_dropout_rate)
    decoder = Decoder.partial(num_layers=num_layers,
                              horizontal_dropout_rate=horizontal_dropout_rate,
                              vertical_dropout_rate=vertical_dropout_rate)
    # compute attention masks
    mask = compute_attention_masks(encoder_inputs.shape, encoder_inputs_lengths)

    # Encode inputs
    hidden_states, init_decoder_states = encoder(encoder_inputs,
                                                 encoder_inputs_lengths,
                                                 shared_embedding, train)
    # Decode outputs.
    logits, predictions, attention_weights = decoder(init_decoder_states,
                                                     hidden_states,
                                                     mask,
                                                     decoder_inputs[:, :-1],
                                                     shared_embedding,
                                                     vocab_size,
                                                     train=train)

    return logits, predictions, attention_weights


"""Syntax based modules."""
# Token / rule embedding size.
ACTION_EMBEDDING_SIZE = 50
NODE_EMBEDDING_SIZE = 20

class ActionEmbed(nn.Module):

  def apply(self,
            action: jnp.array,
            token_embedding: nn.Module,
            rule_vocab_size: int):
    rule_embedding = nn.Embed.partial(
      num_embeddings=rule_vocab_size,
      features=ACTION_EMBEDDING_SIZE,
      embedding_init=nn.initializers.normal(stddev=1.0))

    action_type = action[0]
    action_value = action[1]
    rule_emb = rule_embedding(action_value)
    token_emb = token_embedding(action_value)
    # Return the rule_emb in case the action_type is 0, token_emb in case the
    # action type is 1 and a zero-valued vector otherwise.
    action_emb = jnp.equal(action_type, jnp.array(0)) * rule_emb + \
     jnp.equal(action_type, jnp.array(1)) * token_emb
    return action_emb


class SyntaxDecoderLSTMInput(nn.Module):
  
  def apply(self,
            current_node_type: int,
            previous_action: jnp.array,
            parent_state: jnp.array,
            parent_action: Tuple[int, int],
            context: jnp.array,
            token_embedding: nn.Module,
            rule_vocab_size: int,
            node_vocab_size: int):
    node_embedding = nn.Embed.partial(
        name='node_embedding',
        num_embeddings=node_vocab_size,
        features=NODE_EMBEDDING_SIZE,
        embedding_init=nn.initializers.normal(stddev=1.0))
    action_embedding = ActionEmbed.shared(name='action_embedding',
                                          token_embedding=token_embedding,
                                          rule_vocab_size=rule_vocab_size)
    prev_act_emb = action_embedding(previous_action)
    parent_act_emb = action_embedding(parent_action)
    parent_info = jnp.concatenate([parent_act_emb, parent_state])
    node_emb = node_embedding(current_node_type)

    lstm_input = jnp.concatenate([prev_act_emb, context, parent_info, node_emb])
    return lstm_input

class SyntaxBasedDecoderClassifier(nn.Module):

  def apply(self,
            curr_action_type: jnp.array,
            h: jnp.array,
            rule_vocab_size,
            token_vocab_size):
    rule_projection = nn.Dense.shared(features=rule_vocab_size,
                                       name='rule_projection')
    token_projection = nn.Dense.shared(features=token_vocab_size,
                                       name='token_projection')
    rules_logits = jax.nn.softmax(rule_projection(h))
    rules_logits = jnp.equal(curr_action_type, jnp.array(0)) * rules_logits
    tokens_logits = jax.nn.softmax(token_projection(h))
    token_logits = jnp.equal(curr_action_type, jnp.array(1)) * tokens_logits
    prediction = \
      jnp.equal(curr_action_type, jnp.array(0)) * jnp.argmax(rules_logits) + \
      jnp.equal(curr_action_type, jnp.array(1)) * jnp.argmax(token_logits)
    return rules_logits, tokens_logits, prediction

class SyntaxBasedDecoder(nn.Module):
  """LSTM syntax-based decoder."""

  @staticmethod
  def create_dropout_masks(num_masks: int, shape: Tuple,
                           dropout_rate: float, train: bool):
    if not train or dropout_rate == 0:
      return [None] * num_masks
    masks = []
    for i in range(0, num_masks):
      dropout_mask = random.bernoulli(nn.make_rng(),
                                      p=1 - dropout_rate,
                                      shape=shape)
      # Convert array of boolean values to probabilty distribution.
      dropout_mask = dropout_mask / (1.0 - dropout_rate)
      masks.append(dropout_mask)
    return masks

  def decode_train(self,
                   init_states: jnp.array,
                   encoder_hidden_states: jnp.array,
                   attention_mask: jnp.array,
                   inputs: jnp.array,
                   lstm_input_module: nn.Module,
                   multilayer_lstm_cell: nn.Module,
                   mlp_attention: nn.Module,
                   classifier: nn.Module,
                   projected_keys: jnp.array,
                   h_dropout_masks: jnp.array,
                   vertical_dropout_rate: float,
                   initial_lstm_state: jnp.array):

    action_types = inputs[0]
    action_values = inputs[1]
    node_types = inputs[2]
    parent_steps = inputs[3]

    time_steps = action_types.shape[0]
    initial_h = init_states[-1, 1, :]
    multilayer_lstm_output = (init_states, initial_h)
    carry = (nn.make_rng(), multilayer_lstm_output)
    all_predictions = []
    all_rules_logits = []
    all_tokens_logits = []
    lstm_states = [initial_lstm_state]
    actions = [jnp.array([-1, -1])]

    
    for i in range(time_steps):
      rng, multilayer_lstm_output = carry
      previous_states, h = multilayer_lstm_output
      carry_rng, categorical_rng = jax.random.split(rng, 2)
       
      dec_prev_state = jnp.expand_dims(h, 0)
      context, _ = mlp_attention(dec_prev_state, projected_keys,
                                 encoder_hidden_states, attention_mask)
      parent_idx = parent_steps[i] + 1 # parent -1 => index 0
      lstm_states_arr = jnp.array(lstm_states)
      actions_arr = jnp.array(actions)

      lstm_input = lstm_input_module(
        current_node_type=node_types[i],
        previous_action=actions_arr[i],
        parent_state=lstm_states_arr[parent_idx],
        parent_action=actions_arr[parent_idx],
        context=context)

      states, h = multilayer_lstm_cell(
        horizontal_dropout_masks=h_dropout_masks,
        vertical_dropout_rate=vertical_dropout_rate,
        input=lstm_input,
        previous_states=previous_states,
        train=True)
    
      curr_action_type = action_types[i]
      rules_logits, tokens_logits, prediction = classifier(curr_action_type, h)

      carry = (carry_rng, (states, h))
      actions.append(jnp.array([action_types[i], action_values[i]]))
      lstm_states.append(h)

      all_predictions.append(prediction)
      all_rules_logits.append(rules_logits)
      all_tokens_logits.append(tokens_logits)

    all_predictions = jnp.array(all_predictions)
    all_rules_logits = jnp.array(all_rules_logits)
    all_token_logits = jnp.array(all_tokens_logits)

    return all_rules_logits, all_token_logits, all_predictions, None
    
  def decode_inference(self,
                       init_states: jnp.array,
                       encoder_hidden_states: jnp.array,
                       attention_mask: jnp.array,
                       lstm_input_module: nn.Module,
                       multilayer_lstm_cell: nn.Module,
                       mlp_attention: nn.Module,
                       classifier: nn.Module,
                       projected_keys: jnp.array,
                       h_dropout_masks: jnp.array,
                       vertical_dropout_rate: float,
                       initial_lstm_state: jnp.array,
                       node_vocab: Dict,
                       tokens_list: List,
                       grammar: Grammar):
    
    initial_h = init_states[-1, 1, :]
    multilayer_lstm_output = (init_states, initial_h)
    carry = (nn.make_rng(), multilayer_lstm_output)
    all_predictions = []
    all_rules_logits = []
    all_tokens_logits = []
    lstm_states = [initial_lstm_state]
    actions = [jnp.array([-1, -1])]
    frontier_nodes = []
    time_step=0
    while len(frontier_nodes)!= 0:
      current_node: Node = frontier_nodes.pop()

      rng, multilayer_lstm_output = carry
      previous_states, h = multilayer_lstm_output
      carry_rng, categorical_rng = jax.random.split(rng, 2)

      dec_prev_state = jnp.expand_dims(h, 0)
      context, _ = mlp_attention(dec_prev_state, projected_keys,
                                 encoder_hidden_states, attention_mask)

      parent_node = current_node.parent
      if parent_node is None:
        parent_idx = 0
      else:
        parent_idx = parent_node.time_step + 1
      
      lstm_states_arr = jnp.array(lstm_states)
      actions_arr = jnp.array(actions)

      node_type = current_node.value()
      node_type_id = node_vocab[node_type]
      lstm_input = lstm_input_module(
        current_node_type=node_type_id,
        previous_action=actions_arr[time_step],
        parent_state=lstm_states_arr[parent_idx],
        parent_action=actions_arr[parent_idx],
        context=context)

      states, h = multilayer_lstm_cell(
        horizontal_dropout_masks=h_dropout_masks,
        vertical_dropout_rate=vertical_dropout_rate,
        input=lstm_input,
        previous_states=previous_states,
        train=True)
      if grammar.is_rule_head(node_type):
        curr_action_type = jnp.array(0)
      else:
        curr_action_type = jnp.array(1)
      rules_logits, tokens_logits, prediction = classifier(curr_action_type, h)

      # apply predicted action
      if grammar.is_rule_head(node_type):
        # Apply rule action
        action = (0, prediction)
      else:
        # TODO: change generate action to take token id
        token = tokens_list[prediction]
        action = (1, token)
      frontier_nodes = apply_action(frontier_nodes, action, time_step, grammar)

      carry = (carry_rng, (states, h))
      lstm_states.append(h)

      all_predictions.append(prediction)
      all_rules_logits.append(rules_logits)
      all_tokens_logits.append(tokens_logits)
      time_step+=1

    all_predictions = jnp.array(all_predictions)
    all_rules_logits = jnp.array(all_rules_logits)
    all_token_logits = jnp.array(all_tokens_logits)

    return all_rules_logits, all_token_logits, all_predictions, None
      

      

  def apply(self,
            init_states: jnp.array,
            encoder_hidden_states: jnp.array,
            attention_mask: jnp.array,
            inputs: jnp.array,
            token_embedding: nn.Module,
            token_vocab_size: int,
            rule_vocab_size: int,
            node_vocab_size: int,
            num_layers: int,
            horizontal_dropout_rate: float,
            vertical_dropout_rate: float,
            train: bool = False):
    """
    Args:
      init_states: initial state (c,h) for each layer [num_layers, 2, hidden_size].
      encoder_hidden_states: h vector for each input token
        [input_seq_len, hidden_size].
      attention_mask: attention mask [input_seq_len].
      inputs: decoder inputs comprising of 4 vectors: action_types, action_values,
        node_types, parent_steps [4, output_seq_len].
      token_embedding: token embedding module.
      token_vocab_size: token vocab size.
      rule_vocab_size: rule vocab size.
      node_vocab_size: node vocab size.
      num_layers: number of LSTM layers.
      horizontal_dropout_rate: LSTM horizontal dropout rate.
      horizontal_dropout_rate: LSTM vertical dropout rate.
      train: flag distinguishing between train and test flow.
    """
    lstm_input_module = SyntaxDecoderLSTMInput.partial(
      token_embedding = token_embedding,
      rule_vocab_size = rule_vocab_size,
      node_vocab_size = node_vocab_size).shared(
        name = 'lstm_input_module')
    multilayer_lstm_cell = MultilayerLSTM.partial(num_layers=num_layers).shared(
      name='multilayer_lstm')
    mlp_attention = MlpAttention.partial(hidden_size=ATTENTION_SIZE,
                                         use_batch_axis=False
                                         ).shared(name='attention')
    # The keys projection can be calculated once for the whole sequence.
    projected_keys = nn.Dense(encoder_hidden_states,
                              ATTENTION_SIZE,
                              name='keys',
                              bias=False)
    classifier = SyntaxBasedDecoderClassifier.shared(
      name='decoder_classifier',
      rule_vocab_size=rule_vocab_size,
      token_vocab_size=token_vocab_size)

    hidden_size = encoder_hidden_states.shape[-1]
    h_dropout_masks = Decoder.create_dropout_masks(
        num_masks=num_layers,
        shape=(hidden_size),
        dropout_rate=horizontal_dropout_rate,
        train=train)
    initial_lstm_state = jnp.zeros(hidden_size)

    if train:
      rules_logits, \
      token_logits, \
      predictions, \
      attention_weights = self.decode_train(init_states,
                   encoder_hidden_states,
                   attention_mask,
                   inputs,
                   lstm_input_module,
                   multilayer_lstm_cell,
                   mlp_attention,
                   classifier,
                   projected_keys,
                   h_dropout_masks,
                   vertical_dropout_rate,
                   initial_lstm_state)

    #TODO: move if code in decode_inference
    if not self.is_initializing() and not train:
      # Going [input_seq_len, output_seq_len] -> [output_seq_len, input_seq_len].
      jnp.swapaxes(attention_weights, 0, 1)
    return rules_logits, token_logits, predictions, attention_weights


class Seq2tree(nn.Module):
  """Sequence-to-ast class following Yin and Neubig."""

  def apply(self,
            encoder_inputs: jnp.array,
            decoder_inputs: jnp.array,
            encoder_inputs_lengths: jnp.array,
            token_vocab_size: int,
            rule_vocab_size: int,
            node_vocab_size: int,
            train: bool = True,
            hidden_size: int = LSTM_HIDDEN_SIZE,
            num_layers: int = NUM_LAYERS,
            horizontal_dropout_rate=HORIZONTAL_DROPOUT,
            vertical_dropout_rate=VERTICAL_DROPOUT):
    token_embedding = nn.Embed.shared(
        num_embeddings=token_vocab_size,
        features=ACTION_EMBEDDING_SIZE,
        embedding_init=nn.initializers.normal(stddev=1.0))

    encoder = Encoder.partial(shared_embedding=token_embedding,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              horizontal_dropout_rate=horizontal_dropout_rate,
                              vertical_dropout_rate=vertical_dropout_rate)
    decoder = SyntaxBasedDecoder.partial(
                                   token_embedding = token_embedding,
                                   num_layers=num_layers,
                                   horizontal_dropout_rate=horizontal_dropout_rate,
                                   vertical_dropout_rate=vertical_dropout_rate,
                                   train = train,
                                   token_vocab_size = token_vocab_size,
                                   rule_vocab_size = rule_vocab_size,
                                   node_vocab_size = node_vocab_size)
    vmapped_decoder = jax.vmap(decoder)
    # compute attention masks
    mask = compute_attention_masks(encoder_inputs.shape, encoder_inputs_lengths)

    # Encode inputs
    hidden_states, init_decoder_states = encoder(encoder_inputs,
                                                 encoder_inputs_lengths,
                                                 train=train)
    # [no_layers, 2, batch, hidden_size] -> [batch, no_layers, 2, hidden_size]
    init_decoder_states = jnp.array(init_decoder_states)
    init_decoder_states = jnp.swapaxes(init_decoder_states, 0, 2)
    # inputs_no_bos = decoder_inputs[:, :-1]
    # Decode outputs.
    rules_logits, tokens_logits, predictions, attention_weights = \
      vmapped_decoder(init_decoder_states,
                      hidden_states,
                      mask,
                      decoder_inputs)

    return rules_logits, tokens_logits, predictions, attention_weights

