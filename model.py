import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell

import code

def _linear(args, output_size, bias, bias_start=0.0, scope=None):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: (optional) Variable scope to create parameters in.
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape() for a in args]
  for shape in shapes:
    if shape.ndims != 2:
      raise ValueError("linear is expecting 2D arguments: %s" % shapes)
    if shape[1].value is None:
      raise ValueError("linear expects shape[1] to be provided for shape %s, "
                       "but saw %d" % (shape, shape[1]))
    else:
      total_arg_size += shape[1].value

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  scope = vs.get_variable_scope()
  with vs.variable_scope(scope) as outer_scope:
    weights = vs.get_variable(
        "weights", [total_arg_size, output_size], dtype=dtype)
    if len(args) == 1:
      res = tf.matmul(args[0], weights)
    else:
      res = tf.matmul(tf.concat(1, args), weights)
    if not bias:
      return res
    with vs.variable_scope(outer_scope) as inner_scope:
      inner_scope.set_partitioner(None)
      biases = vs.get_variable(
          "biases", [output_size],
          dtype=dtype,
          initializer=tf.constant_initializer(bias_start, dtype=dtype))
  return res + biases


class HyperCell(rnn_cell.RNNCell):

  def __init__(self, num_units, context_embed, mikolov_adapt=False, hyper_adapt=False):
    self._num_units = num_units
    self._forget_bias = 1.0
    self._activation = tf.tanh
    self.context_embed = context_embed
    self.mikolov_adapt = mikolov_adapt
    self.hyper_adapt = hyper_adapt

    with vs.variable_scope('hyper_lstm_cell'):
      if self.hyper_adapt:
        self.adaptation_weights = tf.get_variable(
          'adaptation_weights', 
          [context_embed.get_shape()[1].value, 4 * self._num_units])
        self.adaptation_bias = tf.get_variable(
          'adaptation_bias', [4 * self._num_units],
          initializer=tf.constant_initializer(np.ones(4 * self._num_units)))

      if self.mikolov_adapt:
        self.biases = tf.get_variable(
          'mikolov_biases', 
          [context_embed.get_shape()[1].value, 4 * self._num_units])

  @property
  def state_size(self):
    return rnn_cell.LSTMStateTuple(self._num_units, self._num_units)
    
  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None, reuse=None):
    """Long short-term memory cell (LSTM)."""

    with vs.variable_scope("hyper_lstm_cell", reuse=reuse):
      # Parameters of gates are concatenated into one multiply for efficiency.
      c, h = state
      adapted = _linear([inputs, h], 4 * self._num_units, True, scope=scope)

      if self.hyper_adapt:
        adaptation_coeff = tf.nn.relu(
          tf.matmul(self.context_embed, self.adaptation_weights) 
          + self.adaptation_bias)
        adapted = tf.mul(adaptation_coeff, adapted)
        
      if self.mikolov_adapt:
        delta = tf.matmul(self.context_embed, self.biases)
        adapted += delta

      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      i, j, f, o = tf.split(1, 4, adapted)

      new_c = (c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) *
               self._activation(j))
      new_h = self._activation(new_c) * tf.sigmoid(o)

      new_state = rnn_cell.LSTMStateTuple(new_c, new_h)

      return new_h, new_state


class BaseModel(object):
  """Hold the code that is shared between all model varients."""

  def __init__(self, params, unigram_probs, context_vocab_sizes=None):
    self.unigram_probs = unigram_probs
    self.max_length = params.max_len
    self.vocab_size = len(unigram_probs)
    self.num_context_vars = len(context_vocab_sizes)
    self.x = tf.placeholder(tf.int32, [params.batch_size, self.max_length], name='x')
    self.y = tf.placeholder(tf.int64, [params.batch_size, self.max_length], name='y')
    self.seq_len = tf.placeholder(tf.int64, [params.batch_size], name='seq_len')


    enable_context_embeds = (params.use_mikolov_adaptation or params.use_hyper_adaptation 
                             or params.use_softmax_adaptation)
    if enable_context_embeds:

      self._embedding_dims = params.embedding_dims
      self.context_placeholders = {}
      self.context_embeddings = {}
      for i in range(self.num_context_vars):
        self.context_placeholders[params.context_vars[i]] = tf.placeholder(
          tf.int64, [None], name='context_var{0}'.format(i))
        self.context_embeddings[params.context_vars[i]] = tf.get_variable(
          'context_embedding{0}'.format(i), 
          [context_vocab_sizes[i], params.context_embed_sizes[i]])

      context_embeds = []
      for context_var in params.context_vars:
        context_embeds.append(tf.nn.embedding_lookup(
          self.context_embeddings[context_var], self.context_placeholders[context_var]))

      if len(context_embeds) == 1:
        self.final_context_embed = context_embeds[0]
      else:
        context_embeds = tf.concat(1, context_embeds)

        context_mlp = tf.get_variable(
          'context_mlp', [sum(params.context_embed_sizes), params.context_embed_size])
        context_bias = tf.get_variable('context_bias', [params.context_embed_size])
        
        self.final_context_embed = tf.nn.tanh(tf.matmul(context_embeds, context_mlp) + 
                                              context_bias)
      
    if params.use_softmax_adaptation:
      self._word_embeddings = tf.get_variable(
        'word_embeddings', [self.vocab_size, self._embedding_dims +
                            params.context_embed_size])
    else:
      self._word_embeddings = tf.get_variable(
        'word_embeddings', [self.vocab_size, self._embedding_dims])

    self._inputs = tf.nn.embedding_lookup(self._word_embeddings, self.x)

    if params.use_softmax_adaptation:  # chop off the context embedding part
      self._inputs = self._inputs[:, :, params.context_embed_size:]

    self.base_bias = tf.get_variable('base_bias', [self.vocab_size])

    self.dropout_keep_prob = tf.placeholder_with_default(1.0, (), name='keep_prob')
    
    # make a mask
    indicator = tf.sequence_mask(tf.to_int32(self.seq_len - 1), self.max_length)
    sz = [params.batch_size, self.max_length]
    self._mask = tf.select(indicator, tf.ones(sz), tf.zeros(sz))

  def OutputHelper(self, outputs, linear_proj, params, use_nce_loss=True):
    reshaped_outputs = tf.reshape(outputs, [-1, outputs.get_shape()[-1].value])
    projected = tf.matmul(reshaped_outputs, linear_proj)

    proj_out =  tf.reshape(projected, [outputs.get_shape()[0].value,
                                       outputs.get_shape()[1].value, -1])

    if use_nce_loss:
      losses = self.DoNCE(proj_out, self._word_embeddings, num_sampled=params.nce_samples)
      masked_loss = tf.mul(losses, self._mask)
    else:
      masked_loss = self.ComputeLoss(proj_out, self._word_embeddings)
      self.masked_loss = masked_loss

    self.cost = tf.reduce_sum(masked_loss) / tf.reduce_sum(self._mask)    


  def DoNCE(self, weights, out_embeddings, num_sampled=256, user_embeddings=None):
    w_list = tf.unpack(weights, axis=1)
    losses = []
    for w, y in zip(tf.unpack(weights, axis=1), tf.split(1, self.max_length, self.y)):
      sampled_values = tf.nn.fixed_unigram_candidate_sampler(
        true_classes=y,
        num_true=1,
        num_sampled=num_sampled,
        unique=True,
        range_max=self.vocab_size,
        unigrams=self.unigram_probs
      )

      self.a, self.b, self.c = sampled_values

      if user_embeddings is not None:
        w = tf.concat(1, [user_embeddings, w])

      nce_loss = tf.nn.sampled_softmax_loss(out_embeddings, self.base_bias, 
                                            w, y, num_sampled, self.vocab_size, 
                                            sampled_values=sampled_values)        
      losses.append(nce_loss)
    return tf.pack(losses, 1)

  def ComputeLoss(self, outputs, out_embeddings, user_embeddings=None):
    reshaped_outputs = tf.reshape(outputs, [-1, outputs.get_shape()[-1].value])

    if user_embeddings is not None:
      replicated = tf.concat(0, [user_embeddings for _ in range(35)])
      reshaped_outputs = tf.concat(1, [replicated, reshaped_outputs])

    reshaped_mask = tf.reshape(self._mask, [-1])
    
    reshaped_labels = tf.reshape(self.y, [-1])
    reshaped_logits = tf.matmul(
      reshaped_outputs, out_embeddings, transpose_b=True) + self.base_bias
    reshaped_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      reshaped_logits, reshaped_labels)
    masked_loss = tf.mul(reshaped_loss, reshaped_mask)

    self.per_word_loss = tf.reshape(masked_loss, outputs.get_shape()[:2])
    per_sentence_loss = tf.reduce_sum(self.per_word_loss, 1)
    self.per_sentence_loss = tf.div(per_sentence_loss, tf.reduce_sum(self._mask, 1))

    return masked_loss

  def CreateDecodingGraph(self, cell, linear_proj, params):
    self.prev_word = tf.placeholder(tf.int32, (), name='prev_word')
    self.prev_c = tf.placeholder(tf.float32, [1, params.cell_size], name='prev_c')
    self.prev_h = tf.placeholder(tf.float32, [1, params.cell_size], name='prev_h')
    prev_embed = tf.nn.embedding_lookup(self._word_embeddings, self.prev_word)
    prev_embed = tf.expand_dims(prev_embed, 0)

    if params.use_softmax_adaptation:
      prev_embed = prev_embed[:, params.context_embed_size:]
    
    state = rnn_cell.LSTMStateTuple(self.prev_c, self.prev_h)
    with vs.variable_scope('RNN', reuse=True):
      result, (self.next_c, self.next_h) = cell(prev_embed, state)
    projected = tf.matmul(result, linear_proj)
    logits = tf.matmul(projected, self._word_embeddings, transpose_b=True) + self.base_bias
    self.next_idx = tf.argmax(logits, 1)
    self.temperature = tf.placeholder_with_default([0.2], [1])
    self.next_prob = tf.nn.softmax(logits / self.temperature)


class HyperModel(BaseModel):

  def __init__(self, params, unigram_probs, context_vocab_sizes, use_nce_loss=True):
    super(HyperModel, self).__init__(params, unigram_probs,
                                     context_vocab_sizes=context_vocab_sizes)

    uembeds = None
    if params.use_mikolov_adaptation or params.use_hyper_adaptation:
      uembeds = self.final_context_embed

    cell = HyperCell(params.cell_size, uembeds, mikolov_adapt=params.use_mikolov_adaptation,
                     hyper_adapt=params.use_hyper_adaptation)

    regularized_cell = rnn_cell.DropoutWrapper(
      cell, output_keep_prob=self.dropout_keep_prob,
      input_keep_prob=self.dropout_keep_prob)
    outputs, _ = tf.nn.dynamic_rnn(regularized_cell, self._inputs, dtype=tf.float32,
                                   sequence_length=self.seq_len)

    linear_proj = tf.get_variable(
      'linear_proj', [params.cell_size, self._word_embeddings.get_shape()[1]])
    self.OutputHelper(outputs, linear_proj, params, use_nce_loss=use_nce_loss)

    self.CreateDecodingGraph(cell, linear_proj, params)


def PrintParams(handle=sys.stdout.write):
  """Print the names of the parameters and their sizes. 

  Args:
    handle: where to write the param sizes to
  """
  handle('NETWORK SIZE REPORT\n')
  param_count = 0
  fmt_str = '{0: <25}\t{1: >12}\t{2: >12,}\n'
  for p in tf.trainable_variables():
    shape = p.get_shape()
    shape_str = 'x'.join([str(x.value) for x in shape])
    handle(fmt_str.format(p.name, shape_str, np.prod(shape).value))
    param_count += np.prod(shape).value
  handle(''.join(['-'] * 60))
  handle('\n')
  handle(fmt_str.format('total', '', param_count))
  if handle==sys.stdout.write:
    sys.stdout.flush()
