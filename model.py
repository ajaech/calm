import code
import collections
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell


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


class MikilovCell(rnn_cell.RNNCell):

  def __init__(self, num_units, user_embeds):
    self._num_units = num_units
    self._forget_bias = 1.0
    self._activation = tf.tanh
    self.user_embeds = user_embeds

  @property
  def state_size(self):
    return rnn_cell.LSTMStateTuple(self._num_units, self._num_units)

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Long short-term memory cell (LSTM)."""
    with vs.variable_scope(scope or "basic_lstm_cell"):
      # Parameters of gates are concatenated into one multiply for efficiency.                                                                                                        
      c, h = state
      concat = _linear([inputs, h], 4 * self._num_units, True, scope=scope)

      biases = tf.get_variable(
        'mikilov_biases', 
        [self.user_embeds.get_shape()[1].value, 4 * self._num_units])

      delta = tf.matmul(self.user_embeds, biases)
      adapted = concat + delta

      # i = input_gate, j = new_input, f = forget_gate, o = output_gate                                                                                                               
      i, j, f, o = tf.split(1, 4, adapted)

      new_c = (c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) *
               self._activation(j))
      new_h = self._activation(new_c) * tf.sigmoid(o)

      new_state = rnn_cell.LSTMStateTuple(new_c, new_h)

      return new_h, new_state
    

class HyperCell(rnn_cell.RNNCell):

  def __init__(self, num_units, user_embeds):
    self._num_units = num_units
    self._forget_bias = 1.0
    self._activation = tf.tanh
    self.user_embeds = user_embeds

  @property
  def state_size(self):
    return rnn_cell.LSTMStateTuple(self._num_units, self._num_units)
    
  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Long short-term memory cell (LSTM)."""
    with vs.variable_scope(scope or "basic_lstm_cell"):
      # Parameters of gates are concatenated into one multiply for efficiency.
      c, h = state
      concat = _linear([inputs, h], 4 * self._num_units, True, scope=scope)

      adaptation_weights = tf.get_variable(
        'adaptation_weights', 
        [self.user_embeds.get_shape()[1].value, 4 * self._num_units])
      adaptation_bias = tf.Variable(np.ones(4 * self._num_units), name='adaptation_bias',
                                    dtype=tf.float32)

      adaptation_coeff = tf.nn.relu(tf.matmul(self.user_embeds, adaptation_weights) 
                                    + adaptation_bias)

      adapted = tf.mul(adaptation_coeff, concat)

      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      i, j, f, o = tf.split(1, 4, adapted)

      new_c = (c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) *
               self._activation(j))
      new_h = self._activation(new_c) * tf.sigmoid(o)

      new_state = rnn_cell.LSTMStateTuple(new_c, new_h)

      return new_h, new_state


class BaseModel(object):
  """Hold the code that is shared between all model varients."""

  def __init__(self, params, vocab_size):
    self.max_length = params.max_len
    self.vocab_size = vocab_size
    self.x = tf.placeholder(tf.int32, [params.batch_size, self.max_length], name='x')
    self.y = tf.placeholder(tf.int64, [params.batch_size, self.max_length], name='y')
    self.seq_len = tf.placeholder(tf.int64, [params.batch_size], name='seq_len')

    self._embedding_dims = params.embedding_dims
    self._word_embeddings = tf.get_variable('word_embeddings',
                                            [vocab_size, self._embedding_dims])

    self._inputs = tf.nn.embedding_lookup(self._word_embeddings, self.x)
    self.base_bias = tf.get_variable('base_bias', [vocab_size])
    
    # make a mask
    lengths_transposed = tf.expand_dims(tf.to_int32(self.seq_len), 1)
    lengths_tiled = tf.tile(lengths_transposed, [1, self.max_length])
    r = tf.range(0, self.max_length, 1)
    range_row = tf.expand_dims(r, 0)
    range_tiled = tf.tile(range_row, [params.batch_size, 1])
    indicator = tf.less(range_tiled, lengths_tiled)
    sz = [params.batch_size, self.max_length]
    self._mask = tf.select(indicator, tf.ones(sz), tf.zeros(sz))

  def DoNCE(self, weights, linear_map, num_sampled=256):
    w_list = tf.unpack(weights, axis=1)
    losses = []
    for w, y in zip(tf.unpack(weights, axis=1), tf.split(1, self.max_length, self.y)):
      w_mapped = tf.matmul(w, linear_map, transpose_b=True)

      sampled_values = tf.nn.learned_unigram_candidate_sampler(
        true_classes=y,
        num_true=1,
        num_sampled=num_sampled,
        unique=True,
        range_max=self.vocab_size)

      nce_loss = tf.nn.nce_loss(self._word_embeddings, self.base_bias, 
                                w_mapped, y, num_sampled, self.vocab_size, 
                                sampled_values=sampled_values)        
      losses.append(nce_loss)
    return tf.pack(losses, 1)

  def ComputeLoss(self, outputs, linear_map):
    reshaped_outputs = tf.reshape(outputs, [-1, outputs.get_shape()[-1].value])
    resized_outputs = tf.matmul(reshaped_outputs, linear_map, transpose_b=True)
    reshaped_mask = tf.reshape(self._mask, [-1])
    
    reshaped_labels = tf.reshape(self.y, [-1])
    reshaped_logits = tf.matmul(
      resized_outputs, self._word_embeddings, transpose_b=True) + self.base_bias
    reshaped_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      reshaped_logits, reshaped_labels)
    masked_loss = tf.mul(reshaped_loss, reshaped_mask)
    return masked_loss

class StandardModel(BaseModel):

  def __init__(self, params, vocab_size, use_nce_loss=True):
    self.batch_size = params.batch_size
    super(StandardModel, self).__init__(params, vocab_size)

    hidden_size = 150
    cell = rnn_cell.LSTMCell(hidden_size)
    outputs, _ = tf.nn.dynamic_rnn(cell, self._inputs, dtype=tf.float32,
                                   sequence_length=self.seq_len)

    linear_map = tf.get_variable('linear_map', [self._embedding_dims, hidden_size])

    if use_nce_loss:
      losses = self.DoNCE(outputs, linear_map)
      masked_loss = tf.mul(losses, self._mask)
    else:
      masked_loss = self.ComputeLoss(outputs, linear_map)
      self.masked_loss = masked_loss
    self.cost = tf.reduce_sum(masked_loss) / tf.reduce_sum(self._mask)

class MikilovModel(BaseModel):

  def __init__(self, params, vocab_size, user_size, use_nce_loss=True):
    super(MikilovModel, self).__init__(params, vocab_size)

    self.username = tf.placeholder(tf.int64, [params.batch_size], name='username')
    user_embeddings = tf.get_variable('user_embeddings', [user_size, 80])
    self._user_embeddings = user_embeddings
    uembeds = tf.nn.embedding_lookup(user_embeddings, self.username)

    hidden_size = 150
    cell = MikilovCell(hidden_size, uembeds)
    outputs, _ = tf.nn.dynamic_rnn(cell, self._inputs, dtype=tf.float32,
                                   sequence_length=self.seq_len)

    linear_map = tf.get_variable('linear_map', [self._embedding_dims, hidden_size])

    if use_nce_loss:
      losses = self.DoNCE(outputs, linear_map)
      masked_loss = tf.mul(losses, self._mask)
    else:
      masked_loss = self.ComputeLoss(outputs, linear_map)
      self.masked_loss = masked_loss
    self.cost = tf.reduce_sum(masked_loss) / tf.reduce_sum(self._mask)


class HyperModel(BaseModel):

  def __init__(self, params, vocab_size, user_size, use_nce_loss=True):
    super(HyperModel, self).__init__(params, vocab_size)
    
    self.username = tf.placeholder(tf.int64, [params.batch_size], name='username')
    user_embeddings = tf.get_variable('user_embeddings', [user_size, 80])
    self._user_embeddings = user_embeddings
    uembeds = tf.nn.embedding_lookup(user_embeddings, self.username)

    hidden_size = 150
    cell = HyperCell(hidden_size, uembeds)
    outputs, _ = tf.nn.dynamic_rnn(cell, self._inputs, dtype=tf.float32,
                                   sequence_length=self.seq_len)

    linear_map = tf.get_variable('linear_map', [self._embedding_dims, hidden_size])

    if use_nce_loss:
      losses = self.DoNCE(outputs, linear_map)
      masked_loss = tf.mul(losses, self._mask)
    else:
      masked_loss = self.ComputeLoss(outputs, linear_map)
      self.masked_loss = masked_loss
    self.cost = tf.reduce_sum(masked_loss) / tf.reduce_sum(self._mask)
    
