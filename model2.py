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

_LSTMStateTuple = collections.namedtuple("LSTMStateTuple", ("c", "h"))


class LSTMStateTuple(_LSTMStateTuple):
  """Tuple used by LSTM Cells for `state_size`, `zero_state`, and output state.
  Stores two elements: `(c, h)`, in that order.
  Only used when `state_is_tuple=True`.
  """
  __slots__ = ()

  @property
  def dtype(self):
    (c, h) = self
    if not c.dtype == h.dtype:
      raise TypeError("Inconsistent internal state: %s vs %s" %
                      (str(c.dtype), str(h.dtype)))
    return c.dtype


class HyperCell(rnn_cell.RNNCell):

  def __init__(self, num_units, user_embeds):
    self._num_units = num_units
    self._forget_bias = 1.0
    self._activation = tf.tanh
    self.user_embeds = user_embeds

  @property
  def state_size(self):
    return LSTMStateTuple(self._num_units, self._num_units)
    
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

      new_state = LSTMStateTuple(new_c, new_h)

      return new_h, new_state


class BaseModel(object):
  """Hold the code that is shared between all model varients."""

  def __init__(self, max_length, vocab_size, batch_size):
    self.max_length = max_length
    self.x = tf.placeholder(tf.int32, [batch_size, max_length], name='x')
    self.y = tf.placeholder(tf.int64, [batch_size, max_length], name='y')
    self.seq_len = tf.placeholder(tf.int64, [batch_size], name='seq_len')

    self._embedding_dims = 180
    self._word_embeddings = tf.get_variable('word_embeddings',
                                            [vocab_size, self._embedding_dims])

    self._inputs = tf.nn.embedding_lookup(self._word_embeddings, self.x)
    self.base_bias = tf.get_variable('base_bias', [vocab_size])
    
    # make a mask
    lengths_transposed = tf.expand_dims(tf.to_int32(self.seq_len), 1)
    lengths_tiled = tf.tile(lengths_transposed, [1, max_length])
    r = tf.range(0, max_length, 1)
    range_row = tf.expand_dims(r, 0)
    range_tiled = tf.tile(range_row, [batch_size, 1])
    indicator = tf.less(range_tiled, lengths_tiled)
    sz = [batch_size, max_length]
    self._mask = tf.select(indicator, tf.ones(sz), tf.zeros(sz))


class StandardModel(BaseModel):

  def __init__(self, max_length, vocab_size, use_nce_loss=True):
    self.batch_size = 100
    super(StandardModel, self).__init__(max_length, vocab_size, self.batch_size)

    hidden_size = 150
    cell = rnn_cell.LSTMCell(hidden_size)
    outputs, _ = tf.nn.dynamic_rnn(cell, self._inputs, dtype=tf.float32,
                                   sequence_length=self.seq_len)

    linear_map = tf.get_variable('linear_map', [self._embedding_dims, hidden_size])

    reshaped_outputs = tf.reshape(outputs, [-1, hidden_size])
    resized_outputs = tf.matmul(reshaped_outputs, linear_map, transpose_b=True)
    reshaped_mask = tf.reshape(self._mask, [-1])

    if use_nce_loss:
      reshaped_labels = tf.reshape(self.y, [-1, 1])
      num_sampled = 256
      sampled_values = tf.nn.learned_unigram_candidate_sampler(
        true_classes=reshaped_labels,
        num_true=1,
        num_sampled=num_sampled,
        unique=False,
        range_max=vocab_size)
      nce_loss = tf.nn.nce_loss(self._word_embeddings, self.base_bias, resized_outputs,
                                reshaped_labels, num_sampled, vocab_size,
                                sampled_values=sampled_values)
      reshaped_loss = tf.reshape(nce_loss, [self.batch_size, max_length])
      masked_loss = tf.mul(nce_loss, reshaped_mask)
    else:
      reshaped_labels = tf.reshape(self.y, [-1])
      reshaped_logits = tf.matmul(
        resized_outputs, self._word_embeddings, transpose_b=True) + self.base_bias
      reshaped_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        reshaped_logits, reshaped_labels)
      masked_loss = tf.mul(reshaped_loss, reshaped_mask)
      self.masked_loss = masked_loss
    self.cost = tf.reduce_sum(masked_loss) / tf.reduce_sum(self._mask)


class HyperModel(BaseModel):

  def __init__(self, max_length, vocab_size, user_size, use_nce_loss=True):
    self.batch_size = 100
    super(HyperModel, self).__init__(max_length, vocab_size, self.batch_size)
    
    self.username = tf.placeholder(tf.int64, [self.batch_size], name='username')
    user_embeddings = tf.get_variable('user_embeddings', [user_size, 80])
    self._user_embeddings = user_embeddings
    uembeds = tf.nn.embedding_lookup(user_embeddings, self.username)

    hidden_size = 150
    cell = HyperCell(hidden_size, uembeds)
    outputs, _ = tf.nn.dynamic_rnn(cell, self._inputs, dtype=tf.float32,
                                   sequence_length=self.seq_len)

    linear_map = tf.get_variable('linear_map', [self._embedding_dims, hidden_size])

    reshaped_outputs = tf.reshape(outputs, [-1, hidden_size])
    resized_outputs = tf.matmul(reshaped_outputs, linear_map, transpose_b=True)
    reshaped_mask = tf.reshape(self._mask, [-1])
    
    if use_nce_loss:
      reshaped_labels = tf.reshape(self.y, [-1, 1])
      num_sampled = 128
      sampled_values = tf.nn.learned_unigram_candidate_sampler(
        true_classes=reshaped_labels,
        num_true=1,
        num_sampled=num_sampled,
        unique=True,
        range_max=vocab_size)
      nce_loss = tf.nn.nce_loss(self._word_embeddings, self.base_bias, resized_outputs,
                                reshaped_labels, num_sampled, vocab_size,
                                sampled_values=sampled_values)
      reshaped_loss = tf.reshape(nce_loss, [100, 35])
      masked_loss = tf.mul(nce_loss, reshaped_mask)
    else:
      reshaped_labels = tf.reshape(self.y, [-1])
      reshaped_logits = tf.matmul(
        resized_outputs, self._word_embeddings, transpose_b=True) + self.base_bias
      reshaped_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        reshaped_logits, reshaped_labels)
      masked_loss = tf.mul(reshaped_loss, reshaped_mask)
      self.masked_loss = masked_loss
    self.cost = tf.reduce_sum(masked_loss) / tf.reduce_sum(self._mask)
    

class BiasModel(BaseModel):
  def __init__(self, max_length, vocab_size, user_size, fancy_bias=True, use_nce_loss=True):

    self.fancy_bias = fancy_bias
    self.batch_size = 100
    super(BiasModel, self).__init__(max_length, vocab_size, self.batch_size)

    self.hash_size = 50000000
    self.corrections = tf.Variable(tf.zeros([self.hash_size]), name='corrections')

    self.username = tf.placeholder(
      tf.int64, [self.batch_size], name='username')

    hidden_size = 150
    cell = tf.nn.rnn_cell.LSTMCell(hidden_size, state_is_tuple=True)
    outputs, _ = tf.nn.dynamic_rnn(cell, self._inputs, dtype=tf.float32,
                                   sequence_length=self.seq_len)

    linear_map = tf.get_variable('linear_map', [self._embedding_dims, hidden_size])
    self._weights = tf.matmul(self._word_embeddings, linear_map)

    if use_nce_loss:
      num_sampled=128
      losses = []
      for oo, yy, xx in zip(tf.unpack(outputs), tf.unpack(self.y), tf.unpack(self.x)):
        self.xx = xx
        yy = tf.expand_dims(yy, 1)
        loss = self.mynce(oo, yy, num_sampled=num_sampled, num_classes=vocab_size)
        losses.append(loss)
      losses = tf.pack(losses)
        
    else:
      # used for calculating perplexity
      reshaped_outputs = tf.reshape(outputs, [-1, hidden_size])
      reshaped_logits = tf.matmul(reshaped_outputs,
                                  self._weights, transpose_b=True)
      logits = tf.reshape(reshaped_logits, [self.batch_size, max_length, -1])
      logits_with_bias = logits + self.base_bias

      if fancy_bias:
        next_word_idx = tf.reshape(tf.range(0, vocab_size), (1, 1, vocab_size))
        prev_word_idx = self.x * vocab_size
        pre_hash = tf.tile(tf.expand_dims(prev_word_idx, dim=2), (1, 1, vocab_size))
        hash = tf.mod(pre_hash + next_word_idx, self.hash_size)
        self.h = (next_word_idx, prev_word_idx, hash)
        correction = tf.nn.embedding_lookup(self.corrections, hash)
        
        logits_with_bias += correction

      losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits_with_bias,
                                                              tf.squeeze(self.y))
      self.logits = logits_with_bias
      self.losses = losses

    self.cost = tf.reduce_sum(losses * self._mask) / tf.reduce_sum(self._mask)

    """
    # next word prediction
    self.wordid = tf.placeholder(tf.int32, [1], name='wordid')
    self.prevstate_c = tf.placeholder(tf.float32, [1, cell.state_size.c],
                                    name='prevstate')
    self.prevstate_h = tf.placeholder(tf.float32, [1, cell.state_size.h],
                                      name='prevstate2')
    word_embedding = tf.nn.embedding_lookup(self._word_embeddings,
                                            self.wordid)
    code.interact(local=locals())
    _o, hidden_out = cell(word_embedding, (self.prevstate_c, self.prevstate_h),
                          scope='LSTMCell')
    self.nextstate_h = hidden_out.h
    self.nextstate_c = hidden_out.c
    q = tf.matmul(tf.matmul(_o, linear_map), self._word_embeddings, 
                  transpose_b=True)
    next_word_logit = q + self.base_bias
    self.next_word_prob = tf.nn.softmax(next_word_logit)
    """

  def mynce(self, inputs, labels, num_sampled,
            num_classes, num_true=1,
            remove_accidental_hits=False,
            partition_strategy='mod',
            name='nce_loss'):

      logits, labels = self._compute_sampled_logits(
        inputs, labels, num_sampled, num_classes,
        num_true=num_true,
        subtract_log_q=True,
        remove_accidental_hits=remove_accidental_hits,
        partition_strategy=partition_strategy,
        name=name)
      sampled_losses = tf.nn.sigmoid_cross_entropy_with_logits(
        logits, labels, name="sampled_losses")
      
      return _sum_rows(sampled_losses)


  def _compute_sampled_logits(self, inputs, labels, num_sampled,
                              num_classes, num_true=1,
                              subtract_log_q=True,
                              remove_accidental_hits=False,
                              partition_strategy="mod",
                              name=None):

    weights = self._weights
    if not isinstance(weights, list):
      weights = [weights]

    with tf.name_scope(name, 'compute_sampled_logits', weights + [inputs, labels]):
      if labels.dtype != tf.int64:
        labels = tf.cast(labels, tf.int64)
      prev = tf.cast(self.xx, tf.int64)
      labels_flat = tf.reshape(labels, [-1])

      # Sample the negative labels.                                                       
      #   sampled shape: [num_sampled] tensor                                             
      #   true_expected_count shape = [batch_size, 1] tensor                              
      #   sampled_expected_count shape = [num_sampled] tensor
      sampled_values = tf.nn.learned_unigram_candidate_sampler(
        true_classes=labels,
        num_true=1,
        num_sampled=num_sampled,
        unique=True,
        range_max=num_classes)

      sampled, true_expected_count, sampled_expected_count = sampled_values

      true_hash = tf.mod(prev * num_classes + labels_flat, self.hash_size)
      true_corrections = tf.nn.embedding_lookup(self.corrections, true_hash)
      sampled_tile = tf.tile(tf.expand_dims(sampled, 0), [self.max_length, 1])
      sampled_hash = tf.mod(tf.add(tf.expand_dims(prev * num_classes, 1),
                                   sampled_tile), self.hash_size)
      sampled_corrections = tf.nn.embedding_lookup(self.corrections, sampled_hash)

      # labels_flat is a [batch_size * num_true] tensor                                   
      # sampled is a [num_sampled] int tensor                                             
      all_ids = tf.concat(0, [labels_flat, sampled])

      # weights shape is [num_classes, dim]                                               
      all_w = tf.nn.embedding_lookup(
        weights, all_ids, partition_strategy=partition_strategy)

      all_b = tf.nn.embedding_lookup(self.base_bias, all_ids, 
                                     partition_strategy=partition_strategy)

      # true_w shape is [batch_size * num_true, dim]                                      
      # true_b is a [batch_size * num_true] tensor                                        
      true_w = tf.slice(
        all_w, [0, 0], tf.pack([tf.shape(labels_flat)[0], -1]))
      true_b = tf.slice(all_b, [0], tf.shape(labels_flat))

      # inputs shape is [batch_size, dim]                                                 
      # true_w shape is [batch_size * num_true, dim]                                      
      # row_wise_dots is [batch_size, num_true, dim]                                      
      dim = tf.shape(true_w)[1:2]
      new_true_w_shape = tf.concat(0, [[-1, num_true], dim])
      row_wise_dots = tf.mul(
        tf.expand_dims(inputs, 1),
        tf.reshape(true_w, new_true_w_shape))
      # We want the row-wise dot plus biases which yields a                               
      # [batch_size, num_true] tensor of true_logits.                                     
      dots_as_matrix = tf.reshape(row_wise_dots,
                                  tf.concat(0, [[-1], dim]))
      true_logits = tf.reshape(_sum_rows(dots_as_matrix), [-1, num_true])
      true_b = tf.reshape(true_b, [-1, num_true])
      true_logits += true_b
      if self.fancy_bias:
        true_logits += true_corrections
      
      # Lookup weights and biases for sampled labels.                                     
      #   sampled_w shape is [num_sampled, dim]                                           
      #   sampled_b is a [num_sampled] float tensor                                       
      sampled_w = tf.slice(
        all_w, tf.pack([tf.shape(labels_flat)[0], 0]), [-1, -1])
      sampled_b = tf.slice(all_b, tf.shape(labels_flat), [-1])

      # inputs has shape [batch_size, dim]                                                
      # sampled_w has shape [num_sampled, dim]                                            
      # sampled_b has shape [num_sampled]                                                 
      # Apply X*W'+B, which yields [batch_size, num_sampled]                              
      sampled_logits = tf.matmul(inputs,
                                 sampled_w,
                                 transpose_b=True) + sampled_b
      if self.fancy_bias:
        sampled_logits += sampled_corrections

      if remove_accidental_hits:
        acc_hits = tf.nn.compute_accidental_hits(
          labels, sampled, num_true=num_true)
        acc_indices, acc_ids, acc_weights = acc_hits

        # This is how SparseToDense expects the indices.                                  
        acc_indices_2d = tf.reshape(acc_indices, [-1, 1])
        acc_ids_2d_int32 = tf.reshape(tf.cast(
          acc_ids, dtypes.int32), [-1, 1])
        sparse_indices = tf.concat(
          1, [acc_indices_2d, acc_ids_2d_int32], "sparse_indices")
        # Create sampled_logits_shape = [batch_size, num_sampled]                         
        sampled_logits_shape = tf.concat(
          0,
          [tf.shape(labels)[:1], tf.expand_dims(num_sampled, 0)])
        if sampled_logits.dtype != acc_weights.dtype:
          acc_weights = tf.cast(acc_weights, sampled_logits.dtype)
          sampled_logits += tf.sparse_to_dense(
            sparse_indices, sampled_logits_shape, acc_weights,
            default_value=0.0, validate_indices=False)

      if subtract_log_q:
        # Subtract log of Q(l), prior probability that l appears in sampled.              
        true_logits -= tf.log(true_expected_count)
        sampled_logits -= tf.log(sampled_expected_count)

      # Construct output logits and labels. The true labels/logits start at col 0.        
      out_logits = tf.concat(1, [true_logits, sampled_logits])
      # true_logits is a float tensor, ones_like(true_logits) is a float tensor           
      # of ones. We then divide by num_true to ensure the per-example labels sum          
      # to 1.0, i.e. form a proper probability distribution.                              
      out_labels = tf.concat(
        1, [tf.ones_like(true_logits) / num_true,
            tf.zeros_like(sampled_logits)])

    return out_logits, out_labels

def _sum_rows(x):
  """Returns a vector summing up each row of the matrix x."""
  # _sum_rows(x) is equivalent to math_ops.reduce_sum(x, 1) when x is                   
  # a matrix.  The gradient of _sum_rows(x) is more efficient than                      
  # reduce_sum(x, 1)'s gradient in today's implementation. Therefore,                   
  # we use _sum_rows(x) in the nce_loss() computation since the loss                    
  # is mostly used for training.                                                        
  cols = tf.shape(x)[1]
  ones_shape = tf.pack([cols, 1])
  ones = tf.ones(ones_shape, x.dtype)
  return tf.reshape(tf.matmul(x, ones), [-1])
