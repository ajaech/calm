import code
import tensorflow as tf
from tensorflow.models.rnn import rnn
from tensorflow.models.rnn import seq2seq
from tensorflow.models.rnn.rnn_cell import RNNCell
from tensorflow.python.ops import variable_scope as vs


class MyLSTMCell(RNNCell):

  def __init__(self, num_units, W, b):
    self._W = W
    self._b = b
    self._num_units = num_units
    self._forget_bias = 1.0

  def __call__(self, inputs, state, scope=None):
    """Long short-term memory cell (LSTM)."""
    with vs.variable_scope(scope or type(self).__name__):  # "MyLSTMCell"
      # Parameters of gates are concatenated into one multiply for efficiency.
      c, h = tf.split(1, 2, state)
      x = tf.concat(1, [h, inputs])
      concat = tf.matmul(x, self._W) + self._b

      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      i, j, f, o = tf.split(1, 4, concat)

      new_c = c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) * tf.tanh(j)
      new_h = tf.tanh(new_c) * tf.sigmoid(o)

      return new_h, tf.concat(1, [new_c, new_h])

  @property
  def state_size(self):
    return 2 * self._num_units


class MultiModel(object):

  def __init__(self, max_length, vocab_size,
               subreddit_vocab_size, year_vocab_size):

    self.x = tf.placeholder(tf.int32, [1, max_length], name='x')
    self.y = tf.placeholder(tf.int32, [1, max_length], name='y')
    self.subreddit = tf.placeholder(tf.int64, [1], 
                                     name='subreddit')
    self.year = tf.placeholder(tf.int64, [1], name='year')
    self.seq_len = tf.placeholder(tf.int64, [1], name='seq_len')

    embedding_dims = 150
    self._word_embeddings = tf.get_variable('word_embeddings',
                                            [vocab_size, embedding_dims])

    num_models = [1, subreddit_vocab_size, year_vocab_size]
    hidden_sizes = [100, 30, 10]
    total_hidden = sum(hidden_sizes)
    input_size = embedding_dims + total_hidden

    main_lstm = tf.get_variable('main_lstm', [input_size, 4, 100])
    main_bias = tf.get_variable('main_bias', [4, 100])
    
    subreddit_lstm = tf.get_variable('subreddit_lstm', 
      [subreddit_vocab_size, input_size, 4, 30])
    subreddit_bias = tf.get_variable('subreddit_bias',
      [subreddit_vocab_size, 4, 30])

    year_lstm = tf.get_variable('year_lstm',
      [year_vocab_size, input_size, 4, 10])
    year_bias = tf.get_variable('year_bias',
      [year_vocab_size, 4, 10])

    sl = tf.squeeze(tf.gather(subreddit_lstm, self.subreddit))
    sl_bias = tf.squeeze(tf.gather(subreddit_bias, self.subreddit))
    yl = tf.squeeze(tf.gather(year_lstm, self.year))
    yl_bias = tf.squeeze(tf.gather(year_bias, self.year))

    lstm_mat = tf.reshape(tf.concat(2, [main_lstm, sl, yl]),
                                    [input_size, -1])
    lstm_bias = tf.reshape(tf.concat(1, [main_bias, sl_bias, yl_bias]),
                           [4 * total_hidden])
    
    cell = MyLSTMCell(total_hidden, lstm_mat, lstm_bias)
    linear_map = tf.get_variable('linear_map', [total_hidden, embedding_dims])
    softmax_bias = tf.get_variable('softmax_bias', [vocab_size])


    # now the computation
    z = tf.nn.embedding_lookup(self._word_embeddings, self.x)
    inputs = [tf.squeeze(_o) for _o in tf.split(1, max_length, z)]
    inputs = [tf.expand_dims(_o, 0) for _o in inputs]

    outputs, _ = rnn.rnn(cell, inputs, dtype=tf.float32)
    
    def GetLogits(_o):
      x = tf.matmul(_o, linear_map)
      logits = tf.matmul(x, self._word_embeddings, transpose_b=True) + softmax_bias
      return tf.squeeze(logits)

    logits = [GetLogits(_o) for _o in outputs]

    self.logits = tf.pack(logits)

    mask = tf.range(0, limit=max_length) < (tf.to_int32(self.seq_len) - 1)
    weights = tf.select(mask, tf.ones([max_length]), tf.zeros([max_length]))

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, 
                                                          tf.squeeze(self.y))
    self.cost = tf.reduce_sum(weights * loss) / tf.to_float(self.seq_len - 1)

    
