# Impelmentation of different word embedding strategies
import gzip
import numpy as np
import tensorflow as tf
from vocab import Vocab

class MikolovEmbeddings(object):
  # traditional word embeddings with optional support for context adaptation

  def __init__(self, params, word_vocab):
    self.embedding_dims = params.embedding_dims
    self.vocab_size = len(word_vocab)

    if len(params.context_vars) == 1:
      context_size = params.context_embed_sizes[0]
    else:
      context_size = params.context_embed_size

    if params.use_softmax_adaptation:
      self._word_embeddings = tf.get_variable(
        'word_embeddings', [self.vocab_size, params.embedding_dims + context_size])
    else:
      self._word_embeddings = tf.get_variable(
        'word_embeddings', [self.vocab_size, params.embedding_dims])
  
  def GetEmbeddings(self, word_ids):
    return tf.nn.embedding_lookup(self._word_embeddings, word_ids)

  def GetAllEmbeddings(self):
    return self._word_embeddings


class PretrainedEmbeddings(object):
  # supports loading pretrained word embeddings

  def __init__(self, params, word_vocab):
    embeddings = []
    with gzip.open(params.pretrained_embedding_filename, 'r') as f:
      for line in f:
        fields = line.split('\t')
        data = [float(x) for x in fields[1:]]
        embeddings.append(data)

    self.word_embeddings = tf.Variable(np.array(embeddings, dtype=np.float32), name='word_embeddings',
                                       trainable=False)
    self.embedding_dims = len(data)

  def GetAllEmbeddings(self):
    return self.word_embeddings

  def GetEmbeddings(self, word_ids):
    return tf.nn.embedding_lookup(self.word_embeddings, word_ids)


class Char2Vec(object):
  # character-based word embeddings

  @staticmethod
  def MakeFilter(width, in_size, num_filters, name):
    filter_sz = [width, in_size, 1, num_filters]
    filter_b = tf.Variable(tf.constant(0.1, shape=[num_filters]),
                           name='{0}_bias'.format(name))
    the_filter = tf.get_variable(name, filter_sz)
    return the_filter, filter_b

  def __init__(self, params, word_vocab, char_vocab, enable_char2vec=True):
    self.enable_char2vec = enable_char2vec

    self.word_embeddings = tf.get_variable(
      'word_embeddings', [len(word_vocab), params.embedding_dims])

    self.all_ids = tf.range(0, len(word_vocab))

    if enable_char2vec: 
      self.MakeCharVocabMat(word_vocab, char_vocab)

      self.char_embedding_size = int(np.log2(len(char_vocab)))
      self.char_embeddings = tf.get_variable(
        'char_embeddings', [len(char_vocab), self.char_embedding_size])

      self.layer1_out_size = params.char2vec_layer1
      self.layer2_out_size = params.char2vec_layer2
      self.filter1, self.bias1 = Char2Vec.MakeFilter(3, self.char_embedding_size, 
                                                     self.layer1_out_size, 'filter1')
      self.filter2 = []
      self.bias2 = []
      self.widths = range(3, 6)
      for width in self.widths:
        f, f_bias = Char2Vec.MakeFilter(width, self.layer1_out_size, self.layer2_out_size,
                                        'filt2_w{0}'.format(width))
        self.filter2.append(f)
        self.bias2.append(f_bias)

      self.embedding_dims = params.embedding_dims + len(self.widths) * self.layer2_out_size
    else:
      self.embedding_dims = params.embedding_dims

  def GetAllEmbeddings(self):
    if self.enable_char2vec:
      return self.GetEmbeddings(self.all_ids)
    else:
      return self.word_embeddings

  def GetEmbeddings(self, word_ids):
    if not self.enable_char2vec:
      return tf.nn.embedding_lookup(self.word_embeddings, word_ids)

    if len(word_ids.get_shape()) > 1:
      unique_ids, unique_idxs = tf.unique(tf.reshape(word_ids, [-1]))
      unique_idxs = tf.reshape(unique_idxs, word_ids.get_shape())
    else:
      unique_ids, unique_idxs = tf.unique(word_ids)
    selected_words = tf.nn.embedding_lookup(self.words_as_chars, unique_ids)

    # z is a tensor of dimensions batch_sz x word_len x embed_dims.
    z = tf.nn.embedding_lookup(self.char_embeddings, selected_words)
    z_expanded = tf.expand_dims(z, -1)

    conv = tf.nn.conv2d(z_expanded, self.filter1, strides=[1, 1, 1, 1],
                        padding='VALID' )
    h = tf.nn.relu(tf.nn.bias_add(tf.squeeze(conv), self.bias1))
    h.set_shape((None, self.max_len - 2, self.layer1_out_size))
    h_expanded = tf.expand_dims(h, -1)

    pools = []
    for f, f_bias, width in zip(self.filter2, self.bias2, self.widths):
      conv2 = tf.nn.conv2d(h_expanded, f, strides=[1, 1, 1, 1],
                           padding='VALID')
      h2 = tf.nn.relu(tf.nn.bias_add(conv2, f_bias))
      pooled = tf.nn.max_pool(h2, ksize=[1, self.max_len - 1 - width, 1, 1],
                              strides=[1, 1, 1, 1], padding='VALID')
      pools.append(pooled)

    pooled = tf.squeeze(tf.concat(axis=3, values=pools), [1, 2])
    charvecs = tf.nn.embedding_lookup(pooled, unique_idxs)
    wordvecs = tf.nn.embedding_lookup(self.word_embeddings, word_ids)
    finalvecs = tf.concat(axis=len(charvecs.get_shape()) - 1, values=[charvecs, wordvecs])

    return finalvecs

  def MakeCharVocabMat(self, word_vocab, char_vocab):
    graphemes = [['{'] + Vocab.Graphemes(x) + ['}'] for x in word_vocab.GetWords()]
    self.max_len = max([len(x) for x in graphemes])
    grapheme_ids = []
    lengths = []
    for g in graphemes:
      ids = [char_vocab[c] for c in g]
      lengths.append(len(ids))
      if len(ids) < self.max_len:
        ids += [char_vocab['}']] * (self.max_len - len(ids))
      grapheme_ids.append(ids)
      
    self.word_lens = tf.Variable(trainable=False, initial_value=lengths, name='word_lens')
    self.words_as_chars = tf.Variable(trainable=False, initial_value=grapheme_ids, 
                                      name='words_as_chars')


