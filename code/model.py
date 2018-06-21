import gzip
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn_cell

from factorcell import FactorCell
import nn_impl


np.random.seed(666)


class BaseModel(object):
  """Hold the code that is shared between all model varients."""

  def __init__(self, params, word_vocab, context_vocab_sizes=None, reverse=False,
               exclude_unk=False, word_embedder=None):
    self.unigram_probs = word_vocab.GetUnigramProbs()
    self.max_length = params.max_len
    self.vocab_size = len(word_vocab)
    self.num_context_vars = len(context_vocab_sizes)
    self.word_ids = tf.placeholder(tf.int64, [params.batch_size, self.max_length + 1],
                                   name='word_ids')
    self.seq_len = tf.placeholder(tf.int64, [params.batch_size], name='seq_len')

    if reverse:  # provides the option to train a backwards language model
      word_ids_reversed = tf.reverse_sequence(self.word_ids, self.seq_len, 1)
      self.x = word_ids_reversed[:, :-1]
      self.y = word_ids_reversed[:, 1:]
    else:
      self.x = self.word_ids[:, :-1]
      self.y = self.word_ids[:, 1:]

    self.l1_penalty = 0.0
    if hasattr(params, 'l1_penalty'):
      self.l1_penalty = params.l1_penalty

    enable_low_rank_adapt = (params.use_mikolov_adaptation or params.use_lowrank_adaptation or
                             params.use_softmax_adaptation)
    if enable_low_rank_adapt or params.use_context_dependent_bias:
      self.context_placeholders = {}
      self.context_embeddings = {}
      context_embeds = []
      for i, c_var in enumerate(params.context_vars):
        # default case is a categorical context var
        if (not hasattr(params, 'context_var_types') or 
            params.context_var_types[i] == 'categorical'):
          self.context_placeholders[c_var] = tf.placeholder(tf.int32, [None], name=c_var)
          if enable_low_rank_adapt:
            if hasattr(params, 'onehot_context') and params.onehot_context:
              context_embeds.append(tf.one_hot(
                self.context_placeholders[c_var], context_vocab_sizes[i],
                dtype=tf.float32))
            else:
              self.context_embeddings[params.context_vars[i]] = tf.get_variable(
                'c_embed_{0}'.format(c_var), 
                [context_vocab_sizes[i], params.context_embed_sizes[i]])

              context_embeds.append(tf.nn.embedding_lookup(
                self.context_embeddings[c_var], self.context_placeholders[c_var]))
        else:  # numerical context var type
          self.context_placeholders[c_var] = tf.placeholder(tf.float32, [None], name=c_var)
          context_expanded = tf.expand_dims(self.context_placeholders[c_var], 1)
          if enable_low_rank_adapt:
            context_mat = tf.get_variable('ctx_mat_{0}'.format(c_var), 
                                          [1, params.context_embed_sizes[i]])
            context_bias = tf.get_variable('ctx_bias_{0}'.format(c_var),
                                           [params.context_embed_sizes[i]])
            context_embed = tf.nn.relu(
              tf.matmul(context_expanded, context_mat) + context_bias + 1.0)
            context_embeds.append(context_embed)

      if len(context_embeds) == 1:
        self.final_context_embed = context_embeds[0]
        self.context_size = params.context_embed_sizes[0]
      elif len(context_embeds) > 1:
        context_embeds = tf.concat(axis=1, values=context_embeds)

        context_mlp = tf.get_variable(
          'context_mlp', [sum(params.context_embed_sizes), params.context_embed_size])
        context_bias = tf.get_variable('context_bias', [params.context_embed_size])
        
        self.final_context_embed = tf.nn.tanh(tf.matmul(context_embeds, context_mlp) + 
                                              context_bias)
        self.context_size = params.context_embed_size

    # Lookup the input embeddings
    self._inputs = word_embedder.GetEmbeddings(self.x)
    if params.use_softmax_adaptation:
      self._inputs = self._inputs[:, :, self.context_size:]
    
    if word_vocab.GetUnigramProbs() is None:
      self.base_bias = tf.get_variable('base_bias', [self.vocab_size])
    else:
      base_bias_init_val = np.log(word_vocab.GetUnigramProbs() + 0.0000001)
      self.base_bias = tf.Variable(base_bias_init_val, name='base_bias',
                                   trainable=not params.use_context_dependent_bias,
                                   dtype=tf.float32)

    self.dropout_keep_prob = tf.placeholder_with_default(1.0, (), name='keep_prob')
    
    # Make a mask to delete the padding
    indicator = tf.sequence_mask(tf.to_int32(self.seq_len - 1), self.max_length)
    if exclude_unk:
      indicator = tf.logical_and(indicator, tf.not_equal(self.y, 0))
    sz = [params.batch_size, self.max_length]
    self._mask = tf.where(indicator, tf.ones(sz), tf.zeros(sz))

  def OutputHelper(self, reshaped_outputs, params, use_nce_loss=True, hash_func=None):
    self.cost = 0.0  # default cost value
    if use_nce_loss:
      # proj_out will be batch_size x max_len x k
      proj_out =  tf.reshape(reshaped_outputs, [self._mask.get_shape()[0].value,
                                                self._mask.get_shape()[1].value, -1])
      # add in the context embeddings
      if params.use_softmax_adaptation:
        packed_context_embed = tf.stack([self.final_context_embed] * params.max_len, 1)
        proj_out = tf.concat(axis=2, values=[packed_context_embed, proj_out])

      losses, l1_losses = self.AltNCE(proj_out, self.word_embedder.GetEmbeddings,
                                      params.nce_samples, hash_func)
      self.l1_loss = self.l1_penalty * tf.reduce_mean(l1_losses)
      self.cost = self.l1_loss
      
      masked_loss = tf.multiply(losses, self._mask)
    else:
      # add in the context embeddings
      if params.use_softmax_adaptation:
        packed_context_embed = tf.stack([self.final_context_embed] * params.max_len, 1)
        reshaped_context = tf.reshape(packed_context_embed, [tf.shape(reshaped_outputs)[0], -1])
        reshaped_outputs = tf.concat(axis=1, values=[reshaped_context, reshaped_outputs])


      masked_loss = self.ComputeLoss(reshaped_outputs, hash_func=hash_func)

    self.per_word_loss = tf.reshape(masked_loss, [-1, self.max_length])
    self.per_sentence_loss = tf.div(tf.reduce_sum(self.per_word_loss, 1),
                                    tf.reduce_sum(self._mask, 1))

    self.cost += tf.reduce_sum(masked_loss) / tf.reduce_sum(self._mask)    

  def _GetSample(self, true_classes, num_sampled):
    """Helper function for sampled softmax loss."""
    return tf.nn.learned_unigram_candidate_sampler(
      true_classes=true_classes,
      num_true=1,
      num_sampled=num_sampled,
      unique=True,
      range_max=self.vocab_size)

  def AltNCE(self, weights, EmbeddingGetter, num_sampled, hash_func):
    """This is the version of NCE that is compatible with the feature hashing.

    To make it work, the sampling is done per sentence rather than per time-step.
    """
    losses = []
    w_unpack = tf.unstack(weights, axis=0)
    y_unpack = tf.unstack(self.y, axis=0)

    # first get all the samples
    y_expanded = []
    self.sampled_values = []
    for y in y_unpack:
      y_expand = tf.expand_dims(y, 1)
      y_expanded.append(y_expand)
      sampled_values = self._GetSample(y_expand, num_sampled)
      self.sampled_values.append(sampled_values)

    l1_losses = []
    for idx, (w, y_expand, sampled_vals) in enumerate(zip(w_unpack, y_expanded,
                                                          self.sampled_values)):
      h_func = None
      if hash_func is not None:
        context_var_dict = {c_var: self.context_placeholders[c_var][idx] 
                            for c_var in self.context_placeholders.keys()}
        h_func = lambda(x): hash_func(x, context_var_dict)

      nce_loss, l1_loss = nn_impl.sampled_softmax_loss(EmbeddingGetter, self.base_bias,
                                                       y_expand, w, num_sampled, self.vocab_size,
                                                       sampled_values=sampled_vals,
                                                       hash_func=h_func)
      l1_losses.append(l1_loss)
      losses.append(nce_loss)
    return tf.stack(losses, 0), l1_losses

  def ComputeLoss(self, reshaped_outputs, hash_func=None):
    """Computes loss without sampling (full vocabulary)."""

    out_embeddings = self.word_embedder.GetAllEmbeddings()

    reshaped_mask = tf.reshape(self._mask, [-1])
    reshaped_labels = tf.reshape(self.y, [-1])

    bias = self.base_bias
    if hasattr(self, 'adapted_bias'):
      bias += self.adapted_bias
    reshaped_logits = tf.matmul(
      reshaped_outputs, out_embeddings, transpose_b=True) + bias

    if hash_func is not None:
      all_ids = tf.range(0, self.vocab_size)
      hash_vals = []
      for idx in range(self.x.get_shape()[0]):  # loop over batch_size
        context_var_dict = {c_var: self.context_placeholders[c_var][idx] 
                            for c_var in self.context_placeholders.keys()}
        hash_vals.append(hash_func(all_ids, context_var_dict))
      hash_vals = tf.stack(hash_vals)
      expanded_hash_vals = tf.stack([hash_vals] * self.max_length, 1)
      reshaped_hash_vals = tf.reshape(expanded_hash_vals, [-1, self.vocab_size])
      reshaped_logits += reshaped_hash_vals
    
    reshaped_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=reshaped_logits, labels=reshaped_labels)
    masked_loss = tf.multiply(reshaped_loss, reshaped_mask)

    return masked_loss

  def CreateDecodingGraph(self, params):
    """Construct the part of the graph used for decoding."""

    out_embeddings = self.word_embedder.GetAllEmbeddings()

    # placeholders for decoder
    self.prev_word = tf.placeholder(tf.int32, (), name='prev_word')
    self.prev_c = tf.get_variable('prev_c', [1, params.cell_size], dtype=tf.float32,
                                  collections=[tf.GraphKeys.LOCAL_VARIABLES])
    self.prev_h = tf.get_variable('prev_h', [1, params.cell_size], dtype=tf.float32,
                                  collections=[tf.GraphKeys.LOCAL_VARIABLES])
    self.temperature = tf.placeholder_with_default([1.0], [1])

    # lookup embedding
    prev_embed = tf.nn.embedding_lookup(out_embeddings, self.prev_word)
    prev_embed = tf.expand_dims(prev_embed, 0)

    if params.use_softmax_adaptation:
      prev_embed = prev_embed[:, self.context_size:]

    # one iteration of recurrent layer
    state = rnn_cell.LSTMStateTuple(self.prev_c, self.prev_h)
    with tf.variable_scope('RNN', reuse=True):
      result, (self.next_c, self.next_h) = self.cell(prev_embed, state)

    proj_result = tf.matmul(result, self.linear_proj)
    if params.use_softmax_adaptation:
      proj_result = tf.concat(axis=1, values=[self.final_context_embed, proj_result])
      
    # softmax layer
    bias = self.base_bias
    if params.use_context_dependent_bias:
      hval = self.hash_func(self.all_ids, self.context_placeholders)
      bias += hval

    self.beam_size = tf.placeholder_with_default(1, (), name='beam_size')
    logits = tf.matmul(proj_result, out_embeddings, transpose_b=True) + bias
    self.next_prob = tf.nn.softmax(logits / self.temperature)
    #self.selected = tf.multinomial(logits / self.temperature, self.beam_size)
    self.selected = tf.squeeze(tf.multinomial(logits / self.temperature, self.beam_size))
    self.selected, _ = tf.unique(self.selected)
    self.selected_p = tf.nn.embedding_lookup(tf.transpose(self.next_prob), self.selected)
    
    assign1 = self.prev_c.assign(self.next_c)
    assign2 = self.prev_h.assign(self.next_h)
    self.assign_op = tf.group(assign1, assign2)

    # reset state
    assign1 = self.prev_c.assign(tf.zeros_like(self.prev_c))
    assign2 = self.prev_h.assign(tf.zeros_like(self.prev_h))
    self.reset_state = tf.group(assign1, assign2)


class HyperModel(BaseModel):

  def __init__(self, params, word_vocab, context_vocabs, use_nce_loss=True, reverse=False,
               exclude_unk=True, word_embedder=None):
    self.all_ids = tf.range(0, len(word_vocab))
    self.word_embedder = word_embedder
    self.word_tensor = tf.constant(word_vocab.GetWords(), name='words')

    context_vocab_sizes = []
    for s in params.context_vars:
      if context_vocabs[s]:
        context_vocab_sizes.append(len(context_vocabs[s]))
      else:
        context_vocab_sizes.append(0)
    super(HyperModel, self).__init__(params, word_vocab,
                                     context_vocab_sizes=context_vocab_sizes,
                                     reverse=reverse, exclude_unk=exclude_unk,
                                     word_embedder=self.word_embedder)

    self.hash_func = None  # setup the hash table
    if params.use_context_dependent_bias:
      self.hash_func = self.GetContextDependentBias(params, context_vocab_sizes)

    context_embeds = None
    if params.use_mikolov_adaptation or params.use_lowrank_adaptation:
      context_embeds = self.final_context_embed

    layer_norm = False
    if hasattr(params, 'use_layer_norm'):
      layer_norm = params.use_layer_norm
    self.cell = FactorCell(params.cell_size, 
                           self.word_embedder.embedding_dims, 
                           context_embeds,
                           mikilovian_adaptation=params.use_mikolov_adaptation,
                           lowrank_adaptation=params.use_lowrank_adaptation,
                           rank=params.rank, 
                           dropout_keep_prob=self.dropout_keep_prob,
                           layer_norm=layer_norm)

    self.linear_proj = tf.get_variable(
      'linear_proj', [params.cell_size, self.word_embedder.embedding_dims])
    outputs, _ = tf.nn.dynamic_rnn(self.cell, self._inputs, dtype=tf.float32,
                                   sequence_length=self.seq_len)
    reshaped_outputs = tf.reshape(outputs, [-1, params.cell_size])
    self.outputs = reshaped_outputs
    projected_outputs = tf.matmul(reshaped_outputs, self.linear_proj)
    self.OutputHelper(projected_outputs, params, use_nce_loss=use_nce_loss,
                      hash_func=self.hash_func)

    self.CreateDecodingGraph(params)


  def GetContextDependentBias(self, params, context_vocab_sizes):
    self.bias_tables = {}
    for context_var, size in zip(params.context_vars, context_vocab_sizes):
      self.bias_tables[context_var] = tf.get_variable(context_var + '_bias',
                                                      [self.vocab_size, size])

    def GetBias(ids, s_ids, debug=False):
      ids = tf.to_int32(ids)
      
      result = 0
      for c_var in s_ids.keys():
        bias_table = self.bias_tables[c_var]

        # first lookup the ids
        selected_ids = tf.nn.embedding_lookup(bias_table, ids)
        result += tf.nn.embedding_lookup(tf.transpose(selected_ids),
                                         s_ids[c_var])
      return result

    self.HashGetter = GetBias
    return GetBias
