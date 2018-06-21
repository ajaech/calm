#!/usr/bin/env python
import argparse
import collections
import copy
import gzip
import logging
import numpy as np
import os
import pandas
import pickle
import random
import tensorflow as tf
import time

from beam import BeamItem, BeamQueue
from char2vec import MikolovEmbeddings, Char2Vec
from model import HyperModel
from vocab import Vocab
from dataset import Dataset
import helper
import metrics


parser = argparse.ArgumentParser()
parser.add_argument('expdir', help='experiment directory')
parser.add_argument('--mode', default='train',
                    choices=['train', 'debug', 'eval', 'dump', 'classify',
                             'uniclass', 'geoclass'])
parser.add_argument('--params', type=str, 
                    help='json file with hyperparameters',
                    default='default_params.json')
parser.add_argument('--vocab', type=str, help='predefined vocab', default=None)
parser.add_argument('--data', type=str, action='append', dest='data',
                    help='where to load the data')
parser.add_argument('--valdata', type=str, action='append', dest='valdata',
                    help='where to load validation data', default=[])
parser.add_argument('--reverse', type=bool, default=False)
parser.add_argument('--threads', type=int, default=12,
                    help='how many threads to use in tensorflow')
args = parser.parse_args()

if not os.path.exists(args.expdir):
  os.mkdir(args.expdir)
elif args.mode == 'train':
  print 'ERROR: expdir already exists!!!!'
  exit()

tf.set_random_seed(int(time.time() * 1000))

params = helper.GetParams(args.params, args.mode, args.expdir)
config = tf.ConfigProto(inter_op_parallelism_threads=args.threads,
                        intra_op_parallelism_threads=args.threads)

if not hasattr(params, 'context_var_types'):
  params.context_var_types = ['categorical'] * len(params.context_vars)

if args.mode != 'train':
  params.batch_size = 5
if args.mode == 'debug':
  params.batch_size = 1

SEPERATOR = ' '
if params.splitter == 'char':
  SEPERATOR = ''


if args.mode in ('train', 'eval', 'classify', 'uniclass', 'geoclass'):
  mode = args.mode

  dataset = Dataset(max_len=params.max_len + 1, 
                    preshuffle=args.mode=='train',
                    batch_size=params.batch_size)
  print 'reading data'
  dataset.ReadData(args.data, params.context_vars + ['text'],
                   splitter=params.splitter,
                   valdata=args.valdata, types=params.context_var_types)

if args.mode == 'train':
  # do the word vocab
  if args.vocab is not None:
    vocab = Vocab.Load(args.vocab)
  else:
    vocab = Vocab.MakeFromData(dataset.GetColumn('text'), min_count=params.min_vocab_count)

  if params.splitter == 'word':  # do the character vocab
    graphemes = [['{'] + Vocab.Graphemes(x) + ['}'] for x in vocab.GetWords()]
    char_vocab = Vocab.MakeFromData(graphemes, min_count=1)
    char_vocab.Save(os.path.join(args.expdir, 'char_vocab.pickle'))
  else:
    char_vocab = None

  context_vocabs = {}  # do the context vocabs
  for i, context_var in enumerate(params.context_vars):
    # skip numerical vocabularies
    if hasattr(params, 'context_var_types') and params.context_var_types[i] == 'numerical':
      context_vocabs[context_var] = None
      continue

    v = Vocab.MakeFromData([[u] for u in dataset.GetColumn(context_var)],
                           min_count=50, no_special_syms=True)
    context_vocabs[context_var] = v
    print 'num {0}: {1}'.format(context_var, len(v))
    
  vocab.Save(os.path.join(args.expdir, 'word_vocab.pickle'))
  print 'vocab size {0}'.format(len(vocab))
  with open(os.path.join(args.expdir, 'context_vocab.pickle'), 'wb') as f:
    pickle.dump(context_vocabs, f)

  dataset.Prepare(vocab, context_vocabs)

else:
  vocab = Vocab.Load(os.path.join(args.expdir, 'word_vocab.pickle'))
  if params.splitter == 'word':
    char_vocab = Vocab.Load(os.path.join(args.expdir, 'char_vocab.pickle'))
  else:
    char_vocab = None
  with open(os.path.join(args.expdir, 'context_vocab.pickle'), 'rb') as f:
    context_vocabs = pickle.load(f)


use_nce_loss = args.mode == 'train'
if len(vocab) < 5000:  # disable NCE for small vocabularies
  use_nce_loss = False
if args.mode == 'classify' and len(vocab) > 5000:
  use_nce_loss = True
  params.nce_samples = 8000

embedder = 'mikolov'
if hasattr(params, 'embedder'):
  embedder = params.embedder
if embedder == 'mikolov':
  word_embedder = MikolovEmbeddings(params, vocab)
else:
  word_embedder = Char2Vec(params, vocab, char_vocab)
model = HyperModel(
  params, vocab,  context_vocabs,
  use_nce_loss=use_nce_loss, reverse=args.reverse, exclude_unk=True, 
  word_embedder=word_embedder)

saver = tf.train.Saver(tf.global_variables())
session = tf.Session(config=config)

def GetFeedDict(batch, use_dropout=True):
  # helper function to prepare feed dict for batch
  s = np.array(list(batch.text.values))  # hacky
  feed_dict = {
    model.word_ids: s,
    model.seq_len: batch.seq_lens.values,
    model.dropout_keep_prob: params.dropout_keep_prob
  }

  if hasattr(model, 'context_placeholders'):
    for context_var in params.context_vars:
      placeholder = model.context_placeholders[context_var]
      feed_dict[placeholder] = batch[context_var].values

  if not use_dropout:
    # if dropout is removed from feed_dict then it will be turned off
    del feed_dict[model.dropout_keep_prob]

  return feed_dict                
  

def Train(expdir):
  """This function performs training."""
  logging.basicConfig(filename=os.path.join(expdir, 'logfile.txt'),
                      level=logging.INFO)
  logging.getLogger().addHandler(logging.StreamHandler())

  tvars = tf.trainable_variables()
  grads, _ = tf.clip_by_global_norm(tf.gradients(model.cost, tvars), 5.0)
  optimizer = tf.train.AdamOptimizer(0.001)
  train_op = optimizer.apply_gradients(zip(grads, tvars))

  print('initalizing')
  session.run(tf.global_variables_initializer())

  avgcost = metrics.MovingAvg(0.90)
  start_time = time.time()
  for idx in xrange(params.iters):
    batch = dataset.GetNextBatch()
    feed_dict = GetFeedDict(batch, use_dropout=True)

    cost,  _ = session.run([model.cost, train_op], feed_dict)
    c = avgcost.Update(cost)

    if idx % 40 == 0:  # every 40th batch, run one batch from the validation set
      end_time = time.time()
      time_diff = end_time - start_time
      start_time = end_time
      seconds_per_batch = time_diff / 40
      print 'seconds per batch {0}'.format(seconds_per_batch)

      feed_dict = GetFeedDict(dataset.GetValBatch(), use_dropout=False)
      val_cost = session.run(model.cost, feed_dict)

      print idx, cost
      logging.info({'iter': idx, 'cost': c, 'rawcost': cost, 'valcost': val_cost})

    if idx % 500 == 0:  # save the model every 500 minibatches
      saver.save(session, os.path.join(expdir, 'model.bin'),
                 write_meta_graph=False)


def DumpEmbeddings(expdir):
  """Dump the word embeddings to a file for offline analysis."""
  saver.restore(session, os.path.join(expdir, 'model.bin'))

  word = model.word_embedder.GetAllEmbeddings().eval(session=session)
  with gzip.open(os.path.join(expdir, 'embeddings.tsv.gz'), 'w') as f:
    for idx, w in enumerate(word):
      f.write(vocab[idx])
      f.write('\t')
      f.write('\t'.join(['{0:.3f}'.format(i) for i in w]))
      f.write('\n')


def ContextBias(expdir):
  # used for model introspection
  saver.restore(session, os.path.join(expdir, 'model.bin'))
  context_bias = tf.trainable_variables()[2].eval(session=session)
  
  for i in range(context_bias.shape[1]):
    z = context_bias[:, i]
    vals = np.argsort(z)
    print '~~~{0}~~~'.format(context_vocabs[params.context_vars[0]][i])
    topwords = ['{0} {1:.2f}'.format(vocab[i], z[i]) for i in vals[-10:]]
    bottomwords = ['{0} {1:.2f}'.format(vocab[i], z[i]) for i in vals[:10]]
    print ' '.join(reversed(topwords))
    print ' '.join(bottomwords)


def Debug(expdir):
  metrics.PrintParams()
  saver.restore(session, os.path.join(expdir, 'model.bin'))

  context_var = params.context_vars[0]
  context_vocab = context_vocabs[context_var]
  subnames = context_vocab.GetWords()

  def Process(s):
    s = np.squeeze(s.T)
    vals = s.argsort()

    topwords = ['{0} {1:.2f}'.format(vocab[vals[-1-i]], s[vals[-1-i]])
                for i in range(10)]
    print ' '.join(topwords)

  Process(model.base_bias.eval(session=session))

  if not params.use_softmax_adaptation:
    return  # nothing else to do here

  uword = word_embedder.GetAllEmbeddings()[:, :model.context_size]
  scores = tf.matmul(uword, model.final_context_embed, transpose_b=True)

  for _ in range(10):
    fd = {}
    GetRandomSetting(fd, {'offering': 'loc_124264'}, print_it=True)
    s = session.run(scores, fd)
    Process(s)
    Process(-s)
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~'


def GetRandomSetting(feed_dict=None, fixed_context={}, print_it=False):
  # choose a random context, useful for debugging
  if print_it:
    print '~' * 40
  context_settings={}
  for context_var, context_type in zip(params.context_vars, params.context_var_types):
    context_vocab = context_vocabs[context_var]
    if context_var in fixed_context:
      selection = fixed_context[context_var]
    else:
      if context_type == 'categorical':
        selection = '<UNK>'
        while selection == '<UNK>':
          selection = context_vocab[np.random.randint(len(context_vocab))]
      else:
        selection = 1.0
    context_settings[context_var] = selection
    if print_it:
      print '{0}:\t{1}'.format(context_var, selection)
  
  if feed_dict is None:
    return context_settings

  if hasattr(model, 'context_placeholders'):
    for context_var in params.context_vars:
      context_vocab = context_vocabs[context_var]
      placeholder = model.context_placeholders[context_var]
      if context_vocab:
        feed_dict[placeholder] = np.array([context_vocab[context_settings[context_var]]])
      else:
        feed_dict[placeholder] = np.array([context_settings[context_var]])


def InitBeam(phrase, settings):
  # helper function to start beam search with a prefix phrase
  prev_c = np.zeros((1, params.cell_size))
  prev_h = np.zeros((1, params.cell_size))
  for word in phrase[:-1]:
    feed_dict = {model.prev_c: prev_c, model.prev_h: prev_h,
       model.prev_word: vocab[word], model.beam_size: 4}
    GetRandomSetting(feed_dict, settings, print_it=False)
    prev_c, prev_h = session.run(
      [model.next_c, model.next_h], feed_dict)

  return prev_c, prev_h

def BeamSearch(expdir):
  saver.restore(session, os.path.join(expdir, 'model.bin'))

  context = {'rating': random.choice(['5_stars', '1_stars', '2_stars', '4_stars']), 'subreddit': '5_stars',
              'offering': random.choice(['loc_76049', 'loc_1218737', 'loc_77094', 'loc_1776857'])}
  context = {}
  settings = GetRandomSetting(None, context, print_it=True)

  # initalize beam
  starting_phrase = random.choice([
    '<S> the bed was',
    '<S> the location was',
    '<S> i stayed here on a business trip and',
    '<S> our room was',
    '<S> we could not believe',
    '<S> when i arrived at the hotel']).split()
  beam_size = 8
  total_beam_size = 200
  init_c, init_h = InitBeam(starting_phrase, settings)
  nodes = [BeamItem(starting_phrase, init_c, init_h)]

  for i in xrange(80):
    new_nodes = BeamQueue(max_size=total_beam_size)
    for node in nodes:
      if node.words[-1] == '</S>':  # don't extend past end-of-sentence token
        new_nodes.Insert(node)
        continue

      feed_dict = {
        model.prev_word: vocab[node.words[-1]],
        model.prev_c: node.prev_c,
        model.prev_h: node.prev_h,
        model.beam_size: beam_size,
      }
      GetRandomSetting(feed_dict, settings, print_it=False)

      current_word_id, current_word_p, node.prev_c, node.prev_h = session.run(
        [model.selected, model.selected_p, model.next_c, model.next_h], feed_dict)
      current_word_p = np.squeeze(current_word_p)
      if len(current_word_p.shape) == 0:
        current_word_p = [float(current_word_p)]

      for top_entry, top_value in zip(current_word_id, current_word_p):
        new_word = vocab[top_entry]
        if new_word != '<UNK>' and node.IsEligible(new_word):
          log_p = -np.log(top_value)

          # check the bound to see if we can avoid adding this to the queue
          if new_nodes.CheckBound(log_p + node.Cost()):
            new_beam = copy.deepcopy(node)
            new_beam.Update(log_p, vocab[top_entry])
            new_nodes.Insert(new_beam)
    nodes = new_nodes
  for item in reversed([b for b in nodes][-4:]):
    print item.Cost(), SEPERATOR.join(item.words)


def UnigramClassify(expdir):
  # turn the model into a linear classifier by using the softmax bias
  saver.restore(session, os.path.join(expdir, 'model.bin'))
  probs = tf.nn.softmax(model.base_bias + 
                        tf.transpose(model.bias_tables['subreddit']))
  log_probs = tf.log(probs).eval(session=session)
  
  lang_vocab = context_vocabs['subreddit']
  vocab_subset = lang_vocab.GetWords()

  print 'preparing dataset'
  dataset.Prepare(vocab, context_vocabs)

  preds = []
  labels = []
  for pos in xrange(dataset.GetNumBatches()):
    if pos % 10 == 0:
      print pos
    batch = dataset.GetNextBatch()
    
    for i in xrange(len(batch)):
      row = batch.iloc[i]
      scores = np.zeros(len(vocab_subset))

      for word_id in row.text[1:row.seq_lens]:
        scores += log_probs[:, word_id]
      preds.append(np.argmax(scores))
      labels.append(row['subreddit'])
  metrics.Metrics([lang_vocab[i] for i in preds],
                  [lang_vocab[i] for i in labels])


def GeoClassify(expdir):
  # classify tweets based on lat/long context
  saver.restore(session, os.path.join(expdir, 'model.bin'))

  print 'preparing dataset'
  dataset.Prepare(vocab, context_vocabs)

  classes = [
    {'lat': 40.4, 'lon': -3.7},  #madrid
    {'lat': 51.5, 'lon': -0.23},  # london
    {'lat': 40.7, 'lon': -74.0},  # nyc
    {'lat': 41.4, 'lon': 2.17},  # barcelona
    {'lat': 34.05, 'lon': -118.2},  # los angeles
    {'lat': 53.5, 'lon': -2.2}  # manchester
  ]
  names = ['madrid', 'london', 'nyc', 'barcelona', 'la', 'manchester']

  results = []
  all_labels = []
  all_preds = []
  for pos in xrange(dataset.GetNumBatches()):
    if pos % 10 == 0:
      print pos
    batch = dataset.GetNextBatch()
    feed_dict = GetFeedDict(batch, use_dropout=False)
    labels = []
    for lat, lon in zip(feed_dict[model.context_placeholders['lat']],
                        feed_dict[model.context_placeholders['lon']]):
      closest_dist = 300
      closest_class = -1
      for i in range(len(names)):
        d = helper.haversine(lon, lat, classes[i]['lon'], classes[i]['lat'])
        if d < closest_dist:
          closest_dist = d
          closest_class = names[i]
      labels.append(closest_class)
    labels = np.array(labels)

    costs = []
    for i in range(len(names)):
      feed_dict[model.context_placeholders['lat']][:] = classes[i]['lat']
      feed_dict[model.context_placeholders['lon']][:] = classes[i]['lon']
      costs.append(session.run(model.per_sentence_loss, feed_dict))
    costs = np.array(costs)

    predictions = np.argmin(costs, 0)
    for p, l in zip(np.squeeze(predictions), labels):
      if l != '-1':
        all_preds.append(p)
        all_labels.append(l)
  metrics.Metrics([names[i] for i in all_preds], all_labels)

def Classify(expdir):
  saver.restore(session, os.path.join(expdir, 'model.bin'))

  context_var = params.context_vars[-1]
  placeholder = model.context_placeholders[context_var]
  lang_vocab = context_vocabs[context_var]
  vocab_subset = [w for w in lang_vocab.GetWords() if w != '<UNK>']
  print vocab_subset

  print 'preparing dataset'
  dataset.Filter(vocab_subset, context_var)
  dataset.Prepare(vocab, context_vocabs)

  results = []
  all_labels = []
  all_preds = []
  for pos in xrange(dataset.GetNumBatches()):
    if pos % 10 == 0:
      print pos
    batch = dataset.GetNextBatch()
    feed_dict = GetFeedDict(batch, use_dropout=False)
    labels = np.array(feed_dict[placeholder])

    def GetCosts():
      costs = []
      if use_nce_loss:
        feed_dict[placeholder][:] = lang_vocab[vocab_subset[0]]
        result = session.run([model.per_sentence_loss] + model.sampled_values, feed_dict)
        sentence_costs, sampled_vals = result[0], result[1:]
        costs.append(sentence_costs)
        # reuse the sampled values
        for i in range(len(model.sampled_values)):
          feed_dict[model.sampled_values[i][0]] = sampled_vals[i][0]
          feed_dict[model.sampled_values[i][1]] = sampled_vals[i][1]
          feed_dict[model.sampled_values[i][2]] = sampled_vals[i][2]

        for i in range(1, len(vocab_subset)):
          feed_dict[placeholder][:] = lang_vocab[vocab_subset[i]]
          costs.append(session.run(model.per_sentence_loss, feed_dict))
      else:  # full softmax
        for i in range(len(vocab_subset)):
          feed_dict[placeholder][:] = lang_vocab[vocab_subset[i]]
          costs.append(session.run(model.per_sentence_loss, feed_dict))
        
      return np.array(costs)

    costs = GetCosts()
    for label, c_array in zip(labels, costs.T):
      d = dict(zip(vocab_subset, c_array))
      d['label'] = lang_vocab[label]
      results.append(d)

    predictions = np.argmin(costs, 0)
    all_preds += [lang_vocab[vocab_subset[x]] for x in predictions]
    all_labels += list(labels)
  
  df = pandas.DataFrame(results)
  df.to_csv(os.path.join(expdir, 'classify.csv'))
  metrics.Metrics([lang_vocab[i] for i in all_preds],
                  [lang_vocab[i] for i in all_labels])


def Eval(expdir):
  # compute the perplexity
  dataset.Prepare(vocab, context_vocabs)
  num_batches = dataset.GetNumBatches()

  print 'loading model'
  saver.restore(session, os.path.join(expdir, 'model.bin'))

  total_word_count = 0
  total_log_prob = 0
  results = []
  for pos in xrange(dataset.GetNumBatches()):
    batch = dataset.GetNextBatch()
    feed_dict = GetFeedDict(batch, use_dropout=False)
    lens = feed_dict[model.seq_len]

    cost, sentence_costs = session.run([model.cost, model.per_sentence_loss],
                                       feed_dict)

    for length, idx, sentence_cost in zip(lens, batch.index, sentence_costs):
      batch_row = batch.loc[idx]
      data_row = {'length': length, 'cost': sentence_cost}
      for context_var in params.context_vars:
        if context_vocabs[context_var]:
          data_row[context_var] = batch_row['orig_{0}'.format(context_var)]
        else:
          data_row[context_var] = batch_row[context_var]
      results.append(data_row)

    words_in_batch = sum(lens - 1)
    total_word_count += words_in_batch
    total_log_prob += float(cost * words_in_batch)
    print '{0}\t{1:.3f}'.format(pos, np.exp(total_log_prob / total_word_count))

  results = pandas.DataFrame(results)
  results.to_csv(os.path.join(expdir, 'pplstats.csv.gz'), compression='gzip')
  

def TopNextProbs(expdir):
  saver.restore(session, os.path.join(expdir, 'model.bin'))

  words = '<S> the'.split()
  prevstate_h = np.zeros((1, params.cell_size))
  prevstate_c = np.zeros((1, params.cell_size))

  fixed_context = {'rating': 'five'}
  fixed_context = GetRandomSetting(None, fixed_context, print_it=True)

  for current_word in words:
    feed_dict = {
      model.prev_word: vocab[current_word],
      model.prev_c: prevstate_c,
      model.prev_h: prevstate_h
    }
    GetRandomSetting(feed_dict, fixed_context)

    next_prob, prevstate_c, prevstate_h = session.run(
      [model.next_prob, model.next_c, model.next_h], feed_dict)
  next_prob = np.squeeze(next_prob)
  idx = np.argsort(-next_prob)
  for ii in idx[:10]:
    print '{0}\t{1:.2f}%'.format(vocab[ii], next_prob[ii] * 100.0)


def Process(fixed_context, greedy=True):
  fixed_context = GetRandomSetting(None, fixed_context, print_it=True)
  session.run(model.reset_state)
  current_word = '<S>'

  words = []
  log_probs = []
  words.append(current_word)
  feed_dict = {                      
    model.prev_word: vocab[current_word],
    model.temperature: [0.6]
  }
  GetRandomSetting(feed_dict, fixed_context)
  for i in range(50):
    current_word_id, current_word_p = session.run(
      [model.selected, model.selected_p], feed_dict)
    log_probs.append(-np.log(current_word_p))
    current_word = vocab[current_word_id[0][0]]
    words.append(current_word)
  
    if current_word == '</S>':
      break
    feed_dict[model.prev_word] = vocab[current_word]

  ppl = np.exp(np.mean(log_probs))
  return ppl, SEPERATOR.join(words)


def Greedy(expdir):
  saver.restore(session, os.path.join(expdir, 'model.bin'))
    
  for idx in range(20000):
    ppl, sentence = Process({}, greedy=True)
    print '{0:.3f}\t{1}'.format(ppl, sentence)

if args.mode == 'train':
  Train(args.expdir)

if args.mode == 'eval':
  Eval(args.expdir)

if args.mode == 'classify':
  Classify(args.expdir)
if args.mode == 'uniclass':
  UnigramClassify(args.expdir)
if args.mode == 'geoclass':
  GeoClassify(args.expdir)

if args.mode == 'debug':

  session.run([model.prev_c.initializer, model.prev_h.initializer])

  #ContextBias(args.expdir)
  #Debug(args.expdir)
  for _ in range(500):
    BeamSearch(args.expdir)
  
  #TopNextProbs(args.expdir)
  #Greedy(args.expdir)
  
if args.mode == 'dump':
  DumpEmbeddings(args.expdir)
