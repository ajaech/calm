#!/usr/bin/env python
import argparse
import bunch
import collections
import copy
import gzip
import json
import logging
import numpy as np
import os
import pandas
import pickle
import shutil
import sys
import tensorflow as tf
import time

from model import HyperModel, PrintParams
from vocab import Vocab
from batcher import Dataset
import metrics

import code


parser = argparse.ArgumentParser()
parser.add_argument('expdir')
parser.add_argument('--mode', default='train',
                    choices=['train', 'debug', 'eval', 'dump', 'classify'])
parser.add_argument('--params', type=argparse.FileType('r'), 
                    default='default_params.json')
parser.add_argument('--data', type=str, 
                    default='/n/falcon/s0/ajaech/reddit.tsv.bz2')
parser.add_argument('--partition_override', type=bool, default=False,
                    help='use to skip train/test partitioning')
parser.add_argument('--threads', type=int, default=12)
parser.add_argument('--initialize', default=None)
args = parser.parse_args()

if args.initialize is not None and args.mode != 'train':
  sys.stderr.write('ERROR: dont use initialize arg when mode is not train.\n')
  exit(-1)

if not os.path.exists(args.expdir):
  os.mkdir(args.expdir)

param_filename = os.path.join(args.expdir, 'params.json')
if args.initialize:  # copy over vocab & param files
  param_filename = os.path.join(args.initialize, 'params.json')
  shutil.copyfile(os.path.join(args.initialize, 'word_vocab.pickle'),
                  os.path.join(args.expdir, 'word_vocab.pickle'))
  shutil.copyfile(os.path.join(args.initialize, 'context_vocab.pickle'),
                  os.path.join(args.expdir, 'context_vocab.pickle'))
  shutil.copyfile(os.path.join(args.initialize, 'params.json'),
                  os.path.join(args.expdir, 'params.json'))

if args.mode == 'train' and args.initialize is None:
  param_dict = json.load(args.params)
  params = bunch.Bunch(param_dict)
  with open(param_filename, 'w') as f:
    json.dump(param_dict, f)
else:
  with open(param_filename, 'r') as f:
    params = bunch.Bunch(json.load(f))
args.params.close()

config = tf.ConfigProto(inter_op_parallelism_threads=args.threads,
                        intra_op_parallelism_threads=args.threads)

batch_size = params.batch_size
if args.mode != 'train':
  params.batch_size = 20

SEPERATOR = ' '
if hasattr(params, 'splitter') and params.splitter == 'char':
  SEPERATOR = ''

if args.mode in ('train', 'eval', 'classify'):
  mode = args.mode
  if args.partition_override:
    mode = 'all'
  splitter = 'word'
  if hasattr(params, 'splitter'):
    splitter = params.splitter

  dataset = Dataset(max_len=params.max_len + 1, 
                    preshuffle=args.mode=='train',
                    batch_size=params.batch_size)
  dataset.ReadData(args.data, params.context_vars + ['text'],
                   mode=mode, splitter=splitter)

if args.mode == 'train' and args.initialize is None:
  vocab = Vocab.MakeFromData(dataset.GetColumn('text'), min_count=20)
  context_vocabs = {}
  for context_var in params.context_vars:
    v = Vocab.MakeFromData([[u] for u in dataset.GetColumn(context_var)],
                           min_count=50, no_special_syms=True)
    context_vocabs[context_var] = v
    print 'num {0}: {1}'.format(context_var, len(v))
    
  vocab.Save(os.path.join(args.expdir, 'word_vocab.pickle'))
  print 'vocab size {0}'.format(len(vocab))
  with open(os.path.join(args.expdir, 'context_vocab.pickle'), 'wb') as f:
    pickle.dump(context_vocabs, f)
else:
  vocab = Vocab.Load(os.path.join(args.expdir, 'word_vocab.pickle'))
  with open(os.path.join(args.expdir, 'context_vocab.pickle'), 'rb') as f:
    context_vocabs = pickle.load(f)


unigram_probs = vocab.GetUnigramProbs()
use_nce_loss = args.mode == 'train'
if len(unigram_probs) < 5000:  # disable NCE for small vocabularies
  use_nce_loss = False
if args.mode == 'eval':
  use_nce_loss = True
  params.nce_samples = 800
model = HyperModel(
  params, unigram_probs,
  [len(context_vocabs[v]) for v in params.context_vars], 
  use_nce_loss=use_nce_loss)

saver = tf.train.Saver(tf.all_variables())
session = tf.Session(config=config)

def GetFeedDict(batch, use_dropout=True):
  s = np.array(list(batch.text.values))  # hacky
  feed_dict = {
    model.x: s[:, :-1],
    model.y: s[:, 1:],
    model.seq_len: batch.seq_lens.values,
    model.dropout_keep_prob: params.dropout_keep_prob
  }

  if hasattr(model, 'context_placeholders'):
    for context_var in params.context_vars:
      placeholder = model.context_placeholders[context_var]
      feed_dict[placeholder] = batch[context_var].values

  if not use_dropout:
    del feed_dict[model.dropout_keep_prob]

  return feed_dict                
  

def Train(expdir):
  dataset.Prepare(vocab, context_vocabs)

  logging.basicConfig(filename=os.path.join(expdir, 'logfile.txt'),
                      level=logging.INFO)
  tvars = tf.trainable_variables()
  grads, _ = tf.clip_by_global_norm(tf.gradients(model.cost, tvars), 5.0)
  optimizer = tf.train.AdamOptimizer(params.learning_rate)
  train_op = optimizer.apply_gradients(zip(grads, tvars))

  print('initalizing')
  session.run(tf.initialize_all_variables())

  if args.initialize:
    saver.restore(session, os.path.join(args.initialize, 'model.bin'))

  avgcost = metrics.MovingAvg(0.90)
  start_time = time.time()
  for idx in xrange(params.iters):
    batch = dataset.GetNextBatch()
    feed_dict = GetFeedDict(batch, use_dropout=True)

    cost, _ = session.run([model.cost, train_op], feed_dict)

    if idx % 40 == 0:
      end_time = time.time()
      time_diff = end_time - start_time
      start_time = end_time
      batches_per_second = time_diff / 40
      print 'batches per second {0}'.format(batches_per_second)

      feed_dict = GetFeedDict(dataset.GetValBatch(), use_dropout=False)
      val_cost = session.run(model.cost, feed_dict)
      print idx, cost
      logging.info({'iter': idx, 'cost': avgcost.Update(cost),
                    'rawcost': cost, 'valcost': val_cost})

      if idx % 1000 == 0:
        saver.save(session, os.path.join(expdir, 'model.bin'))


def DumpEmbeddings(expdir):
  saver.restore(session, os.path.join(expdir, 'model.bin'))

  word = model._word_embeddings.eval(session=session)

  with gzip.open(os.path.join(expdir, 'embeddings.tsv.gz'), 'w') as f:
    for w in word:
      f.write('\t'.join(['{0:.3f}'.format(i) for i in w]))
      f.write('\n')


def Debug(expdir):
  saver.restore(session, os.path.join(expdir, 'model.bin'))
  PrintParams()

  def Process(s, subname):
    s = np.squeeze(s.T)
    vals = s.argsort()

    print '~~~{0}~~~'.format(subname)
    topwords = ['{0} {1:.2f}'.format(vocab[vals[-1-i]], s[vals[-1-i]])
                for i in range(10)]
    print ' '.join(topwords)
    
  #subnames = ['en', 'es', 'pt', 'de', 'it']
  subnames = ['exmormon', 'askwomen', 'todayilearned', 'nfl', 'pics', 'videos', 'worldnews']
  for subname in subnames:
    s = session.run(model.adapted_bias, {model.context_placeholders['subreddit']:
                                         np.expand_dims(context_vocabs['subreddit'][subname], 0)})
    Process(s, subname)


class BeamItem(object):
  
  def __init__(self, prev_word, prev_c, prev_h):
    self.log_probs = [0.0]
    self.words = [prev_word]
    self.prev_c = prev_c
    self.prev_h = prev_h

  def Update(self, log_prob, new_word, new_c, new_h):
    self.prev_c = new_c
    self.prev_h = new_h
    self.words.append(new_word)
    self.log_probs.append(log_prob)

  def Cost(self):
    return sum(self.log_probs)


def BeamSearch(expdir):
  saver.restore(session, os.path.join(expdir, 'model.bin'))

  varname = 'subreddit'
  subname = 'nfl'

  beam_size = 12
  beam_items = []

  # initalize beam
  beam_items.append(
    BeamItem('<S>', 
             np.zeros((1, params.cell_size)), 
             np.zeros((1, params.cell_size))))

  for i in xrange(6):
    new_beam_items = []
    for beam_item in beam_items:
      feed_dict = {
        model.prev_word: vocab[beam_item.words[-1]],
        model.prev_c: beam_item.prev_c,
        model.prev_h: beam_item.prev_h
      }
      if hasattr(model, 'context_placeholders'):
        for context_var in params.context_vars:
          placeholder = model.context_placeholders[context_var]
          if context_var == varname:
            feed_dict[placeholder] = np.array([context_vocabs[context_var][subname]])
          else:
            feed_dict[placeholder] = np.array([0])

      a = session.run([model.next_prob, model.next_c, model.next_h], feed_dict)
      current_prob, prevstate_h, prevstate_c = a
      top_entries = np.argsort(current_prob)[0, -beam_size:]
      top_values = current_prob[0, top_entries]
      for top_entry, top_value in zip(top_entries, top_values):
        new_beam = copy.deepcopy(beam_item)
        new_word = vocab[top_entry]
        if new_word != '<UNK>':
          new_beam.Update(-np.log(top_value), vocab[top_entry], prevstate_c, prevstate_h)
          new_beam_items.append(new_beam)
      new_beam_items = sorted(new_beam_items, key=lambda x: x.Cost())
      beam_items = new_beam_items[:beam_size]
  for item in beam_items:
    print item.Cost(), ' '.join(item.words)


def Greedy(expdir):
  saver.restore(session, os.path.join(expdir, 'model.bin'))

  def Process(varname, subname):
    current_word = '<S>'
    prevstate_h = np.zeros((1, params.cell_size))
    prevstate_c = np.zeros((1, params.cell_size))

    words = []
    log_probs = []
    for i in xrange(15):
      feed_dict = {                      
        model.prev_word: vocab[current_word],
        model.prev_c: prevstate_c,
        model.prev_h: prevstate_h,
      }
      if hasattr(model, 'context_placeholders'):
        for context_var in params.context_vars:
          placeholder = model.context_placeholders[context_var]
          if context_var == varname:
            feed_dict[placeholder] = np.array([context_vocabs[context_var][subname]])
          else:
            feed_dict[placeholder] = np.array([1])

      a = session.run([model.next_prob, model.next_c, model.next_h], feed_dict)
      current_prob, prevstate_h, prevstate_c = a
      cumulative = np.cumsum(current_prob)
      current_word_id = np.argmin(cumulative < np.random.rand())
      log_probs.append(-np.log(current_prob[0, current_word_id]))
      current_word = vocab[current_word_id]
      words.append(current_word)
      if '</S>' in current_word:
        break
    ppl = np.exp(np.mean(log_probs))
    return ppl, SEPERATOR.join(words)
    
  sample_list = ['AskWomen', 'AskMen', 'exmormon', 'Music', 'worldnews',
                 'tifu', 'WTF', 'AskHistorians', 'hockey']
  #sample_list = ['en', 'fr', 'pt', 'es', 'eu']

  for n in sample_list:
    print '~~~{0}~~~'.format(n)
    for _ in range(5):
      ppl, sentence = Process('subreddit', n)
      print '{0:.2f}\t{1}'.format(ppl, sentence)


def GetText(s, seq_len):
  ws = [vocab[s[0, i]] for i in range(seq_len)]
  return ' '.join(ws)


def Classify(expdir):
  dataset.Prepare(vocab, context_vocabs)

  print 'loading model'
  saver.restore(session, os.path.join(expdir, 'model.bin'))

  results = []
  all_labels = []
  all_preds = []
  for pos in xrange(dataset.GetNumBatches()):
    batch = dataset.GetNextBatch()
    feed_dict = GetFeedDict(batch, use_dropout=False)
    langs = feed_dict[model.context_placeholders['lang']]

    costs = []
    labels = np.array(langs)
    lang_vocab = context_vocabs['lang']
    for i in range(len(lang_vocab)):
      feed_dict[model.context_placeholders['lang']][:] = i
    
      sentence_costs =  session.run(model.per_sentence_loss, feed_dict)
      costs.append(sentence_costs)

    lang_names = [lang_vocab[i] for i in range(len(lang_vocab))]
    for label, c_array in zip(labels, np.array(costs).T):
      d = dict(zip(lang_names, c_array))
      d['label'] = lang_vocab[label]
      results.append(d)

    predictions = np.argmin(np.array(costs), 0)
    all_preds += list(predictions)
    all_labels += list(labels)
  df = pandas.DataFrame(results)
  df.to_csv(os.path.join(expdir, 'classify.csv'))
  metrics.Metrics([lang_vocab[i] for i in all_preds],
                  [lang_vocab[i] for i in all_labels])


def Eval(expdir):
  dataset.Prepare(vocab, context_vocabs)
  num_batches = dataset.GetNumBatches()

  print 'loading model'
  saver.restore(session, os.path.join(expdir, 'model.bin'))

  total_word_count = 0
  total_log_prob = 0
  results = []
  for pos in xrange(min(dataset.GetNumBatches(), 3000)):
    batch = dataset.GetNextBatch()
    feed_dict = GetFeedDict(batch)

    cost, sentence_costs = session.run([model.cost, model.per_sentence_loss],
                                       feed_dict)


    lens = feed_dict[model.seq_len]
    if hasattr(model, 'context_placeholders'):
      unames = feed_dict[model.context_placeholders['subreddit']]
    else: 
      unames = ['None'] * len(lens)

    for length, uname, sentence_cost in zip(lens, unames, sentence_costs):
      results.append({'length': length, 'uname': context_vocabs['subreddit'][uname],
                      'cost': sentence_cost})

    seq_len = feed_dict[model.seq_len]
    total_word_count += sum(seq_len)
    total_log_prob += float(cost * sum(seq_len))
    ppl = np.exp(total_log_prob / total_word_count)
    print '{0}\t{1:.3f}'.format(pos, ppl)
  
  results = pandas.DataFrame(results)
  results.to_csv(os.path.join(expdir, 'pplstats.csv.gz'), compression='gzip')


if args.mode == 'train':
  Train(args.expdir)

if args.mode == 'eval':
  Eval(args.expdir)

if args.mode == 'classify':
  Classify(args.expdir)

if args.mode == 'debug':
  #Debug(args.expdir)
  # Greedy(args.expdir)
  BeamSearch(args.expdir)

if args.mode == 'dump':
  DumpEmbeddings(args.expdir)
