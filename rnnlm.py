#!/usr/bin/env python
import argparse
import bunch
import collections
import gzip
import json
import logging
import numpy as np
import os
import pandas
import pickle
import tensorflow as tf

from model import HyperModel, PrintParams
from vocab import Vocab
from batcher import Dataset

import code

parser = argparse.ArgumentParser()
parser.add_argument('expdir')
parser.add_argument('--mode', default='train',
                    choices=['train', 'debug', 'eval', 'dump', 'classify'])
parser.add_argument('--params', type=argparse.FileType('r'), 
                    default='default_params.json')
parser.add_argument('--data', type=str, 
                    default='/n/falcon/s0/ajaech/reddit.tsv.bz2')
parser.add_argument('--partition_override', type=bool, default=False)
parser.add_argument('--threads', type=int, default=12)
args = parser.parse_args()

if not os.path.exists(args.expdir):
  os.mkdir(args.expdir)

param_filename = os.path.join(args.expdir, 'params.json')
if args.mode == 'train':
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

if args.mode == 'train':
  
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

  for idx in xrange(params.iters):
    batch = dataset.GetNextBatch()
    feed_dict = GetFeedDict(batch)

    a = session.run([model.cost, train_op], feed_dict)

    if idx % 50 == 0:
      print idx, float(a[0])
      logging.info({'iter': idx, 'cost': float(a[0])})

      if idx % 1000 == 0:
        saver.save(session, os.path.join(expdir, 'model.bin'))


def DumpEmbeddings(expdir):
  saver.restore(session, os.path.join(expdir, 'model.bin'))

  word = model._word_embeddings.eval(session=session)
  user = model._user_embeddings.eval(session=session)

  with gzip.open(os.path.join(expdir, 'embeddings.tsv.gz'), 'w') as f:
    for w in word:
      f.write('\t'.join(['{0:.3f}'.format(i) for i in w]))
      f.write('\n')

  with gzip.open(os.path.join(expdir, 'userembeddings.tsv.gz'), 'w') as f:
    for w in user:
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
    
  base_bias = session.run(model.base_bias)
  Process(base_bias.T, 'Baseline')

  uword = model._word_embeddings[:, :params.user_embedding_size]
  subreddit = tf.placeholder(tf.int32, ())
  scores = tf.matmul(uword, tf.expand_dims(model._user_embeddings[subreddit, :], 1))

  for subname in ['exmormon', 'AskMen', 'AskWomen', 'Music', 'aww',
                  'dogs', 'cats', 'worldnews', 'tifu', 'books', 'WTF',
                  'RealGirls', 'relationships', 'Android']:
    s = session.run(scores, {subreddit: username_vocab[subname]})
    Process(s, subname)


def Greedy(expdir):
  saver.restore(session, os.path.join(expdir, 'model.bin'))

  def Process(subname):
    current_word = '<S>'
    prevstate_h = np.zeros((1, params.cell_size))
    prevstate_c = np.zeros((1, params.cell_size))

    words = []
    log_probs = []
    for i in xrange(50):
      feed_dict = {                      
        model.prev_word: vocab[current_word],
        model.prev_c: prevstate_c,
        model.prev_h: prevstate_h,
      }
      for context_var in params.context_vars:
        placeholder = model.context_placeholders[context_var]
        feed_dict[placeholder] = np.array([0])

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
                 'GoneWild', 'tifu', 'WTF', 'AskHistorians', 'hockey']
  sample_list = ['en', 'fr', 'pt', 'es', 'und']

  for n in sample_list:
    print '~~~{0}~~~'.format(n)
    for _ in range(5):
      ppl, sentence = Process(n)
      print '{0:.2f}\t{1}'.format(ppl, sentence)


def GetText(s, seq_len):
  ws = [vocab[s[0, i]] for i in range(seq_len)]
  return ' '.join(ws)


def Classify(expdir):
  dataset.Prepare(vocab, context_vocabs)

  print 'loading model'
  saver.restore(session, os.path.join(expdir, 'model.bin'))


  all_labels = []
  all_preds = []
  for pos in xrange(min(dataset.GetNumBatches(), 400)):
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

    predictions = np.argmin(np.array(costs), 0)
    all_preds += list(predictions)
    all_labels += list(labels)
  Metrics([lang_vocab[i] for i in all_preds],
          [lang_vocab[i] for i in all_labels])


def Eval(expdir):
  dataset.Prepare(vocab, context_vocabs)

  print 'loading model'
  saver.restore(session, os.path.join(expdir, 'model.bin'))

  total_word_count = 0
  total_log_prob = 0
  results = []
  for pos in xrange(min(dataset.GetNumBatches(), 400)):
    batch = dataset.GetNextBatch()
    feed_dict = GetFeedDict(batch)

    cost, sentence_costs = session.run([model.cost, model.per_sentence_loss],
                                       feed_dict)

    lens = feed_dict[model.seq_len]
    unames = feed_dict[model.context_placeholders['lang']]

    for length, uname, sentence_cost in zip(lens, unames, sentence_costs):
      results.append({'length': length, 'uname': context_vocabs['lang'][uname],
                      'cost': sentence_cost})

    seq_len = feed_dict[model.seq_len]
    total_word_count += sum(seq_len)
    total_log_prob += float(cost * sum(seq_len))
    ppl = np.exp(total_log_prob / total_word_count)
    print '{0}\t{1:.3f}'.format(pos, ppl)
  
  results = pandas.DataFrame(results)
  results.to_csv(os.path.join(expdir, 'pplstats.csv.gz'), compression='gzip')


def Metrics(preds, labs, show=True):
  """Print precision, recall and F1 for each language.
  Assumes a single language per example, i.e. no code switching.
  Args:
    preds: list of predictions
    labs: list of labels
    show: flag to toggle printing
  """
  all_langs = set(preds + labs)
  preds = np.array(preds)
  labs = np.array(labs)
  label_totals = collections.Counter(labs)
  pred_totals = collections.Counter(preds)
  confusion_matrix = collections.Counter(zip(preds, labs))
  num_correct = 0
  for lang in all_langs:
    num_correct += confusion_matrix[(lang, lang)]
  acc = num_correct / float(len(preds))
  print 'accuracy = {0:.3f}'.format(acc)
  if show:
    print ' Lang     Prec.   Rec.   F1'
    print '------------------------------'
  scores = []
  fmt_str = '  {0:6}  {1:6.2f} {2:6.2f} {3:6.2f}'
  for lang in sorted(all_langs):
    idx = preds == lang
    total = max(1.0, pred_totals[lang])
    precision = 100.0 * confusion_matrix[(lang, lang)] / total
    idx = labs == lang
    total = max(1.0, label_totals[lang])
    recall = 100.0 * confusion_matrix[(lang, lang)] / total
    if precision + recall == 0.0:
      f1 = 0.0
    else:
      f1 = 2.0 * precision * recall / (precision + recall)
    scores.append([precision, recall, f1])
    if show:
      print fmt_str.format(lang, precision, recall, f1)
  totals = np.array(scores).mean(axis=0)
  if show:
    print '------------------------------'
    print fmt_str.format('Total:', totals[0], totals[1], totals[2])
  return totals[2]


if args.mode == 'train':
  Train(args.expdir)

if args.mode == 'eval':
  Eval(args.expdir)

if args.mode == 'classify':
  Classify(args.expdir)

if args.mode == 'debug':
  #Debug(args.expdir)
  Greedy(args.expdir)

if args.mode == 'dump':
  DumpEmbeddings(args.expdir)
