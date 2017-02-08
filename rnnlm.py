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
import tensorflow as tf

from model import HyperModel, PrintParams
from vocab import Vocab
from batcher import Dataset, ReadData

import code

parser = argparse.ArgumentParser()
parser.add_argument('expdir')
parser.add_argument('--mode', default='train',
                    choices=['train', 'debug', 'eval', 'dump'])
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

if args.mode in ('train', 'eval'):
  mode = args.mode
  if args.partition_override:
    mode = 'all'
  splitter = 'word'
  if hasattr(params, 'splitter'):
    splitter = params.splitter
  usernames, texts = ReadData(args.data, mode=mode, splitter=params.splitter)
  dataset = Dataset(max_len=params.max_len + 1, 
                    preshuffle=args.mode=='train',
                    batch_size=params.batch_size)
  dataset.AddDataSource(usernames, texts)

if args.mode == 'train':
  if hasattr(params, 'vocab'):
    vocab = Vocab.Load(params.vocab)
  else:
    vocab = Vocab.MakeFromData(texts, min_count=20)
  username_vocab = Vocab.MakeFromData([[u] for u in usernames],
                                      min_count=50)
  vocab.Save(os.path.join(args.expdir, 'word_vocab.pickle'))
  username_vocab.Save(os.path.join(args.expdir, 'username_vocab.pickle'))
  print 'num users {0}'.format(len(username_vocab))
  print 'vocab size {0}'.format(len(vocab))
else:
  vocab = Vocab.Load(os.path.join(args.expdir, 'word_vocab.pickle'))
  username_vocab = Vocab.Load(os.path.join(args.expdir, 'username_vocab.pickle'))


unigram_probs = vocab.GetUnigramProbs()
use_nce_loss = args.mode == 'train'
if len(unigram_probs) < 5000:
  use_nce_loss = False
model = HyperModel(params, unigram_probs, len(username_vocab), 
                   use_nce_loss=use_nce_loss)

saver = tf.train.Saver(tf.all_variables())
session = tf.Session(config=config)


def Train(expdir):
  dataset.Prepare(vocab, username_vocab)

  logging.basicConfig(filename=os.path.join(expdir, 'logfile.txt'),
                      level=logging.INFO)
  tvars = tf.trainable_variables()
  grads, _ = tf.clip_by_global_norm(tf.gradients(model.cost, tvars), 5.0)
  optimizer = tf.train.AdamOptimizer(params.learning_rate)
  train_op = optimizer.apply_gradients(zip(grads, tvars))

  print('initalizing')
  session.run(tf.initialize_all_variables())

  for idx in xrange(params.iters):
    s, seq_len, usernames = dataset.GetNextBatch()

    feed_dict = {
      model.x: s[:, :-1],
      model.y: s[:, 1:],
      model.seq_len: seq_len,
      model.username: usernames,
      model.dropout_keep_prob: params.dropout_keep_prob
    }

    a = session.run([model.cost, train_op], feed_dict)

    if idx % 25 == 0:
      ws = [vocab[s[0, i]] for i in range(seq_len[0])]
      print SEPERATOR.join(ws)
      print float(a[0])
      print '-------'
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
      a = session.run([model.next_prob, model.next_c, model.next_h],
                      {
                        model.username: np.array([username_vocab[subname]]),
                        model.prev_word: vocab[current_word],
                        model.prev_c: prevstate_c,
                        model.prev_h: prevstate_h,
                      })
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

def Eval(expdir):
  dataset.Prepare(vocab, username_vocab)

  print 'loading model'
  saver.restore(session, os.path.join(expdir, 'model.bin'))

  total_word_count = 0
  total_log_prob = 0
  results = []
  for pos in xrange(min(dataset.GetNumBatches(), 400)):
    s, seq_len, usernames = dataset.GetNextBatch()

    feed_dict = {
        model.x: s[:, :-1],
        model.y: s[:, 1:],
        model.seq_len: seq_len,
        model.username: usernames
    }

    cost, sentence_costs = session.run([model.cost, model.per_sentence_loss],
                                       feed_dict)

    z = session.run(model.per_word_loss, feed_dict)
    word_ids = feed_dict[model.y]

    lens = feed_dict[model.seq_len]
    unames = feed_dict[model.username]

    for length, uname, sentence_cost in zip(lens, unames, sentence_costs):
      results.append({'length': length, 'uname': username_vocab[uname],
                      'cost': sentence_cost})

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

if args.mode == 'debug':
  Debug(args.expdir)
  Greedy(args.expdir)

if args.mode == 'dump':
  DumpEmbeddings(args.expdir)
