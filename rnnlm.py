#!/usr/bin/env python
import argparse
import bunch
import collections
import gzip
import json
import logging
import random
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

if args.mode in ('train', 'eval'):
  filename = '/n/falcon/s0/ajaech/reddit.tsv.bz2'
  usernames, texts = ReadData(filename, mode=args.mode)
  dataset = Dataset(max_len=params.max_len + 1, 
                    preshuffle=args.mode=='train',
                    batch_size=params.batch_size)
  dataset.AddDataSource(usernames, texts)

if args.mode == 'train':
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


model = HyperModel(params, len(vocab), len(username_vocab), 
                   use_nce_loss=args.mode == 'train')

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

  for idx in xrange(90000):
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
      ws = [vocab.idx_to_word[s[0, i]] for i in range(seq_len[0])]
      print ' '.join(ws)
      print float(a[0])
      print '-------'
      logging.info({'iter': idx, 'cost': float(a[0])})

      if idx % 1000 == 0:
        saver.save(session, os.path.join(expdir, 'model.bin'))


def DumpEmbeddings(expdir):
  saver.restore(session, os.path.join(expdir, 'model.bin'))

  word = model._word_embeddings.eval(session=session)
  user = model._user_embeddings.eval(session=session)
  v = vocab
  uv = username_vocab

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

  uword = model._word_embeddings[:, :params.user_embedding_size]
  uu = model._user_embeddings

  subreddit = tf.placeholder(tf.int32, ())

  scores = tf.matmul(uword, tf.expand_dims(uu[subreddit, :], 1))

  def Process(subname):
    s = session.run(scores, {subreddit: username_vocab[subname]})
    vals = np.squeeze(s.T).argsort()

    print '~~~{0}~~~'.format(subname)
    topwords = ['{0} {1:.2f}'.format(vocab[vals[-1-i]], s[vals[-1-i]][0])
                for i in range(10)]
    print ' '.join(topwords)

  for subname in ['exmormon', 'AskMen', 'AskWomen', 'Music', 'aww',
                  'dogs', 'cats', 'worldnews', 'tifu', 'books', 'WTF',
                  'RealGirls', 'relationships', 'Android']:
    Process(subname)


def Greedy(expdir):
  saver.restore(session, os.path.join(expdir, 'model.bin'))

  def Process(subname):
    current_word = '<S>'
    prevstate_h = np.zeros((1, params.cell_size))
    prevstate_c = np.zeros((1, params.cell_size))

    words = []
    for i in xrange(10):
      a = session.run([model.next_idx, model.next_prob, model.next_c, model.next_h],
                      {
                        model.username: np.array([username_vocab[subname]]),
                        model.prev_word: vocab[current_word],
                        model.prev_c: prevstate_c,
                        model.prev_h: prevstate_h
                      })
      current_word_id, current_prob, prevstate_h, prevstate_c = a
      current_word_id = current_word_id[0]
      q = np.random.rand()
      cumulative = np.cumsum(current_prob)
      current_word_id = np.argmin((cumulative < q))
      current_word = vocab.idx_to_word[current_word_id]
      words.append(current_word)
      if current_word == '</S>':
        break
    print '\t' + ' '.join(words)

  for n in ['AskWomen', 'AskMen', 'exmormon', 'Music', 'worldnews', 'AskReddit',
            'GoneWild', 'tifu', 'WTF', 'AskHistorians', 'nfl']:
    print '~~~{0}~~~'.format(n)
    for _ in range(3):
      Process(n)


def GetText(s, seq_len):
  ws = [vocab.idx_to_word[s[0, i]] for i in range(seq_len)]
  return ' '.join(ws)

def Eval(expdir):
  dataset.Prepare(vocab, username_vocab)

  print 'loading model'
  saver.restore(session, os.path.join(expdir, 'model.bin'))

  total_word_count = 0
  total_log_prob = 0
  results = []
  for pos in xrange(min(dataset.GetNumBatches(), 2000)):
    s, seq_len, usernames = dataset.GetNextBatch()

    feed_dict = {
        model.x: s[:, :-1],
        model.y: s[:, 1:],
        model.seq_len: seq_len,
        model.username: usernames
    }

    cost, sentence_costs = session.run([model.cost, model.per_sentence_loss],
                                       feed_dict)

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
  Greedy(args.expdir)

if args.mode == 'dump':
  DumpEmbeddings(args.expdir)
