import argparse
import code
import collections
import bz2
import numpy as np
import os
import random
import tahiti
import tensorflow as tf

from vocab import Vocab
from batcher import Dataset
from model import MultiModel, BiasModel


parser = argparse.ArgumentParser()
parser.add_argument('expdir')
parser.add_argument('--mode', choices=['train', 'debug', 'eval'],
                    default='train')
parser.add_argument('--model', choices=['multi', 'bias'],
                    default='multi')
args = parser.parse_args()

if not os.path.exists(args.expdir):
  os.mkdir(args.expdir)

config = tf.ConfigProto(inter_op_parallelism_threads=10,
                        intra_op_parallelism_threads=10)

def ReadData(filename):
  subreddits = []
  years = []
  texts = []

  with bz2.BZ2File(filename, 'r') as f:
    for line in f:
      idnum, subreddit, year, text = line.split('\t')

      if args.mode == 'train' and int(idnum) % 10 < 3:
        continue
      if args.mode != 'train' and int(idnum) % 10 > 3:
        continue

      subreddits.append(subreddit)
      years.append(year)
      texts.append(['<S>'] + text.lower().split() + ['</S>'])

  return subreddits, years, texts

logger = tahiti.Client(args.expdir, args.expdir)

max_len = 36
if args.mode != 'debug':
  dataset = Dataset(max_len=max_len)
  subreddits, years, texts = ReadData('/s0/ajaech/reddit.tsv.bz')
  dataset.AddDataSource(subreddits, years, texts)

if args.mode == 'train':
  vocab = Vocab.MakeFromData(texts, min_count=60)
  logger.LogInfo('Input Vocab Size: {0}'.format(len(vocab)))
  subreddit_vocab = Vocab.MakeFromData([[s] for s in subreddits],
                                       min_count=1, no_special_syms=True)
  year_vocab = Vocab.MakeFromData([[y] for y in years], min_count=1,
                                  no_special_syms=True)
  vocab.Save(os.path.join(args.expdir, 'word_vocab.pickle'))
  subreddit_vocab.Save(os.path.join(args.expdir, 'subreddit_vocab.pickle'))
  year_vocab.Save(os.path.join(args.expdir, 'year_vocab.pickle'))
else:
  vocab = Vocab.Load(os.path.join(args.expdir, 'word_vocab.pickle'))
  subreddit_vocab = Vocab.Load(os.path.join(args.expdir, 'subreddit_vocab.pickle'))
  year_vocab = Vocab.Load(os.path.join(args.expdir, 'year_vocab.pickle'))

if args.mode != 'debug':
  dataset.Prepare(vocab, subreddit_vocab, year_vocab)

models = {'multi': MultiModel, 'bias': BiasModel}
model = models[args.model](max_len-1, len(vocab), len(subreddit_vocab),
                           len(year_vocab))

saver = tf.train.Saver(tf.all_variables())
session = tf.Session(config=config)

def Train(expdir):
  tvars = tf.trainable_variables()
  grads, _ = tf.clip_by_global_norm(tf.gradients(model.cost, tvars), 5.0)
  optimizer = tf.train.AdamOptimizer(0.001)
  train_op = optimizer.apply_gradients(zip(grads, tvars))

  session.run(tf.initialize_all_variables())

  for idx in xrange(100000):
    s, seq_len, subreddit, year = dataset.GetNextBatch()

    a = session.run([model.cost, train_op], {
        model.x: s[:, :-1],
        model.y: s[:, 1:],
        model.seq_len: seq_len,
        model.year: year, 
        model.subreddit: subreddit})
  
    if idx % 25 == 0:
      ws = [vocab.idx_to_word[s[0, i]] for i in range(seq_len)]
      print subreddit_vocab.idx_to_word[subreddit[0]]
      print ' '.join(ws)
      print float(a[0])
      print '-------'

      logger.LogData({'iter': idx, 'cost': float(a[0])})

      if idx % 5000 == 0:
        saver.save(session, os.path.join(expdir, 'model.bin'))


def Greedy(expdir):
  saver.restore(session, os.path.join(expdir, 'model.bin'))

  current_word = '<S>'
  prevstate = np.zeros((1, 280))

  for i in xrange(10):

    a = session.run([model.next_word_prob, model.hidden_out],
                    {model.wordid: np.array([vocab[current_word]]),
                     model.year: np.array([0]),
                     model.subreddit: np.array([1]),
                     model.prevstate: prevstate
                     })
    probs, prevstate = a
    current_word_id = np.argmax(probs)
    current_word = vocab.idx_to_word[current_word_id]
    print current_word
    

def GetText(s, seq_len):
  ws = [vocab.idx_to_word[s[0, i]] for i in range(seq_len)]
  return ' '.join(ws)

def Eval(expdir):
  saver.restore(session, os.path.join(expdir, 'model.bin'))

  word_counts = collections.defaultdict(int)
  cost_counts = collections.defaultdict(int)

  for pos in xrange(1000):
    s, seq_len, subreddit, year = dataset.GetNextBatch()

    a = session.run([model.cost], {
        model.x: s[:, :-1],
        model.y: s[:, 1:],
        model.seq_len: seq_len,
        model.year: year, 
        model.subreddit: subreddit})

    rows.append({'text': GetText(s, seq_len),
                 'year': int(year),
                 'subreddit': int(subreddit),
                 'seq_len': int(seq_len),
                 'cost': float(a[0])})

  code.interact(local=locals())

if args.mode == 'train':
  Train(args.expdir)

if args.mode == 'eval':
  Eval(args.expdir)

if args.mode == 'debug':
  Greedy(args.expdir)
