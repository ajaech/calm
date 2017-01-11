import argparse
import bunch
import code
import collections
import json
import logging
import random
import numpy as np
import os
import tensorflow as tf
import time

from vocab import Vocab
from batcher import Dataset, ReadData
from model import HyperModel, StandardModel, MikolovModel


parser = argparse.ArgumentParser()
parser.add_argument('expdir')
parser.add_argument('--mode', choices=['train', 'debug', 'eval'],
                    default='train')
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

filename = '/n/falcon/s0/ajaech/clean.tsv.bz'
time.sleep(random.randint(0, 60))
usernames, texts = ReadData(filename, mode=args.mode)

batch_size = params.batch_size
if args.mode != 'train':
  params.batch_size = 20
dataset = Dataset(max_len=params.max_len + 1, preshuffle=True, batch_size=params.batch_size)
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

if args.mode != 'debug':
  dataset.Prepare(vocab, username_vocab)


models = {'hyper': HyperModel, 'mikolov': MikolovModel, 
          'standard': StandardModel}
model = models[params.model](params, len(vocab), len(username_vocab), 
                             use_nce_loss=args.mode == 'train')

saver = tf.train.Saver(tf.all_variables())
session = tf.Session(config=config)

def Train(expdir):
  logging.basicConfig(filename=os.path.join(expdir, 'logfile.txt'),
                      level=logging.INFO)
  tvars = tf.trainable_variables()
  grads, _ = tf.clip_by_global_norm(tf.gradients(model.cost, tvars), 5.0)
  optimizer = tf.train.AdamOptimizer(params.learning_rate)
  train_op = optimizer.apply_gradients(zip(grads, tvars))

  print('initalizing')
  session.run(tf.initialize_all_variables())

  for idx in xrange(200000):
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


def Greedy(expdir):
  saver.restore(session, os.path.join(expdir, 'model.bin'))

  m = model
  sess = session
  v = vocab

  for var in tf.trainable_variables():
    values = var.eval(session=session)
    print '{0}\t{1:.3f}\t{2:.3f}'.format(var.name, values.max(), values.min())

  return
  current_word = '<S>'
  prevstate_h = np.zeros((1, 150))
  prevstate_c = np.zeros((1, 150))

  for i in xrange(10):

    a = session.run([model.next_word_prob, model.nextstate_h, model.nextstate_c],
                    {model.wordid: np.array([vocab[current_word]]),
                     model.prevstate_c: prevstate_c,
                     model.prevstate_h: prevstate_h
                     })
    probs, prevstate_h, prevstate_c = a
    current_word_id = np.argmax(probs)
    current_word = vocab.idx_to_word[current_word_id]
    print current_word
    

def GetText(s, seq_len):
  ws = [vocab.idx_to_word[s[0, i]] for i in range(seq_len)]
  return ' '.join(ws)

def Eval(expdir):
  print 'loading model'
  saver.restore(session, os.path.join(expdir, 'model.bin'))

  v = vocab

  total_word_count = 0
  total_log_prob = 0
  for pos in xrange(dataset.GetNumBatches()):
    print pos
    s, seq_len, usernames = dataset.GetNextBatch()

    feed_dict = {
        model.x: s[:, :-1],
        model.y: s[:, 1:],
        model.seq_len: seq_len,
        model.username: usernames
    }

    a = session.run([model.cost], feed_dict=feed_dict)[0]
      

    m = model
    sess = session

    total_word_count += sum(seq_len)
    total_log_prob += float(a * sum(seq_len))

    print np.exp(total_log_prob / total_word_count)

if args.mode == 'train':
  Train(args.expdir)

if args.mode == 'eval':
  Eval(args.expdir)

if args.mode == 'debug':
  Greedy(args.expdir)
