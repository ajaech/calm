#!/usr/bin/env python
import argparse
import bunch
import json
import os
import pickle
import tensorflow as tf
import numpy as np

from model import HyperModel
from vocab import Vocab


import code


parser = argparse.ArgumentParser()
parser.add_argument('expdir')
parser.add_argument('--threads', type=int, default=8)
args = parser.parse_args()


param_filename = os.path.join(args.expdir, 'params.json')
with open(param_filename, 'r') as f:
  params = bunch.Bunch(json.load(f))
params.nce_samples = 1000
params.batch_size = 1


vocab = Vocab.Load(os.path.join(args.expdir, 'word_vocab.pickle'))
with open(os.path.join(args.expdir, 'context_vocab.pickle'), 'rb') as f:
  context_vocabs = pickle.load(f)

model = HyperModel(
  params, vocab.GetUnigramProbs(),
  [len(context_vocabs[v]) for v in params.context_vars], 
  use_nce_loss=True)

saver = tf.train.Saver(tf.all_variables())
config = tf.ConfigProto(inter_op_parallelism_threads=args.threads,
                        intra_op_parallelism_threads=args.threads)
session = tf.Session(config=config)

saver.restore(session, os.path.join(args.expdir, 'model.bin'))

for _ in range(10):
  sentence = raw_input("Please enter something: ")
  words = ['<S>'] + sentence.split() + ['</S>']
  seq_len = len(words)
  if seq_len < params.max_len:
    words += ['</S>'] * (params.max_len - seq_len + 1)
  subreddit = 'AskReddit'

  feed_dict = {
    model.x: np.expand_dims([vocab[w] for w in words[:-1]], 0),
    model.y: np.expand_dims([vocab[w] for w in words[1:]], 0),
    model.seq_len: np.reshape(seq_len, [1,]),
    model.context_placeholders['subreddit']: np.reshape(
      context_vocabs['subreddit'][subreddit], [1,])
  }

  r = session.run(model.per_sentence_loss, feed_dict)
  print r
