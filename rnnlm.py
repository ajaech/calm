import code
import bz2
import numpy as np
import random
import tensorflow as tf

from vocab import Vocab
from batcher import Dataset
from model import MultiModel


def ReadData(filename):
  subreddits = []
  years = []
  texts = []

  with bz2.BZ2File(filename, 'r') as f:
    for line in f:
      if random.choice([True, False, True]):
        continue
      subreddit, year, text = line.split('\t')
      subreddits.append(subreddit)
      years.append(year)
      texts.append(text.lower().split())

  return subreddits, years, texts


max_len = 36
dataset = Dataset(max_len=max_len)
subreddits, years, texts = ReadData('/s0/ajaech/reddit.tsv')
dataset.AddDataSource(subreddits, years, texts)

vocab = Vocab.MakeFromData(texts, min_count=100)
subreddit_vocab = Vocab.MakeFromData([[s] for s in subreddits],
                                     min_count=1, no_special_syms=True)
year_vocab = Vocab.MakeFromData([[y] for y in years], min_count=1,
                                no_special_syms=True)
dataset.Prepare(vocab, subreddit_vocab, year_vocab)

model = MultiModel(max_len-1, len(vocab), len(subreddit_vocab),
                   len(year_vocab))

tvars = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(model.cost, tvars), 5.0)
optimizer = tf.train.AdamOptimizer(0.001)
train_op = optimizer.apply_gradients(zip(grads, tvars))

session = tf.Session()
session.run(tf.initialize_all_variables())

for i in range(10):
  s, seq_len, subreddit, year = dataset.GetNextBatch()
  a = session.run([model.cost], {
      model.x: s[:, :-1],
      model.y: s[:, 1:],
      model.seq_len: seq_len,
      model.year: year, 
      model.subreddit: subreddit})
  print s, seq_len, subreddit, year
  code.interact(local=locals())
