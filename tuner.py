import json
import numpy as np
import os
import random

threads = 8
#data = '/g/ssli/data/LowResourceLM/tweets/train.tsv.gz'
data = '/n/falcon/s0/ajaech/reddit.tsv.bz2'
#data = '/g/ssli/data/scotus/aaron/scotus_big.tsv.gz'

def GetStratified(n):
  model_params = {}

  model_params['nce_samples'] = 100
  model_params['batch_size'] = 400
  model_params['embedding_dims'] = 200
  model_params['cell_size'] = 240
  model_params['learning_rate'] = 0.001
  model_params['iters'] = 70000

  model_params['context_vars'] = ['subreddit']
  model_params['context_embed_sizes'] = [25]
  model_params['context_embed_size'] = None
  model_params['min_vocab_count'] = 20
  model_params['max_len'] = 35
  model_params['dropout_keep_prob'] = 1.0
  model_params['val_data'] = None

  model_params['splitter'] = 'word'
  model_params['hash_table_size'] = 70000027
  model_params['disable_bloom'] = False

  if n & 1 > 0:
    model_params['use_hash_table'] = True
  else:
    model_params['use_hash_table'] = False

  if n & 2 > 0:
    model_params['use_hyper_adaptation'] = True
  else:
    model_params['use_hyper_adaptation'] = False

  if n & 4 > 0:
    model_params['use_mikolov_adaptation'] = True
  else:
    model_params['use_mikolov_adaptation'] = False

  if n & 8 > 0:
    model_params['use_softmax_adaptation'] = True
  else:
    model_params['use_softmax_adaptation'] = False

  return model_params


def GetRandomSetting(i=None):
  """Gets a random parameter setting."""
  model_params = {}


  # tweets
  model_params['context_vars'] = ["lang"]
  model_params['context_embed_sizes'] = random.choice([[3], [6], [12]])
  model_params['context_embed_size'] = None
  model_params['min_vocab_count'] = 5
  model_params['max_len'] = 200
  model_params['dropout_keep_prob'] = random.choice([0.9, 0.95, 0.99, 1.0])
  model_params['val_data'] = '/g/ssli/data/LowResourceLM/tweets/val.tsv.gz'

  """
  # scotus
  model_params['context_vars'] = ['case', 'person', 'role']
  model_params['context_embed_sizes'] = random.choice(
    [[4, 7, 3], [5, 7, 4], [6, 8, 5], [9, 15, 8]])
  model_params['context_embed_size'] = random.choice(
    [14, 21, 28, 35])
  model_params['min_vocab_count'] = 10
  model_params['max_len'] = 47
  model_params['dropout_keep_prob'] = random.choice([0.5, 0.55, 0.6, 0.7, 0.9])
  model_params['val_data'] = None
  """

  """
  # reddit
  model_params['context_vars'] = ['subreddit']
  model_params['context_embed_sizes'] = random.choice([[20], [40], [60], [90]])
  model_params['context_embed_size'] = None
  model_params['min_vocab_count'] = 20
  model_params['max_len'] = 35
  model_params['dropout_keep_prob'] = random.choice([0.9, 0.95, 1.0])
  model_params['val_data'] = None
  """

  model_params['nce_samples'] = random.randint(50, 200)
  model_params['batch_size'] = random.randint(50, 250)

  model_params['embedding_dims'] = random.randint(12, 40)
  model_params['cell_size'] = 300

  model_params['use_softmax_adaptation'] = random.choice([True, True, False, False, False])
  model_params['use_mikolov_adaptation'] = random.choice([True, True, False, False, False])
  model_params['use_hyper_adaptation'] = random.choice([True, True, False, False, False])
  if model_params['use_hyper_adaptation']:
    model_params['use_mikolov_adaptation'] = True

  model_params['learning_rate'] = 0.001
  model_params['iters'] = random.choice([42000, 55000, 80000, 95000])
  model_params['splitter'] = 'char'

  #model_params['use_hash_table'] = random.choice([True, True, False, False, False])
  model_params['use_hash_table'] = False
  model_params['hash_table_size'] = random.choice([10000003, 3000007])
  
  model_params['l1_penalty'] = 0.0
  if model_params['use_hash_table']:
    model_params['l1_penalty'] = random.choice([10.0, 5.0, 1.0, 0.01, 0.1, 0.001, 0.0, 1.0])

  return model_params


cmd = './rnnlm.py exps/finalstrat{0} --params={1} --threads={2} --data={3} 2> exps/finalstrat{0}.error.log'

for i in xrange(16):
  d = GetStratified(i)
  fname = os.path.join('settings', '{0}.json'.format(i))
  with open(fname, 'w') as f:
    json.dump(d, f)

  print cmd.format(i, fname, threads, data)
