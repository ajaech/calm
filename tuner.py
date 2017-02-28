import json
import numpy as np
import os
import random

threads = 10
#data = '/g/ssli/data/LowResourceLM/tweets/train.tsv.gz'
data = '/n/falcon/s0/ajaech/reddit.tsv.bz2'

def GetRandomSetting():
  """Gets a random parameter setting."""
  model_params = {}

  model_params['val_data'] = None
  # '/g/ssli/data/LowResourceLM/tweets/val.tsv.gz'
  model_params['context_vars'] = ['subreddit']
  model_params['context_embed_sizes'] = random.choice(
    [[7], [9], [14]])
  model_params['context_embed_size'] = random.choice(
    [7, 13, 20])
  model_params['dropout_keep_prob'] = random.choice([1.0])
  model_params['nce_samples'] = random.choice([95, 120, 160])
  model_params['batch_size'] = random.choice([50, 57, 65])

  model_params['embedding_dims'] = random.choice([118, 129, 140])
  model_params['cell_size'] = random.choice([280, 305, 315])

  model_params['max_len'] = 35

  model_params['use_softmax_adaptation'] = False
  model_params['use_mikolov_adaptation'] = random.choice([True, True, False, False, False])
  model_params['use_hyper_adaptation'] = random.choice([True, True, False, False, False])

  model_params['learning_rate'] = 0.001
  model_params['iters'] = random.choice([130000, 140000, 160000])
  model_params['splitter'] = 'word'

  model_params['use_hash_table'] = random.choice([True, True, False])
  model_params['hash_table_size'] = random.choice([30000001, 10000001, 50000001])

  return model_params


cmd = './rnnlm.py exps/bloom{0} --params={1} --threads={2} --data={3} 2> exps/bloom{0}.error.log'

for i in xrange(30):
  d = GetRandomSetting()
  fname = os.path.join('settings', '{0}.json'.format(i))
  with open(fname, 'w') as f:
    json.dump(d, f)

  print cmd.format(i, fname, threads, data)
