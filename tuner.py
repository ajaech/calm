import json
import numpy as np
import os
import random

threads = 10
#data = '/g/ssli/data/LowResourceLM/tweets/train.tsv.gz'
#data = '/n/falcon/s0/ajaech/reddit.tsv.bz2'
data = '/homes/ajaech/Downloads/scotus.tsv.gz'

def GetRandomSetting():
  """Gets a random parameter setting."""
  model_params = {}

  model_params['val_data'] = None
  # '/g/ssli/data/LowResourceLM/tweets/val.tsv.gz'
  model_params['context_vars'] = ['case', 'person', 'role']
  model_params['context_embed_sizes'] = random.choice(
    [[4, 7, 3], [5, 7, 4]])
  model_params['context_embed_size'] = random.choice(
    [9, 15, 22, 30])
  model_params['dropout_keep_prob'] = random.choice([0.55, 0.65, 0.75])
  model_params['nce_samples'] = random.choice([70, 110])
  model_params['batch_size'] = random.choice([90, 120])

  model_params['embedding_dims'] = random.choice([80, 110, 130])
  model_params['cell_size'] = random.choice([120, 140, 175])

  model_params['max_len'] = 47

  model_params['use_softmax_adaptation'] = random.choice([True, False, False, False])
  model_params['use_mikolov_adaptation'] = random.choice([True, True, False, False, False])
  model_params['use_hyper_adaptation'] = random.choice([True, True, False, False, False])
  if model_params['use_hyper_adaptation']:
    model_params['use_mikolov_adaptation'] = True

  model_params['learning_rate'] = 0.001
  model_params['iters'] = random.choice([50000, 85000, 60000])
  model_params['splitter'] = 'word'

  model_params['use_hash_table'] = random.choice([True, True, True, False, False])
  model_params['use_hash_table'] = False
  model_params['hash_table_size'] = random.choice([30000001, 50000001, 80000001])
  model_params['fixed_hyper'] = model_params['use_hyper_adaptation']  # temp flag

  model_params['l1_penalty'] = 0.0
  if model_params['use_hash_table']:
    model_params['l1_penalty'] = random.choice([0.4, 0.01, 0.1, 0.25, 0.001])

  model_params['min_vocab_count'] = 5
  return model_params


cmd = './rnnlm.py exps/scotus{0} --params={1} --threads={2} --data={3} 2> exps/scotus{0}.error.log'

for i in xrange(40, 45):
  d = GetRandomSetting()
  fname = os.path.join('settings', '{0}.json'.format(i))
  with open(fname, 'w') as f:
    json.dump(d, f)

  print cmd.format(i, fname, threads, data)
