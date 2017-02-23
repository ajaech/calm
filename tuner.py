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
    [[6], [8], [12]])
  model_params['context_embed_size'] = random.choice(
    [7, 13, 20])
  model_params['dropout_keep_prob'] = random.choice([1.0])
  model_params['nce_samples'] = random.choice([70, 130, 145])
  model_params['batch_size'] = random.choice([42, 55, 70])

  model_params['embedding_dims'] = random.choice([125, 135, 145])
  model_params['cell_size'] = random.choice([290, 310, 325])

  model_params['max_len'] = 35

  model_params['use_softmax_adaptation'] = random.choice([True, True, False, False, False])
  model_params['use_mikolov_adaptation'] = random.choice([True, True, False, False, False])
  model_params['use_hyper_adaptation'] = random.choice([True, True, False, False, False])

  model_params['learning_rate'] = 0.001
  model_params['iters'] = random.choice([125000, 135000, 150000])
  model_params['splitter'] = 'word'

  return model_params


cmd = './rnnlm.py exps/reddit{0} --params={1} --threads={2} --data={3} 2> exps/reddit{0}.error.log'

for i in xrange(12):
  d = GetRandomSetting()
  fname = os.path.join('settings', '{0}.json'.format(i))
  with open(fname, 'w') as f:
    json.dump(d, f)

  print cmd.format(i, fname, threads, data)

