import json
import numpy as np
import os
import random

threads = 10
data = 'tweets.tsv.gz'

def GetRandomSetting():
  """Gets a random parameter setting."""
  model_params = {}

  model_params['context_vars'] = ['lang', 'dataset']
  model_params['context_embed_sizes'] = random.choice(
    [[3, 3], [5, 5]])
  model_params['context_embed_size'] = random.choice(
    [5, 9, 15])
  model_params['dropout_keep_prob'] = random.choice([0.8, 0.9, 1.0])
  model_params['nce_samples'] = None
  model_params['batch_size'] = random.choice([15, 30, 60])

  model_params['embedding_dims'] = random.choice([10, 15, 20])
  model_params['cell_size'] = random.choice([128, 190, 230, 270])

  model_params['max_len'] = 200

  model_params['use_softmax_adaptation'] = random.choice([True, True, False, False, False])
  model_params['use_mikolov_adaptation'] = random.choice([True, True, False, False, False])
  model_params['use_hyper_adaptation'] = random.choice([True, True, False, False, False])

  model_params['learning_rate'] = 0.001
  model_params['iters'] = random.choice([60000, 80000, 100000, 120000])
  model_params['splitter'] = 'char'

  return model_params


cmd = './rnnlm.py exps/dual{0} --params={1} --threads={2} --data={3} 2> exps/dual{0}.error.log'

for i in xrange(30):
  d = GetRandomSetting()
  fname = os.path.join('settings', '{0}.json'.format(i))
  with open(fname, 'w') as f:
    json.dump(d, f)

  print cmd.format(i, fname, threads, data)

