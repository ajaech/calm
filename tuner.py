import json
import numpy as np
import os
import random

threads = 14

def GetRandomSetting():
  """Gets a random parameter setting."""
  model_params = {}

  model_params['dropout_keep_prob'] = random.choice(np.arange(0.5, 1.0, 0.1))
  model_params['nce_samples'] = random.choice(range(100, 500, 50))
  model_params['batch_size'] = random.choice([100, 200, 500])

  model_params['embedding_dims'] = random.choice([180, 196, 256])
  model_params['cell_size'] = random.choice([200, 250, 300])
  model_params['user_embedding_size'] = random.choice([10, 15, 30])

  model_params['max_len'] = 35

  model_params['model'] = random.choice(['hyper', 'mikolov', 'standard'])

  model_params['learning_rate'] = random.choice([0.001, 0.005])

  return model_params


for i in xrange(25):
  d = GetRandomSetting()
  fname = os.path.join('settings', '{0}.json'.format(i))
  with open(fname, 'w') as f:
    json.dump(d, f)

  print 'python rnnlm.py exps/exp{0} --params={1} --threads={2} 2> exps/exp{0}/error.log'.format(
    i, fname, threads)
