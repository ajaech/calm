import json
import numpy as np
import os
import random

threads = 10

def GetRandomSetting():
  """Gets a random parameter setting."""
  model_params = {}

  #model_params['dropout_keep_prob'] = random.choice(
  #  np.arange(0.75, 1.01, 0.05))
  model_params['dropout_keep_prob'] = 1.0
  model_params['nce_samples'] = random.choice(range(150, 400, 30))
  model_params['batch_size'] = random.choice([150, 220, 280])

  model_params['embedding_dims'] = random.choice([128, 150, 175, 220])
  model_params['cell_size'] = random.choice([200, 220, 260, 280])
  model_params['user_embedding_size'] = random.choice([12, 18, 25, 36])

  model_params['max_len'] = 35

  model_params['use_softmax_adaptation'] = random.choice([True, True, False, False, False])
  model_params['use_mikolov_adaptation'] = random.choice([True, True, False, False, False])
  model_params['use_hyper_adaptation'] = random.choice([True, True, False, False, False])

  model_params['learning_rate'] = 0.001
  model_params['iters'] = random.choice([80000, 100000, 120000])

  return model_params


for i in xrange(10, 40):
  d = GetRandomSetting()
  fname = os.path.join('settings', '{0}.json'.format(i))
  with open(fname, 'w') as f:
    json.dump(d, f)

  print './rnnlm.py exps/exp{0} --params={1} --threads={2} 2> exps/exp{0}.error.log'.format(
    i, fname, threads)
