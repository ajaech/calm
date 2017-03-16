import glob
import json
import numpy as np
import os
import pandas
import code

results = []

for dirname in glob.glob('exps/scotus*'):

  if os.path.isfile(dirname):
    continue

  # load the params file
  params_filename = os.path.join(dirname, 'params.json')
  if not os.path.exists(params_filename):
    continue  # this is just an empty directory
  with open(params_filename, 'r') as g:
    params = json.load(g)
  params['dir'] = dirname

  model_filename = os.path.join(dirname, 'model.bin')
  if os.path.exists(model_filename):
    modtime = os.path.getmtime(model_filename)
    params['finish_time'] = modtime

  # Check for perplexity logfile
  filename = os.path.join(dirname, 'ppl.txt')
  if os.path.exists(filename):
    with open(filename, 'r') as f:
      lines = f.readlines()
      if len(lines):
        ppl = lines[-1].split()[-1]

        print ppl, dirname

        try:
          params['ppl'] = float(ppl)
        except ValueError:
          print 'error: could not convert ppl <{0}>'.format(ppl)
          continue

  # Check for F1 score logfile
  filename = os.path.join(dirname, 'f1.txt')
  if os.path.exists(filename):
    with open(filename, 'r') as f:
      lines = f.readlines()
      if len(lines) > 3:
        f1 = lines[-1].split()[-1]
        accuracy = lines[1].split()[-1]
        params['f1'] = f1
        params['acc'] = accuracy

  results.append(params)

  # make the pplsummary file
  filename = os.path.join(dirname, 'pplstats.csv.gz')
  summary_filename = os.path.join(dirname, 'pplsummary.csv')
  if os.path.exists(filename) and ((not os.path.exists(summary_filename)) or
                                   os.path.getmtime(filename) > os.path.getmtime(summary_filename)):
    print filename
    data = pandas.read_csv(filename)

    cname = 'uname'
    if 'uname' not in data.columns:
      if 'subreddit' in data.columns:
        cname = 'subreddit'
      else:
        cname = 'person'
   
    if len(data[cname].unique()) < 2:
      continue
 
    def PPL(df):
      return (df.cost * df.length).sum() / df.length.sum()

    summary = data.groupby(cname).length.agg(len).reset_index()
    ppl = data.groupby(cname).apply(PPL)

    summary['ppl'] = ppl.apply(np.exp).values
    summary = summary[summary.length > 150].sort_values('ppl')

    summary.to_csv(summary_filename)

df = pandas.DataFrame(results)
if 'ppl' in df.columns:
  df = df.sort_values('ppl')
else:
  df = df.sort_values('f1')
df.to_csv('results.csv', index=False, sep='\t')
