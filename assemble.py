import glob
import json
import numpy as np
import os
import pandas

results = []

for dirname in glob.glob('exps/exp1*'):
  filename = os.path.join(dirname, 'ppl.txt')
  if os.path.exists(filename):
    with open(filename, 'r') as f:
      lines = f.readlines()
      if len(lines):
        ppl = lines[-1].split()[-1]

        print ppl, dirname

        with open(os.path.join(dirname, 'params.json'), 'r') as g:
          params = json.load(g)

          try:
            params['ppl'] = float(ppl)
          except ValueError:
            print 'error: could not convert ppl <{0}>'.format(ppl)
            continue
          params['dir'] = dirname

          results.append(params)

  filename = os.path.join(dirname, 'pplstats.csv.gz')
  if os.path.exists(filename):
    print filename
    data = pandas.read_csv(filename)
    
    def PPL(df):
      return (df.cost * df.length).sum() / df.length.sum()

    summary = data.groupby('uname').length.agg(len).reset_index()
    ppl = data.groupby('uname').apply(PPL)

    summary['ppl'] = ppl.apply(np.exp).values
    summary = summary[summary.length > 60].sort_values('ppl')

    summary.to_csv(os.path.join(dirname, 'pplsummary.csv'))

df = pandas.DataFrame(results)
df = df.sort_values('ppl')
df.to_csv('results.csv', index=False, sep='\t')
