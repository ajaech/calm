import glob
import json
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

df = pandas.DataFrame(results)
df = df.sort_values('ppl')
df.to_csv('results.csv', index=False, sep='\t')
