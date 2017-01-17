import json
import os
import pandas


results = []

for i in range(16):
  filename = 'exps/exp{0}/ppl.txt'.format(i)
  if os.path.exists(filename):
    with open(filename, 'r') as f:
      lines = f.readlines()
      if len(lines):
        ppl = lines[-1].split()[-1]

        print ppl, i

        with open('settings/{0}.json'.format(i), 'r') as g:
          params = json.load(g)

          params['ppl'] = float(ppl)

          results.append(params)

df = pandas.DataFrame(results)
df.to_csv('results.csv')
