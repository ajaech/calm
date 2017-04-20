import pandas
import numpy as np


df = pandas.read_csv('/s0/ajaech/oldexps/bloom137/embeddings.tsv', 
                     sep='\t', header=None)
names = pandas.read_csv('/s0/ajaech/oldexps/bloom137/subreddit.vocab',
                        header=None)
df['subreddit'] = names
df = df.set_index('subreddit')


for _ in range(10):
  subreddit = raw_input('choose a subreddit:')
  v = df.loc[subreddit]

  dist = np.power(df.values - v.values, 2.0).sum(1)
  idx = np.argsort(dist)

  for i in range(10):
    print '{0:.3f}\t{1}'.format(dist[idx[i]], df.index[idx[i]])
