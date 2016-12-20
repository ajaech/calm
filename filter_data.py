import bz2
import code
import collections

limit = 25000000

counts = collections.Counter()
filename = '/s0/ajaech/reddit.tsv'
with bz2.BZ2File(filename, 'r') as f:
  for idx, line in enumerate(f):
    username, _ = line.split('\t')
    counts.update([username])

    if idx % 100000 == 0:
      print idx
    if idx > limit:
        break

with bz2.BZ2File(filename, 'r') as f:
  with bz2.BZ2File('/s0/ajaech/clean.tsv.bz', 'w') as g:
    for idx, line in enumerate(f):
      username, _ = line.split('\t')
      if counts[username] > 100:
        g.write(line)
      if idx > limit:
        break
