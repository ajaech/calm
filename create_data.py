import bz2
import code
import json
import random
import re

from nltk import tokenize

f2 = '/g/ssli/data/reddit/reddit_data_2007_2015/2015/RC_2015-04.bz2'

outfile = bz2.BZ2File('/s0/ajaech/reddit.tsv', 'w')

def Extract(fname, year):

  count = 0
  with bz2.BZ2File(fname, 'r') as f:
    for idx, line in enumerate(f):
      data = json.loads(line)
      text = data['body']
      if text == '[deleted]':
        continue
      author = data['author']

      sentences = tokenize.sent_tokenize(text)
      for sentence in sentences:
        re.sub(ur'http://\S*', '<URL>', sentence)
        words = ' '.join(tokenize.word_tokenize(sentence.lower()))
        
        fields = u'\t'.join((author, words))
        line = u'{0}\n'.format(fields)
        outfile.write(line.encode('utf8'))
        break

Extract(f2, 2015)

outfile.close()
