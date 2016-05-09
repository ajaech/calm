import bz2
import code
import json
import random
import re

from nltk import tokenize

f1 = '/g/ssli/data/reddit/reddit_data_2007_2015/2011/RC_2011-06.bz2'
f2 = '/g/ssli/data/reddit/reddit_data_2007_2015/2015/RC_2015-04.bz2'

subreddits = set(('books', 'nba', 'buildapc'))

outfile = bz2.BZ2File('/s0/ajaech/reddit.tsv', 'w')

def Extract(fname, year):

  count = 0
  with bz2.BZ2File(fname, 'r') as f:
    for idx, line in enumerate(f):
      data = json.loads(line)
      text = data['body']
      if text == '[deleted]':
        continue

      subreddit = data['subreddit']
      if subreddit not in subreddits:
        continue
      if subreddit == 'nba' and random.choice([True, False]):
        continue  # skip half of nba comments
    
      sentences = tokenize.sent_tokenize(text)
      for sentence in sentences:
        re.sub(ur'http://\S*', '<URL>', sentence)
        words = ' '.join(tokenize.word_tokenize(sentence))
        
        count += 1
        fields = u'\t'.join((str(count), subreddit, 
                             str(year), words))
        line = u'{0}\n'.format(fields)
        outfile.write(line.encode('utf8'))
        break

Extract(f1, 2011)
Extract(f2, 2015)

outfile.close()
