import bz2
import code
import json

from nltk import tokenize

f1 = '/g/ssli/data/reddit/reddit_data_2007_2015/2011/RC_2011-06.bz2'
f2 = '/g/ssli/data/reddit/reddit_data_2007_2015/2015/RC_2015-04.bz2'

subreddits = set(('books', 'nba', 'buildapc'))

outfile = bz2.BZ2File('/s0/ajaech/reddit.tsv', 'w')

def Extract(fname, year):

  with bz2.BZ2File(fname, 'r') as f:
    for idx, line in enumerate(f):
      data = json.loads(line)
      text = data['body']
      if text == '[deleted]':
        continue

      subreddit = data['subreddit']
      if subreddit not in subreddits:
        continue
    
      sentences = tokenize.sent_tokenize(text)
      for sentence in sentences:
        words = ' '.join(tokenize.word_tokenize(sentence))
        line = u'{0}\t{1}\t{2}\n'.format(subreddit, year, words)
        outfile.write(line.encode('utf8'))
        break

Extract(f1, 2011)
Extract(f2, 2015)

outfile.close()
