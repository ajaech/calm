import code
import itertools
import numpy as np
import random


class Dataset(object):

  def __init__(self, max_len=35, preshuffle=True, name='unnamed'):
    """Init the dataset object.

    Args:
      batch_size: size of mini-batch
      preshuffle: should the order be scrambled before the first epoch
      name: optional name for the dataset
    """
    self._sentences = []
    self._subreddits = []
    self._years = []
    self.name = name

    self.batch_size = 1
    self.preshuffle = preshuffle
    self._max_len = max_len

  def AddDataSource(self, subreddits, years, sentences):
    self._sentences.append(sentences)
    self._subreddits.append(subreddits)
    self._years.append(years)

  def GetSentences(self):
    return itertools.chain(*self._sentences)

  def Prepare(self, word_vocab, subreddit_vocab, year_vocab):
    sentences = list(itertools.chain(*self._sentences))
  
    self.seq_lens = np.array([min(len(x), self._max_len) for x in sentences])
    
    self.current_idx = 0

    self.sentences = self.GetNumberLines(sentences, word_vocab,
                                         self._max_len)
    self.subreddits = np.array([subreddit_vocab[s] for s in 
                                itertools.chain(*self._subreddits)])
    self.years = np.array([year_vocab[y] for y in 
                           itertools.chain(*self._years)])

    self.N = len(sentences)
    if self.preshuffle:
      self._Permute()


  @staticmethod
  def GetNumberLines(lines, vocab, pad_length):
    """Convert list of words to matrix of word ids."""
    out = []
    for line in lines:
      if len(line) > pad_length:
        line = line[:pad_length]
      elif len(line) < pad_length:
        line += ['}'] * (pad_length - len(line))
      out.append([vocab[w] for w in line])
    return np.array(out)

  def GetNumBatches(self):
    """Returns num batches per epoch."""
    return self.N / self.batch_size

  def _Permute(self):
    """Shuffle the training data."""
    s = np.arange(self.N)
    np.random.shuffle(s)

    self.sentences = self.sentences[s, :]
    self.seq_lens = self.seq_lens[s]
    self.years = self.years[s]
    self.subreddits = self.subreddits[s]

  def GetNextBatch(self):
    if self.current_idx + self.batch_size > self.N:
      self.current_idx = 0

      self._Permute()    

    idx = range(self.current_idx, self.current_idx + self.batch_size)
    self.current_idx += self.batch_size

    return (self.sentences[idx, :], self.seq_lens[idx], 
            self.subreddits[idx], self.years[idx])
