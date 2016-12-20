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
    self._usernames = []
    self.name = name

    self.batch_size = 100
    self.preshuffle = preshuffle
    self._max_len = max_len

  def AddDataSource(self, usernames, sentences):
    self._sentences.append(sentences)
    self._usernames.append(usernames)

  def GetSentences(self):
    return itertools.chain(*self._sentences)

  def Prepare(self, word_vocab, username_vocab):
    sentences = list(itertools.chain(*self._sentences))
  
    self.seq_lens = np.array([min(len(x), self._max_len) for x in sentences])
    
    self.current_idx = 0

    self.sentences = self.GetNumberLines(sentences, word_vocab,
                                         self._max_len)
    self.usernames = np.array([username_vocab[u] for u in 
                               itertools.chain(*self._usernames)])

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
    self.usernames = self.usernames[s]

  def GetNextBatch(self):
    if self.current_idx + self.batch_size > self.N:
      self.current_idx = 0

      self._Permute()    

    idx = range(self.current_idx, self.current_idx + self.batch_size)
    self.current_idx += self.batch_size

    return (self.sentences[idx, :], self.seq_lens[idx], 
            self.usernames[idx])
