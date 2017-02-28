import argparse
import bz2
import gzip
import itertools
import numpy as np
import pandas
import random


def GetFileHandle(filename):
  if filename.endswith('.bz2'):
    return bz2.BZ2File(filename, 'r')
  if filename.endswith('.gz'):
    return gzip.open(filename, 'r')
  return open(filename, 'r')


def WordSplitter(text):
  text = text.replace('\n', ' ')
  return ['<S>'] + text.lower().split() + ['</S>']


def CharSplitter(text):
  return ['<S>'] + list(text.strip()) + ['</S>']


def NgramSplitter(text):
  chars = list(text.strip())
  return [c if c != ' ' else 'SPACE' for c in chars]


def ReadData(filename, columns, limit):
  with GetFileHandle(filename) as f:
    data = pandas.read_csv(f, sep='\t', nrows=limit, header=None)
    data = data.fillna('')
  if columns:
    data.columns = columns

  return data


class Dataset(object):

  def __init__(self, max_len=35, batch_size=100, preshuffle=True, name='unnamed'):
    """Init the dataset object.

    Args:
      batch_size: size of mini-batch
      preshuffle: should the order be scrambled before the first epoch
      name: optional name for the dataset
    """
    self.batch_size = batch_size
    self.preshuffle = preshuffle
    self._max_len = max_len

  def GetColumn(self, name):
    return self.data[name]

  def ReadData(self, filename, columns, limit=10000000, mode='train', splitter='word'):
    data = ReadData(filename, columns, limit)

    SplitFunc = {'word': WordSplitter, 'char': CharSplitter}[splitter]
    data['text'] = data['text'].apply(SplitFunc)

    self.valdata = None
    if mode == 'all':
      self.data = data
      self.valdata = ReadData('/g/ssli/data/LowResourceLM/tweets/val.tsv.gz',
                              columns, limit)
      self.valdata['text'] = self.valdata['text'].apply(SplitFunc)
    else:
      train_data = data[(data.index.values - 1) % 10 > 1]
      eval_data = data[(data.index.values - 1) % 10 <= 1]
      if mode == 'train':
        self.data = train_data
        self.valdata = eval_data.reset_index(drop=True)
      elif mode == 'eval':
        self.data = eval_data.reset_index(drop=True)
    print 'loaded {0} sentences'.format(len(self.data))

  def GetSentences(self):
    return self.data['text']

  @staticmethod
  def GetNumberLine(line, vocab, pad_length):
    """Convert list of words to matrix of word ids."""
    ids = [vocab[w] for w in line[:pad_length]]
    if len(ids) < pad_length:
      ids += [vocab['</S>']] * (pad_length - len(ids))
    return np.array(ids)

  def Prepare(self, word_vocab, context_vocabs):
    self.current_idx = 0

    self._Prepare(self.data, word_vocab, context_vocabs)
    if self.valdata is not None:
      self._Prepare(self.valdata, word_vocab, context_vocabs)
      self.current_val_idx = 0

    if self.preshuffle:
      self._Permute()

  def _Prepare(self, df, word_vocab, context_vocabs):
    df['seq_lens'] = df['text'].apply(
      lambda x: min(len(x), self._max_len))
    df['text'] = df['text'].apply(
      lambda x: self.GetNumberLine(x, word_vocab, self._max_len))
    for context_var in context_vocabs:
      vocab = context_vocabs[context_var]
      df[context_var] = df[context_var].apply(
        lambda x: vocab[x])

  def GetNumBatches(self):
    """Returns num batches per epoch."""
    return len(self.data) / self.batch_size

  def _Permute(self):
    """Shuffle the training data."""
    self.data = self.data.sample(frac=1).reset_index(drop=True)

  def GetNextBatch(self):
    if self.current_idx + self.batch_size > len(self.data):
      self.current_idx = 0

      self._Permute()    

    idx = range(self.current_idx, self.current_idx + self.batch_size)
    self.current_idx += self.batch_size

    return self.data.iloc[idx]

  def GetValBatch(self):
    if self.current_val_idx + self.batch_size > len(self.valdata):
      self.current_val_idx = 0

    idx = range(self.current_val_idx, self.current_val_idx + self.batch_size)
    self.current_val_idx += self.batch_size
    return self.data.iloc[idx]


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--filename', default='/s0/ajaech/reddit.tsv.bz2')
  parser.add_argument('--mode', choices=['train', 'eval'])
  parser.add_argument('--out')
  args = parser.parse_args()

  data = ReadData(args.filename, mode=args.mode, columns=None,
                  limit=None)
  usernames = data[0]
  texts = data[2].apply(NgramSplitter)
  with open(args.out, 'w') as f:
    for uname, t in zip(usernames, texts):
      f.write('{0}\t'.format(uname))
      f.write(' '.join(t[1:-1]))
      f.write('\n')
