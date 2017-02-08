import argparse
import collections
import numpy as np
import pickle
import re


class Vocab(object):

  def __init__(self, tokenset, unk_symbol='<UNK>', token_counts=None):
    self.vocab_size = len(tokenset)
    self.unk_symbol = unk_symbol

    self.word_to_idx = dict(zip(sorted(tokenset),
                                range(self.vocab_size)))
    self.idx_to_word = dict(zip(self.word_to_idx.values(),
                            self.word_to_idx.keys()))

    self.token_counts = [token_counts[self.idx_to_word[i]] for i in
                         range(self.vocab_size)]

  def GetUnigramProbs(self):
    return self.token_counts

  @staticmethod
  def Load(filename):
    with open(filename, 'rb') as f:
      v = pickle.load(f)
    return v

  @classmethod
  def MakeFromData(cls, lines, min_count, unk_symbol='<UNK>',
                   max_length=None, no_special_syms=False):
    token_counts = collections.Counter()

    for line in lines:
      token_counts.update(line)

    tokenset = set()
    for word in token_counts.keys():
      if max_length and len(word) > max_length:
        continue
      if token_counts[word] >= min_count:
        tokenset.add(word)
      else:
        token_counts[unk_symbol] += token_counts[word]

    if not no_special_syms:
      tokenset.add(unk_symbol)
      tokenset.add('</S>')

    return cls(tokenset, unk_symbol=unk_symbol, token_counts=token_counts)


  @classmethod
  def LoadFromTextFile(cls, filename, unk_symbol='<UNK>'):
    tokens = []
    with open(filename, 'r') as f:
      for line in f:
        line = line.strip()
        tokens.append(line)
    return cls(tokens, unk_symbol=unk_symbol)

  def GetWords(self):
    """Get a list of words in the vocabulary."""
    return [self.idx_to_word[i] for i in xrange(len(self.word_to_idx))]
  
  def LookupIdx(self, token):
    if token in self.word_to_idx:
      return self.word_to_idx[token]
    return self.word_to_idx.get(self.unk_symbol, None)

  def __contains__(self, key):
    return key in self.word_to_idx

  def __getitem__(self, key):
    """If key is an int lookup word by id, if key is a word then lookup id."""
    if type(key) == int or type(key) == np.int64:
      return self.idx_to_word[key]

    return self.LookupIdx(key)

  def __iter__(self):
    word_list = [self.idx_to_word[x] for x in xrange(self.vocab_size)]
    return word_list.__iter__()

  def __len__(self):
    return self.vocab_size

  def Save(self, filename):
    if filename.endswith('.pickle'):
     with open(filename, 'wb') as f:
      pickle.dump(self, f)
    elif filename.endswith('.txt'):
      with open(filename, 'w') as f:
        for i in range(self.vocab_size):
          f.write('{0}\n'.format(self.idx_to_word[i]))


if __name__ == '__main__':
  """Print the contents of the vocabulary."""
  parser = argparse.ArgumentParser()
  parser.add_argument('filename')
  args = parser.parse_args()

  if args.filename.endswith('.pickle'):
    v = Vocab.Load(args.filename)

    for i in v.GetWords():
      print i

  else:
    with open(args.filename, 'r') as f:
      lines = f.readlines()
    v = Vocab.MakeFromData([x.split() for x in lines], 2)
    v.Save('turkvocab.pickle')
