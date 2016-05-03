import collections
import pickle
import numpy as np
import re

class Vocab(object):

  def _normalize(self, word):
    if re.match(ur'^<.*>$', word):
      return word

    if self.specialcase:
      newword = []
      prev_is_lower = True
      for letter in word:
        if letter.isupper():
          if prev_is_lower:
            newword.append('~')
          prev_is_lower = False
        else:
          prev_is_lower = True
        newword.append(letter.lower())
      word = ''.join(newword)

    if self.lowercase:
      word = word.lower()

    if self.numreplace:
      word = re.sub(ur'\d', '#', word)

    return word

  def __init__(self, tokenset, unk_symbol='<UNK>',
               lowercase=False, numreplace=False):
    self.lowercase = lowercase
    self.numreplace = numreplace

    self.specialcase = False

    tokenset = set([self._normalize(w) for w in tokenset])

    self.vocab_size = len(tokenset)
    self.unk_symbol = unk_symbol

    self.word_to_idx = dict(zip(sorted(tokenset),
                                range(self.vocab_size)))
    self.idx_to_word = dict(zip(self.word_to_idx.values(),
                            self.word_to_idx.keys()))


  @staticmethod
  def Load(filename):
    with open(filename, 'rb') as f:
      v = pickle.load(f)
    return v

  @classmethod
  def MakeFromData(cls, lines, min_count, unk_symbol='<UNK>',
                   max_length=None, no_special_syms=False, normalize=False):
    lowercase=False
    numreplace=False

    if normalize:
      lowercase=True
      numreplace=True

    token_counts = collections.Counter()

    for line in lines:
      token_counts.update(line)

    tokenset = set()
    for word in token_counts:
      if max_length and len(word) > max_length:
        continue
      if token_counts[word] >= min_count:
        tokenset.add(word)

    if not no_special_syms:
      tokenset.add(unk_symbol)
      tokenset.add('<S>')
      tokenset.add('</S>')
    return cls(tokenset, unk_symbol=unk_symbol, lowercase=lowercase, 
               numreplace=numreplace)

  @classmethod
  def ByteVocab(cls):
    """Creates a vocab that has a token for each possible byte.

    It's useful to have a fixed byte vocab so that the subset of bytes
    that form the vocab is not dependent on the dataset being used. Thus,
    the learned byte embeddings can be reused on different datasets.
    """
    c = '0123456789abcdef'
    tokens = ['<S>', '</S>']
    for i in c:
      for j in c:
        tokens.append(i + j)
    return cls(tokens)

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
    return self.word_to_idx.keys()
  
  def LookupIdx(self, token):
    token = self._normalize(token)
    if token in self.word_to_idx:
      return self.word_to_idx[token]
    return self.word_to_idx.get(self.unk_symbol, None)

  def __contains__(self, key):
    key = self._normalize(key)
    return key in self.word_to_idx

  def __getitem__(self, key):
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
