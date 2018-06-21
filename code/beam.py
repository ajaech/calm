# This file has helper functions for doing beam search decoding
import collections

from Queue import PriorityQueue


class BeamItem(object):
  """This is a node in the beam search tree."""
  
  def __init__(self, prev_word, prev_c, prev_h):
    self.log_probs = 0.0
    if type(prev_word) == list:
      self.words = prev_word
    else:
      self.words = [prev_word]
    self.prev_c = prev_c
    self.prev_h = prev_h
    self.counts = collections.Counter()

  def __str__(self):
    return 'beam {0:.3f}: '.format(self.Cost()) + ' '.join(self.words)

  def Update(self, log_prob, new_word):
    self.words.append(new_word)
    if len(self.words) > 1:  # bigrams
      self.counts['{0}_{1}'.format(self.words[-2], self.words[-1])] += 1
      if len(self.words) > 2:  # trigrams
        self.counts['{0}_{1}_{2}'.format(
          self.words[-3], self.words[-2], self.words[-1])] += 1
    self.log_probs += log_prob

  def IsEligible(self, word, min_length=20, allow_repeated_trigrams=False):
    """Eligibility function allows us to constrain the beam search.

    This is primarily used to force it to generate longer sentences and to 
    prevent excessive repitition. """
    if word == '<S>':  # not sure why some models want to generate <S>
      return False

    if word == '</S>' and len(self.words) < min_length:
      return False

    if allow_repeated_trigrams:
      return True

    # check if bigram has been used before
    if len(self.words):
      bigram = '{0}_{1}'.format(self.words[-1], word)
      if self.counts[bigram] > 1:
        return False

      if len(self.words) > 1:  # check trigrams
        trigram = '{0}_{1}_{2}'.format(self.words[-2], self.words[-1], word)
        if self.counts[trigram]:
          return False
    return self.counts[word] < 5

  def Cost(self):
    return self.log_probs


class BeamQueue(object):
  """Bounded priority queue.

  This is a wrapper around python's built-in priority queue. The wrapper 
  keeps track of the size of the largest item in the beam. It doesn't
  even try to do an insert if the beam is full and the cost of the new
  item is bigger than the biggest thing currently in the queue. 
  """
    
  def __init__(self, max_size=10):
    self.max_size = max_size
    self.size = 0
    self.bound = None
    self.q = PriorityQueue()
        
  def Insert(self, item):
    self.size += 1
    self.q.put((-item.Cost(), item))
    if self.size > self.max_size:
      self.Eject()
            
  def CheckBound(self, val):
    # If the queue is full then we know that there is no chance of a new item
    # being accepted if it's priority is worse than the last thing that got
    # ejected.
    if self.size >= self.max_size and self.bound is not None and val > self.bound:
      return False
    return True
        
  def Eject(self):
    score, _ = self.q.get()
    self.bound = -score
    self.size -= 1
        
  def __iter__(self):
    return self

  def next(self):
    if not self.q.empty():
      _, item = self.q.get()
      return item
    raise StopIteration
