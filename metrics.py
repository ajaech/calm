import collections

import numpy as np


def Metrics(preds, labs, show=True):
  """Print precision, recall and F1 for each language.
  Assumes a single language per example, i.e. no code switching.
  Args:
    preds: list of predictions
    labs: list of labels
    show: flag to toggle printing
  """
  all_langs = set(preds + labs)
  preds = np.array(preds)
  labs = np.array(labs)
  label_totals = collections.Counter(labs)
  pred_totals = collections.Counter(preds)
  confusion_matrix = collections.Counter(zip(preds, labs))
  num_correct = 0
  for lang in all_langs:
    num_correct += confusion_matrix[(lang, lang)]
  acc = num_correct / float(len(preds))
  print 'accuracy = {0:.3f}'.format(acc)
  if show:
    print ' Lang     Prec.   Rec.   F1'
    print '------------------------------'
  scores = []
  fmt_str = '  {0:6}  {1:6.2f} {2:6.2f} {3:6.2f}'
  for lang in sorted(all_langs):
    idx = preds == lang
    total = max(1.0, pred_totals[lang])
    precision = 100.0 * confusion_matrix[(lang, lang)] / total
    idx = labs == lang
    total = max(1.0, label_totals[lang])
    recall = 100.0 * confusion_matrix[(lang, lang)] / total
    if precision + recall == 0.0:
      f1 = 0.0
    else:
      f1 = 2.0 * precision * recall / (precision + recall)
    scores.append([precision, recall, f1])
    if show:
      print fmt_str.format(lang, precision, recall, f1)
  totals = np.array(scores).mean(axis=0)
  if show:
    print '------------------------------'
    print fmt_str.format('Total:', totals[0], totals[1], totals[2])
  return totals[2]


class MovingAvg(object):
  
  def __init__(self, p):
    self.val = None
    self.p = p

  def Update(self, v):
    if self.val is None:
      self.val = v
      return v
    self.val = self.p * self.val + (1.0 - self.p) * v
    return self.val
      
  
