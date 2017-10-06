import collections
import sys

import numpy as np
import tensorflow as tf
from sklearn import linear_model


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

  for (pred, lab), count in confusion_matrix.most_common(30):
    if pred == lab:
      continue
      
    percent = count / float(max(1, label_totals[lab]))
    print '{0:.2f}% of {1} classified as {2}'.format(100 * percent, lab, pred)

  num_correct = 0
  for lang in all_langs:
    num_correct += confusion_matrix[(lang, lang)]
  acc = num_correct / float(len(preds))
  if show:
    print 'accuracy = {0:.4f}'.format(acc)
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
  return totals[2], acc


class MovingAvg(object):
  
  def __init__(self, p, burn_in=1):
    self.val=None
    self.p=p
    self.burn_in=burn_in

  def Update(self, v):
    if self.burn_in > 0:
      self.burn_in -= 1
      return v

    if self.val is None:
      self.val = v
      return v
    self.val = self.p * self.val + (1.0 - self.p) * v
    return self.val
      

def PrintParams(handle=sys.stdout.write):
  """Print the names of the parameters and their sizes. 

  Args:
    handle: where to write the param sizes to
  """
  handle('NETWORK SIZE REPORT\n')
  param_count = 0
  fmt_str = '{0: <25}\t{1: >12}\t{2: >12,}\n'
  for p in tf.trainable_variables():
    shape = p.get_shape()
    shape_str = 'x'.join([str(x.value) for x in shape])
    handle(fmt_str.format(p.name, shape_str, np.prod(shape).value))
    param_count += np.prod(shape).value
  handle(''.join(['-'] * 60))
  handle('\n')
  handle(fmt_str.format('total', '', param_count))
  if handle==sys.stdout.write:
    sys.stdout.flush()


def GetLine(x, y):
    model = linear_model.LinearRegression()
    nan_idx = ~(np.isnan(x) | np.isnan(y))
    y = y[nan_idx]
    x = x[nan_idx]
    model.fit(x.values.reshape(len(x), 1), y)
    return model
