#! /usr/bin/env python
# This file is a collection of hacks to plot the training cost
# from the log files.
import code
import argparse
import json
import numpy as np
import os
import pandas
from matplotlib import pyplot

parser = argparse.ArgumentParser()
parser.add_argument('filename', nargs='+')
parser.add_argument('--var', default='cost', nargs='+')
parser.add_argument('--diff', dest='diff', default=False, action='store_true')
parser.add_argument('--smooth', type=int, default=1)
parser.add_argument('--invert', default=False, action='store_true')
args = parser.parse_args()

def PlotIt(filename, var, diff=False, invert=False, smooth=1.0):

  data = []
  with open(filename, 'rt') as f:
    for line in f:
      # When processing files created by the python logging module,
      # it is necessary to find the json dict inside the log string.
      # This is a hack to make that happen.
      if '{' not in line:
        continue
      if line.startswith('INFO:root:'):
        line = line[10:]
      line = line.replace("'", '"')
      if ' nan,' in line:
        continue
      data.append(json.loads(line))
  data = pandas.DataFrame(data[1:])

  values = data[var]
  if diff:
    # normalize by diff of iters
    x_diff = data.iter.diff()
    values = data[var].diff() / x_diff

  if invert:
    values = 1.0/values

  # do smoothing
  values = np.convolve(values, 1.0 / smooth * np.ones(smooth),
                       mode='valid').T
  print values[-1], filename

  # remove nans
  nan_idx = ~np.isnan(values)
  values = values[nan_idx]
  iters = data.iter.values[nan_idx]
  pyplot.plot(iters, values)

  pyplot.xlabel('Iteration')
  ylabel = var
  if invert:
    if diff:
      ylabel = 'Iterations / change in {0}'.format(var)
    else:
      ylabel = '1 / {0}'.format(var)
  pyplot.ylabel(ylabel)


legend = []
for filename in args.filename:
  if os.path.isdir(filename):
    filename = os.path.join(filename, 'logfile.txt')

  if not os.path.exists(filename):
    print '{0} does not exist'.format(filename)
    continue

  if type(args.var) is not list:
    var_list = [args.var]
  else:
    var_list = args.var

  for v in var_list:
    legend.append('{0} {1}'.format(v, filename))
    PlotIt(filename, v, args.diff, args.invert, args.smooth)

pyplot.legend(legend, loc='best')
pyplot.show()
