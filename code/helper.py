import bunch
import json
import os
from math import radians, cos, sin, asin, sqrt


def GetParams(filename, mode, expdir):
  param_filename = os.path.join(expdir, 'params.json')
  if mode == 'train':
    with open(filename, 'r') as f:
      param_dict = json.load(f)
      params = bunch.Bunch(param_dict)
    with open(param_filename, 'w') as f:
      json.dump(param_dict, f)
  else:
    with open(param_filename, 'r') as f:
      params = bunch.Bunch(json.load(f))
  return params


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    km = 6367 * c
    return km

