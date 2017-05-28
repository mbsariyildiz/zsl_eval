"""
Mert Bulent Sariyildiz,
mert.sariyildiz@bilkent.edu.tr
28 May, 2017

This work is a part of Python implementation of the MATLAB code published by Yongqin Xian.
For source, please see
https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly/
"""

import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler

def load_dataset(path, normalization='tr', verbose=False):
  """
  Loads dataset file compatible with h5py.

  Inputs:
      path,           path of dataset file to load
      normalization,  zero-mean unit-variance normalization scheme. 
                      Possible values are 'tr', 'trva', '' indicating sets used to compute
                      mean and variance. 
                      When it is 'tr', mean and variance will be computed on Xtr and 
                      both Xtr and Xva will be normalized with these statistics. 
                      When it is 'trva', mean and variance will be computed on
                      Xtrva, and Xtrva, Xte_seen and Xte_unseen will be normalized with 
                      thse statistics.
                      If no normalization is desired, leave it as empty string, i.e, ''.
                  
  """
  dset = dict()
  with h5py.File(path, 'r') as hf:
    for k,v in hf.iteritems():
      dset[k] = v.value.astype(np.float32).T

      if verbose:
        if isinstance(dset[k], np.ndarray):
          print k, dset[k].shape, dset[k].dtype
        else:
          print k, dset[k]

  if normalization == '':
    pass
  elif normalization == 'tr':
    scaler = StandardScaler(copy=False)
    scaler.fit(dset['Xtr'])
    scaler.transform(dset['Xtr'])
    scaler.transform(dset['Xva'])

    max_ = np.max(dset['Xtr'])
    dset['Xtr'] /= max_
    dset['Xva'] /= max_

  elif normalization == 'trva':
    scaler = StandardScaler(copy=False)
    scaler.fit(dset['Xtrva'])
    scaler.transform(dset['Xtrva'])
    scaler.transform(dset['Xte_seen'])
    scaler.transform(dset['Xte_unseen'])

    max_ = np.max(dset['Xtrva'])
    dset['Xtrva'] /= max_
    dset['Xte_seen'] /= max_
    dset['Xte_unseen'] /= max_

  return dset
