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
from scipy.spatial.distance import cdist

def score_ale(X, S_gt, model_path):
  """
  Score function of ALE published by Yongqin Xian.
  """
  with h5py.File(model_path, 'r') as hf:
    W = hf['W'].value

    # make sure dimensions match for matmul
    if X.shape[1] != W.shape[0]:
      W = W.T

  scores = X.dot(W).dot(S_gt.T)
  return scores


  

  