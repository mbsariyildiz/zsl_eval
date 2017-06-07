"""
Mert Bulent Sariyildiz,
mert.sariyildiz@bilkent.edu.tr
28 May, 2017

Score function of semantic auto-encoder proposed in Semantic Autoencoder for Zero-Shot Learning by Kodirov et al. 
See the paper:  https://arxiv.org/abs/1704.08345
See the code published by Elyor Kodirov: https://github.com/Elyorcv/SAE
"""

import numpy as np
import h5py
from scipy.spatial.distance import cdist

def score_sae(X, S_gt, model_path):
  with h5py.File(model_path, 'r') as hf:
    W = hf['W'].value

    # make sure dimensions match for matmul
    if X.shape[1] != W.shape[0]:
      W = W.T

    if W.shape[1] != S_gt.shape[0]:
      S_gt = S_gt.T

  S_preds = np.matmul(X, W)
  # distance between gt attributes and predicted ones
  scores = 1.0 - cdist(S_preds, S_gt, 'cosine')

  return scores
