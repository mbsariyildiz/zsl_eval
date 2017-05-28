import numpy as np
import h5py

def score_ale(X, S_gt, model_path):
  """
  
  """
  with h5py.File(model_path, 'r') as hf:
    W = hf['W'].value

    # make sure dimensions match for matmul
    if X.shape[1] != W.shape[0]:
      W = W.T

    if W.shape[1] != S_gt.shape[0]:
      S_gt = S_gt.T

  return np.matmul(np.matmul(X, W), S_gt)

  