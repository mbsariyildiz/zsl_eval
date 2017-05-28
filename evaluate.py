"""
Mert Bulent Sariyildiz,
mert.sariyildiz@bilkent.edu.tr
28 May, 2017

This work is a part of Python implementation of the MATLAB code published by Yongqin Xian.
For source, please see
https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly/
"""

import numpy as np
from utils import load_dataset

def evaluate(func, dset_path, model_path):
  """
  A method that evaluates zero-shot/generalized zero-shot performance of particular
  method, i.e. func, on particular dataset, i.e. one of ['CUB', 'AWA', 'SUN', 'APY']

  Inputs:
      func,       function handle for particular ZSL method which takes 
                    input set X, [n_samples, d_features]
                    ground-truth output embeddings (or attributes) per class, S, [n_classes, d_attributes]
                    path indicating where pretrained model is stored. func will load these parameters in order to compute score of each sample.
                  Note that this function must be provided by you. So design func such that it first loads pretrained model parameters and then computes scores of each input. In demo, score function and pretrained models of ALE are used, you may check it out.

      dset_path,  path for dataset to load. This file must be compatible with hdf5 and must have the following fields:
                  *Xtrva,         features of each sample in training+validation set [m1, d_features]
                  *Xte_seen,      features of each sample in seen test set [m2, d_features]
                  *Xte_unseen,    features of each sample in unseen test set [m3, d_features]
                                  note that m1+m2+m3 equals to number of samples in the whole set
                  *Sall_gt,       attributes of each class, [n_classes_all, d_attributes]
                  *Ste_seen_gt,   attributes of seen test classes [n_classes_seen_test, d_attributes]
                  *Ste_unseen_gt, attributes of unseen test classes [n_classes_unseen_test, d_attributes]
                  *Lte_seen,      ground-truth labels of samples in seen test set [m2, 1]
                  *Lte_unseen,    ground-truth labels of samples in unseen test set [m3, 1]
                  *Cte_seen,      labels of classes in seen test set [n_classes_seen_test, 1]
                  *Cte_unseen,    labels of classes in unseen test set [n_classes_unseen_test, 1]
                  *Call,          labels of all classes [n_classes_all, 1]

      model_path, path where model parameters of func is stored. func is supposed to load these parameters.
  """
  dset = load_dataset(dset_path, 'trva', False)

  """
  average class-based zero-shot accuracy
  """
  scores = func(dset['Xte_unseen'], dset['Ste_unseen_gt'], model_path)
  preds = np.argmax(scores, 1)
  preds = dset['Cte_unseen'][preds]
  acc_zsl = compute_acc(dset['Lte_unseen'], preds)

  """
  average class-based generalized zsl accuracy on seen test classes
  """
  scores = func(dset['Xte_seen'], dset['Sall_gt'], model_path)
  preds = np.argmax(scores, 1)
  preds = dset['Call'][preds]
  acc_gzsl_seen = compute_acc(dset['Lte_seen'], preds)

  """
  average class-based generalized zsl accuracy on unseen test classes
  """
  scores = func(dset['Xte_unseen'], dset['Sall_gt'], model_path)
  preds = np.argmax(scores, 1)
  preds = dset['Call'][preds]
  acc_gzsl_unseen = compute_acc(dset['Lte_unseen'], preds)

  print 'ZSL accuracy: ', acc_zsl
  print 'Generalized ZSL accuracy on seen classes: ', acc_gzsl_seen
  print 'Generalized ZSL accuracy on unseen classes: ', acc_gzsl_unseen


def compute_acc(trues, preds):
  """
  Given true and predicted labels, computes average class-based accuracy.
  """

  # class labels in ground-truth samples
  classes = np.unique(trues)
  # class-based accuracies
  cb_accs = np.zeros(classes.shape, np.float32)

  for i, label in enumerate(classes):
    inds_ci = np.where(trues == label)[0]

    cb_accs[i] = np.mean(
      np.equal(
        trues[inds_ci],
        preds[inds_ci]
      ).astype(np.float32)
    )

  return np.mean(cb_accs)

