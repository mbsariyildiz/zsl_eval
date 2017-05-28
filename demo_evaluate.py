"""
Mert Bulent Sariyildiz,
mert.sariyildiz@bilkent.edu.tr
28 May, 2017

Demo script for evaluating zero-shot/generalized zero shot performance of
particular method on one of datasets (['CUB', 'SUN', 'AWA', 'APY']). 
This work is a part of Python implementation of the MATLAB code published by Yongqin Xian.
For source, please see
https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly/
"""

from __future__ import print_function
from evaluate import evaluate
from score_ale import score_ale

print ('Evaluating ALE on CUB ... ')
evaluate(score_ale, 'CUB.mat', 'ale_CUB.mat')

print ('\nEvaluating ALE on AWA ... ')
evaluate(score_ale, 'AWA.mat', 'ale_AWA.mat')

print ('\nEvaluating ALE on SUN ... ')
evaluate(score_ale, 'SUN.mat', 'ale_SUN.mat')

print ('\nEvaluating ALE on APY ... ')
evaluate(score_ale, 'APY.mat', 'ale_APY.mat')
