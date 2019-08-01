# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 09:41:21 2019

@author: Guest
"""

import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn import datasets, svm
import matplotlib.pyplot as plt

digits = datasets.load_digits()
X = digits.data
y = digits.target
k_fold = KFold(n_splits=5)
svc = svm.SVC(kernel='linear')
score_mean = []
for C in np.logspace(-10, 0, 10):
    svc = svm.SVC(kernel='linear', C=C)
    score_mean.append(cross_val_score(svc, X, y, cv=k_fold).mean())
plt.plot(np.logspace(-10, 0, 10), score_mean)
plt.ylabel('CV score')
plt.xscale('log')
plt.show()
