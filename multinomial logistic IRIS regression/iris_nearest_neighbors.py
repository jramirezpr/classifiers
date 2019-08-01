# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 14:05:28 2019

@author: Guest
"""

from sklearn import datasets, neighbors, linear_model
import numpy as np
digits = datasets.load_digits()
X_digits = digits.data / digits.data.max()
y_digits = digits.target
indices = np.random.permutation(len(X_digits))
X_digits_train = X_digits[indices[:-179]]
y_digits_train = y_digits[indices[:-179]]
X_digits_test = X_digits[indices[-179:]]
y_digits_test = y_digits[indices[-179:]]
knn = neighbors.KNeighborsClassifier()
knn.fit(X_digits_train, y_digits_train)
y_predict = knn.predict(X_digits_test)
print(len(y_digits_test[y_predict != y_digits_test]))
log = linear_model.LogisticRegression(solver='newton-cg', C=10,
                                      multi_class='multinomial')
log.fit(X_digits_train, y_digits_train)
y_predict2 = log.predict(X_digits_test)
print(len(y_digits_test[y_predict2 != y_digits_test]))
