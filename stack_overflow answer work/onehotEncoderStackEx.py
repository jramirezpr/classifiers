# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 08:23:52 2019

@author: Guest
"""

import numpy as np
import pandas as pd


dataset = pd.read_csv('googleplaystore.csv')
dataset= dataset.dropna(subset = ['Rating'])

X = dataset.iloc[:, :3].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer 

labelencoder = LabelEncoder()


X[:, 1] = labelencoder.fit_transform(X[:, 1])


onehotencoder = OneHotEncoder(categories='auto')
y = onehotencoder.fit_transform(X[:, 1].reshape(-1, 1))

