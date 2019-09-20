# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 10:21:17 2019

@author: user
"""

import statsmodels
import pandas as pd
import statsmodels.api as sm
duncan_prestige = sm.datasets.get_rdataset("Duncan", "carData")
Boston = sm.datasets.get_rdataset("Boston")
import sklearn
from sklearn.datasets import load_boston
import sklearn.model_selection
from sklearn.linear_model import LinearRegression


Boston = load_boston()
print(Boston.keys())
print(Boston.feature_names)
type(Boston)
Boston.data
bos = pd.DataFrame(Boston.data, columns=Boston.feature_names)
bos
bos.head()
bos['PRICE'] = Boston.target
bos.head()
print(bos.describe())
X = bos.drop('PRICE', axis=1)
Y = bos['PRICE']
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
        X, Y, test_size=0.33, random_state=5)
lm = LinearRegression()
lm.fit(X_train, Y_train)
Y_pred = lm.predict(X_test)
plt.scatter(Y_test, Y_pred)
import matplotlib.pyplot as plt
plt.scatter(Y_test, Y_pred)
mse = sklearn.metrics.mean_squared_error(Y_test, Y_pred)
print(mse)
lm.score
lm.score()
help(lm.score)
lm.score(X_test,Y_test)
import statsmodels.api as sm
df = sm.datasets.get_rdataset("Guerry", "HistData").data

df
df.keys
df = sm.datasets.get_rdataset("Guerry", "HistData").keys
df = sm.datasets.get_rdataset("Guerry", "HistData").keys()
df = sm.datasets.get_rdataset("Guerry", "HistData").columns
df = sm.datasets.get_rdataset("Guerry", "HistData").data
df = sm.datasets.get_rdataset("Guerry", "HistData").columns
df[['Lottery', 'Literacy', 'Wealth', 'Region']].dropna()
type(df)
df1 = sm.datasets.get_rdataset("Guerry", "HistData").data
type(df1)
df1.columns
mod = smf.ols(formula='Lottery ~ Literacy + Wealth + Region', data=df)
import statsmodels.formula.api as smf
mod = smf.ols(formula='Lottery ~ Literacy + Wealth + Region', data=df)
mod.fit()
res = mod.fit()
print(res.summary())
res1 = smf.ols(formula='Lottery ~ Literacy : Wealth - 1', data=df).fit()
res1.params
print(res1.params)
mod = smf.ols(formula='Lottery ~ Literacy + Wealth + Region', data=df)
res = mod.fit()
res.params()
res.params
help(res.predict)
help(res.resid)
type(res)
res.predict()
ypred = res.predict()
ypred
 res.resid
fig, ax = plt.subplots(figsize=(12,8))
import statsmodels.api as sm
sm.graphics.influence_plot(res, ax=ax, criterion="cooks")
res.resid
res.rsquared
print(res.summary())
res.resid
res.scale
help(res.scale)
res.scale
res.rsquared
res.resid
norm(res.resid)/sqrt(84)
import numpy as np
np.norm(res.resid)/sqrt(84)
np.linalg.norm(res.resid)/sqrt(84)
np.linalg.norm(res.resid)/np.sqrt(84)