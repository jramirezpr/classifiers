# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 17:03:54 2019

@author: Guest
"""

import pandas as pd
from  sklearn.svm import  LinearSVC
df = pd.read_csv("data.csv", names=['x', 'y', 'class'], header=None)
df['class'] = df['class']
ax1 = df.plot.scatter(x='x',
                      y='y',
                      kind= 'scatter',
                      ax = ax,
                      c='class', 
                      colormap='viridis')
svc = LinearSVC(fit_intercept=False, C=100)
model = svc.fit(df[['x','y']],df['class'])
w = model.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx
plt.scatter(df['x'],df['y'])
plt.plot(xx, yy, 'k-')
plt.show()