# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 09:06:03 2019

@author: user
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

df_sensor = pd.read_csv(r"C:\Users\user\Documents\task\task_data.csv",
                        index_col="sample index")
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(14,4))
sensor_names = ["sensor{}".format(i) for i in range(10)]
counter = 0
df_sensor["bin1"] = pd.cut(df_sensor["sensor1"], 10)

df_sensor["class_label2"]=df_sensor['class_label']==1
df_sensor["class_label3"]=df_sensor['class_label']==-1
df_binned = df_sensor[['bin1',
                       'class_label2',
                       'class_label3']].groupby(['bin1']).sum()

for axrow in axes:
    for ax in axrow:
        col_name = sensor_names[counter]
        ax.scatter(df_sensor[col_name], df_sensor['class_label'])
        counter = counter + 1
        