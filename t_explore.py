# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 09:06:03 2019

@author: user
"""

import pandas as pd
import matplotlib.pyplot as plt

DF_SENSOR = pd.read_csv(r"C:\Users\user\Documents\task\task_data.csv",
                        index_col="sample index")
SENSOR_NAMES = ["sensor{}".format(i) for i in range(10)]


def compute_relval(row):
    return (row['class_label2']
            / max(row['class_label2'] + row['class_label3'], 0)
            )


def relative_bin_val(sensor_name, df_sensor):
    bins = sensor_name + "bins"
    df_sensor[bins] = pd.cut(df_sensor[sensor_name], 10)
    df_sensor["class_label2"] = df_sensor['class_label'] == 1
    df_sensor["class_label3"] = df_sensor['class_label'] == -1
    df_binned = df_sensor[[bins,
                           'class_label2',
                           'class_label3']].groupby([bins]).sum()
    df_binned['class_1 in_bin concentration'] = df_binned.apply(compute_relval, axis=1)
    old_ind = list(df_binned.index.values)
    df_binned = df_binned.rename(index=dict(zip(old_ind, range(10))))
    return df_binned


def plots_per_sensor(df_sensor):
    counter = 0
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(14, 8))
    for i in range(4):
        for j in range(3):
            if(i * 3 + j <= 9):
                col_name = SENSOR_NAMES[counter]
                df_binned = relative_bin_val(col_name, df_sensor)
                df_binned.plot.bar(ax=axes[i, j], y='class_1 in_bin concentration')
                counter = counter + 1
    fig.tight_layout()
