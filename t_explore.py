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
    df_sensor["bin1"] = pd.cut(df_sensor[sensor_name], 10)
    df_sensor["class_label2"] = df_sensor['class_label'] == 1
    df_sensor["class_label3"] = df_sensor['class_label'] == -1
    df_binned = df_sensor[['bin1',
                           'class_label2',
                           'class_label3']].groupby(['bin1']).sum()
    df_binned['rel_val'] = df_binned.apply(compute_relval, axis=1)
    return df_binned


def plots_per_sensor(df_sensor):
    counter = 0
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(14, 4))

    for i in range(3):
        for j in range(3):
            col_name = SENSOR_NAMES[counter]
            df_binned = relative_bin_val(col_name, df_sensor)
            df_binned.plot.bar(ax=axes[i, j], y='rel_val')
            counter = counter + 1
