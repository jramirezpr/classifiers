# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 09:06:03 2019

@author: Juan Carlos Ramirez
"""
import scipy
from sklearn import neighbors
from sklearn import model_selection
import pandas as pd
import matplotlib.pyplot as plt


DF_SENSOR = pd.read_csv(r"C:\Users\user\Documents\task\task_data.csv",
                        index_col="sample index")
SENSOR_NAMES = ["sensor{}".format(i) for i in range(10)]


def compute_relval(row):
    """compute relative amount of 1's"""
    return (row['class_label2']
            / max(row['class_label2'] + row['class_label3'], 0)
            )


def relative_bin_val(df_sensor, sensor_name):
    """return DataFrame with within-bin proportion of
    elements with label 1"""
    bins = sensor_name + "bins"
    df_sensor[bins] = pd.cut(df_sensor[sensor_name], 10)
    df_sensor["class_label2"] = df_sensor['class_label'] == 1
    df_sensor["class_label3"] = df_sensor['class_label'] == -1
    df_binned = df_sensor[[bins,
                           'class_label2',
                           'class_label3']].groupby([bins]).sum()
    catname = 'class_1 in_bin concentration'
    df_binned[catname] = df_binned.apply(compute_relval, axis=1)
    old_ind = list(df_binned.index.values)
    df_binned = df_binned.rename(index=dict(zip(old_ind, range(10))))
    return df_binned


def plots_per_sensor(df_sensor):
    """plot all sensor bar charts of the relative
    frequency of class 1 within a bin (each bar corres
    ponds to an equally spaced bin of sensor values).
    Used for data visualization ,best predictors
    are the sensors corresponding
    to the plots where the bars are farthest away from 0.5
    """
    counter = 0
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(14, 8))
    for i in range(4):
        for j in range(3):
            if (i * 3 + j) <= 9:
                col_name = SENSOR_NAMES[counter]
                df_binned = relative_bin_val(df_sensor, col_name)
                df_binned.plot.bar(ax=axes[i, j],
                                   y='class_1 in_bin concentration')
                counter = counter + 1
    fig.tight_layout()


def split(df_sensor,test_size):
    """function splits into train and test data"""
    return model_selection.train_test_split(
        df_sensor[SENSOR_NAMES],
        df_sensor['class_label'],
        test_size=test_size,
        random_state=42)


def train_sensor_classifier(x_train, y_train):
    """ function trains the k-nearest-neighbors classifier with
    the training data. the number of neighbors chosen k is tuned by
    choosing the k with the maximal score in cross-validation.
    We use K-fold cross validation (not the same K as number of neighbors)
    in this case, 4 splits used"""
    kf_4 = model_selection.KFold(n_splits=4, shuffle=True, random_state=42)
    max_score = 0
    best_k = 1
    score_list = []
    for i in range(5, 50):
        knn = neighbors.KNeighborsClassifier(
            n_neighbors=i,
            weights='distance'
            )
        score = model_selection.cross_val_score(
            knn,
            x_train,
            y_train,
            cv=kf_4,
            scoring='accuracy'
            ).mean()
        score_list.append(score)
        if score > max_score:
            max_score = score
            best_k = i
    return {"scores": score_list,
            "best score": max_score,
            "best k": best_k
            }


def rank_sensors(df_sensor,test_size):
    """function ranks sensors by percentage of prediction accuracy"""
    x_train, x_test, y_train, y_test = split(df_sensor, test_size)
    top_scores = []
    test_scores = []
    for sensor in SENSOR_NAMES:
        x_sensor = x_train[sensor]
        x_sensor = x_sensor.values.reshape(-1, 1)
        res_dict = train_sensor_classifier(x_sensor, y_train)
        top_scores.append((res_dict["best score"].copy(), sensor))
        knn = neighbors.KNeighborsClassifier(
            n_neighbors=res_dict["best k"],
            weights='distance'
            )
        x_test_ar = x_test[sensor].values.reshape(-1, 1)
        knn.fit(x_sensor, y_train)
        test_scores.append((knn.score(x_test_ar, y_test), sensor))
    top_scores.sort()
    test_scores.sort()
    return [top_scores, test_scores]


TRAIN_RANK, TEST_RANK = rank_sensors(DF_SENSOR,test_size =0.1)
TRAIN_RANK2, TEST_RANK = rank_sensors(DF_SENSOR,test_size = 0.5)

TRAIN_RANK_NUM = [int(val[1][6:]) for val in TRAIN_RANK]
TRAIN_RANK_NUM.reverse()
for x in TRAIN_RANK_NUM:
    print(x)

TEST_RANK_NUM = [int(val[1][6:]) for val in TEST_RANK]
# to figure out the kendalltau statistic, uncomment the following
# when running
#TRAIN_RANK_NUM = [int(val[1][6:]) for val in TRAIN_RANK2]
#print(scipy.stats.kendalltau(TRAIN_RANK_NUM, TEST_RANK_NUM)[0])
