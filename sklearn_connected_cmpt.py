# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 13:16:13 2019

@author: Guest
"""

# import matplotlib.pyplot as plt

from skimage.data import coins
from skimage.transform import rescale
from scipy.ndimage.filters import gaussian_filter
import numpy as np
from sklearn.feature_extraction.image import grid_to_graph
# from sklearn.cluster import AgglomerativeClustering


# #############################################################################
# Generate data
orig_coins = coins()

# Resize it to 20% of the original size to speed up the processing
# Applying a Gaussian filter for smoothing prior to down-scaling
# reduces aliasing artifacts.
smoothened_coins = gaussian_filter(orig_coins, sigma=2)
rescaled_coins = rescale(smoothened_coins,
                         0.2,
                         multichannel=False,
                         mode="reflect")

X = np.reshape(rescaled_coins, (-1, 1))

# #############################################################################
# Define the structure A of the data. Pixels connected to their neighbors.
connectivity = grid_to_graph(*rescaled_coins.shape)
