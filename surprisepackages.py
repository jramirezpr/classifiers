# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 13:03:23 2019

@author: user
"""

from surprise import BaselineOnly
#used for reading files
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate

# path to dataset file
file_path = os.path.expanduser('~/.surprise_data/ml-100k/ml-100k/u.data')

# As we're loading a custom dataset, we need to define a reader. In the
# movielens-100k dataset, each line has the following format:
# 'user item rating timestamp', separated by '\t' characters.
reader = Reader(line_format='user item rating timestamp', sep='\t')
data = Dataset.load_from_file(file_path, reader=reader)


# We can now use this dataset as we please, e.g. calling cross_validate
cross_validate(BaselineOnly(), data, verbose=True)

trainset, testset = train_test_split(data, test_size=.25)
algo = SVD()
algo.fit(trainset)
predictions = algo.test(testset)
accuracy.rmse(predictions)

## ---(Thu Sep 19 08:10:46 2019)---
import numpy as np
import scipy
import pandas as pd
import math
import random
import sklearn
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
articles_df = pd.read_csv(r'C:\Users\user\Desktop\articles-sharing-and-reading-from-ci-t-deskdrop\shared_articles.csv')
articles_df.head(5)
interactions_df = pd.read_csv(r'C:\Users\user\Desktop\articles-sharing-and-reading-from-ci-t-deskdrop\users_interactions.csv')
articles_df.head(5)
print(interactions_df.head(5))
event_type_strength = {
   'VIEW': 1.0,
   'LIKE': 2.0, 
   'BOOKMARK': 2.5, 
   'FOLLOW': 3.0,
   'COMMENT CREATED': 4.0,  
}

interactions_df['eventStrength'] = interactions_df['eventType'].apply(lambda x: event_type_strength[x])
users_interactions_count_df = interactions_df.groupby(['personId', 'contentId']).size().groupby('personId').size()
users_interactions_count_dfs
users_interactions_count_df
users_interactions_count_df = interactions_df.groupby(['personId', 'contentId']).size()
users_interactions_count_df
users_interactions_count_df = interactions_df.groupby(['personId', 'contentId']).size().groupby('personId').size()
users_interactions_count_df
print('# users: %d' % len(users_interactions_count_df))
type(users_interactions_count_df)
users_interactions_count_df[users_interactions_count_df>=5]
users_interactions_count_df[users_interactions_count_df>=5].reset_index()[['personId']]
users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= 5].reset_index()[['personId']]
print('# of interactions: %d' % len(interactions_df))
interactions_from_selected_users_df = interactions_df.merge(users_with_enough_interactions_df, 
               how = 'right',
               left_on = 'personId',
               right_on = 'personId')
print('# of interactions from users with at least 5 interactions: %d' % len(interactions_from_selected_users_df))
interactions_full_df = interactions_from_selected_users_df.groupby(['personId', 'contentId'])['eventStrength']
interactions_full_df
interactions_full_df = interactions_from_selected_users_df.groupby(['personId', 'contentId'])
interactions_full_df
interactions_full_df = interactions_from_selected_users_df.groupby(['personId', 'contentId'])['eventStrength'].sum()
interactions_full_df
def smooth_user_preference(x):
    return math.log(1+x, 2)
    
type(interactions_full_df = interactions_from_selected_users_df.groupby(['personId', 'contentId'])['eventStrength'].sum())
type(interactions_full_df)
def smooth_user_preference(x):
    return math.log(1+x, 2)
    
iteractions_full_df=iteractions_full_df.apply(smooth_user_preference)
type(interactions_full_df)
interactions_full_df=interactions_full_df.apply(smooth_user_preference)
interactions_full_df
interactions_full_df.reset_index()
interactions_full_df=interactions_full_df.reset_index()
interactions_train_df, interactions_test_df = train_test_split(interactions_full_df,
                                   stratify=interactions_full_df['personId'], 
                                   test_size=0.20,
                                   random_state=42)
interactions_full_indexed_df = interactions_full_df.set_index('personId')
interactions_train_indexed_df = interactions_train_df.set_index('personId')
interactions_test_indexed_df = interactions_test_df.set_index('personId')
interactions_train_indexed_df
def get_items_interacted(person_id, interactions_df):
    # Get the user's data and merge in the movie information.
    interacted_items = interactions_df.loc[person_id]['contentId']
    return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])
    
get_items_interacted(701021893037319987,interactions_train_indexed_df)
interactions_train_indexed_df
df = pd.DataFrame([[1, 2], [4, 5], [7, 8]],
index=['cobra', 'viper', 'sidewinder'],
columns=['max_speed', 'shield'])
df = pd.DataFrame([[1, 2], [4, 5], [7, 8]],
index=['cobra', 'cobra', 'sidewinder'],
columns=['max_speed', 'shield'])
df.loc['cobra']
sers_items_pivot_matrix_df = interactions_train_df.pivot(index='personId', 
                                                          columns='contentId', 
                                                          values='eventStrength')
users_items_pivot_matrix_df = interactions_train_df.pivot(index='personId', 
                                                          columns='contentId', 
                                                          values='eventStrength')
users_items_pivot_matrix_df
interactions_train_df