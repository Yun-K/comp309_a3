'''this script store the helpful method that can be invoked by .ipynb notebook.'''

# importing pandas, and other necessary modules
import glob
import io
import sys

import matplotlib.image as mpimg
import matplotlib.pyplot as plt  # visualising
import missingno as msno
import numpy as np  # linear algebra
# import pandas, and other necessary modules
import pandas as pd  # data processing
# easy for structing the report , need pip install pandas-profiling first
import pandas_profiling as pp
import phik
import pydotplus
import seaborn as sns  # visualising
import sklearn
from IPython.display import Image
from numpy import cov
from scipy.stats import pearsonr
from sklearn import datasets, tree
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
# Encoding categorical features with preserving the missing values in incomplete features
from sklearn.preprocessing import (KBinsDiscretizer, LabelEncoder,
                                   OneHotEncoder, OrdinalEncoder,
                                   StandardScaler)
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def print_NA_details(dataset):
    '''method for checking whether the passed in dataset has missing (i.e. NA) fetatures '''
    missing = dataset.isnull().sum()
    missing = missing[missing > 0]
    print('\nmissing feature and values:\n', missing)


def replace_by_mean(whole_noMissing):
    # duration_ms_mean =  whole_noMissing["duration_ms"].astype('float').mean(axis=0)
    # tempo_mean = whole_noMissing["tempo"].astype('float').mean(axis=0)
    # print(duration_ms_mean)
    # print(tempo_mean)
    whole_noMissing['tempo'].fillna(
        (whole_noMissing['tempo'].mean()), inplace=True)
    whole_noMissing['duration_ms'].fillna(
        (whole_noMissing['duration_ms'].mean()), inplace=True)
    return whole_noMissing


def encode_class_label(whole_noMissing):
    # le = LabelEncoder()
    # lab = le.fit_transform(whole_noMissing.music_genre)
    # class_lab = pd.DataFrame(lab, columns = ["music_genre"])
    # whole_noMissing['music_genre'] = class_lab.values
    lbe = LabelEncoder()
    whole_noMissing['music_genre'] = lbe.fit_transform(
        whole_noMissing['music_genre'])
    return whole_noMissing


def encode_feature_mode(whole_noMissing):
    '''mode has only major and minor, so use label encoder is fine'''
    # le = LabelEncoder()
    # lab = le.fit_transform(whole_noMissing.music_genre)
    # class_lab = pd.DataFrame(lab, columns = ["mode"])
    # whole_noMissing['mode'] = class_lab.values
    lbe = LabelEncoder()
    whole_noMissing['mode'] = lbe.fit_transform(whole_noMissing['mode'])

    return whole_noMissing


def ordinal_encoder(dataset,feature_list=['key', 'obtained_date']  ):
    
    '''encode default ['key','obtained_date'] '''
    # feature_list = ['key', 'obtained_date']  # ,'mode']

    encoder = OrdinalEncoder()
    for i in range(len(dataset)):
        dataset[feature_list[i]] = encoder.fit_transform(
            dataset[feature_list[i]].to_numpy().reshape(-1, 1))

    return dataset

    # ordinal_encoded = encoder.fit_transform(dataset.key.values.reshape(-1, 1))
    # key_feature = pd.DataFrame(ordinal_encoded, columns=["key"])
    # dataset['key'] = key_feature.values

    # encoder = OrdinalEncoder()
    # ordinal_encoded = encoder.fit_transform(
    #     dataset.obtained_date.values.reshape(-1, 1))
    # obtained_date_feature = pd.DataFrame(
    #     ordinal_encoded, columns=["obtained_date"])
    # dataset['obtained_date'] = obtained_date_feature.values

    # encoder = OrdinalEncoder()
    # ordinal_encoded = encoder.fit_transform(dataset.track_name.values.reshape(-1, 1))
    # track_name_feature = pd.DataFrame(ordinal_encoded, columns = ["track_name"])
    # dataset['track_name'] = track_name_feature.values

    # ordinal_encoded = encoder.fit_transform(dataset.mode.values.reshape(-1, 1))
    # mode_fea = pd.DataFrame(ordinal_encoded, columns = ["mode"])
    # dataset['mode'] = mode_fea.values
    return dataset


# TODO: https://stackoverflow.com/questions/57507832/unable-to-allocate-array-with-shape-and-data-type
def one_hot_encoder(dataset, feature_list=['artist_name', 'track_hash', 'track_name']):
    ''' 3rd, use one hot encoder to encode 'artist_name','track_hash','track_name'''
    # feature_list = ['artist_name', 'track_hash', 'track_name']

    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    for i in range(len(feature_list)):
        dataset[feature_list[i]] = encoder.fit_transform(
            dataset[feature_list[i]].to_numpy().reshape(-1, 1))

    return dataset
# pd.DataFrame(enc.fit_transform(bridge_df[['Bridge_Types_Cat']]).toarray())
    # encoded_set = encoder.transform(dataset)
    # return encoded_set

    # hot_encoded = encoder.fit_transform(dataset.artist_name.values().reshape(-1,1))

    # encoder = OneHotEncoder()
    # hot_encoded = encoder.fit_transform(dataset.artist_name.values().reshape(-1,1))
    # artist_name = pd.DataFrame(hot_encoded, columns = ["artist_name"])
    # dataset['artist_name'] = artist_name.values

    # encoder = OneHotEncoder()
    # hot_encoded = encoder.fit_transform(dataset.track_hash.values().reshape(-1,1))
    # track_hash = pd.DataFrame(hot_encoded, columns = ["track_hash"])
    # dataset['track_hash'] = track_hash.values
    # track_hash

    # encoder = OneHotEncoder()
    # hot_encoded = encoder.fit_transform(dataset.track_name.values().reshape(-1,1))
    # track_name = pd.DataFrame(hot_encoded, columns = ["track_name"])
    # dataset['track_name'] = track_name.values
