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
from matplotlib import pyplot
from matplotlib.colors import ListedColormap
from numpy import asarray, cov, isnan, mean, polyfit
from scipy.stats import pearsonr
from sklearn import datasets, metrics, preprocessing, tree
from sklearn.cluster import KMeans
# import sklearn
from sklearn.datasets import (make_blobs, make_circles,  # , load_iris
                              make_classification, make_hastie_10_2,
                              make_moons)
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                              ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import (LinearRegression, LogisticRegression,
                                  PassiveAggressiveClassifier, RidgeClassifier,
                                  SGDClassifier)
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import (KFold, LeaveOneOut, ShuffleSplit,
                                     cross_val_score, cross_validate,
                                     train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
# Encoding categorical features with preserving the missing values in incomplete features
from sklearn.preprocessing import (KBinsDiscretizer, LabelEncoder,
                                   OneHotEncoder, OrdinalEncoder,
                                   StandardScaler)
from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.tree import (DecisionTreeClassifier, DecisionTreeRegressor,
                          ExtraTreeClassifier)

import myutil  # myutil

# label_encoder = LabelEncoder()
# ordinal_encoder = OrdinalEncoder()
# hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)


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




def label_encoder(label_encoder,dataset,feature_list=['mode', 'music_genre']  ):
    for feature in feature_list:
        dataset[feature] = label_encoder.fit_transform(dataset[feature])
    return dataset


def ordinal_encoder(ordinal_encoder,dataset,feature_list=['key', 'obtained_date']  ):
    '''encode default ['key','obtained_date'] '''
    # feature_list = ['key', 'obtained_date']  # ,'mode']

    for feature in feature_list:
        dataset[feature] = ordinal_encoder.fit_transform(dataset[feature].values.reshape(-1,1))


    return dataset



# TODO: https://stackoverflow.com/questions/57507832/unable-to-allocate-array-with-shape-and-data-type
def one_hot_encoder(hot_encoder,dataset, feature_list=['artist_name', 'track_hash', 'track_name']):
    ''' 3rd, use one hot encoder to encode 'artist_name','track_hash','track_name'''
    # feature_list = ['artist_name', 'track_hash', 'track_name']

    for i in range(len(feature_list)-1):
        dataset[feature_list[i]] = hot_encoder.fit_transform(
            dataset[feature_list[i]].to_numpy().reshape(-1, 1))

    return dataset



def plot_heatmap(cor, name):
    """plot_heatmap by the cor parameter
    Args:
        cor ([type]): corelation value that you want to use to plot
    """
    fig = plt.figure(figsize=(15, 9))
    sns.heatmap(cor,  cmap='Blues', annot=True)
    fig.suptitle("Heatmap of "+ name)


# get a list of models to evaluate
def get_models():
    '''get a list of models to evaluate'''
    models = list()
    models.append(LogisticRegression())
    models.append(RidgeClassifier())
    models.append(SGDClassifier())
    models.append(PassiveAggressiveClassifier())
    models.append(KNeighborsClassifier())
    models.append(DecisionTreeClassifier())
    models.append(ExtraTreeClassifier())
    models.append(LinearSVC())
    models.append(SVC())
    models.append(GaussianNB())
    models.append(AdaBoostClassifier())
    models.append(BaggingClassifier())
    models.append(RandomForestClassifier())
    models.append(ExtraTreesClassifier())
    models.append(GaussianProcessClassifier())
    models.append(GradientBoostingClassifier())
    models.append(LinearDiscriminantAnalysis())
    models.append(QuadraticDiscriminantAnalysis())
    return models


# def encode_class_label(whole_noMissing):
#     # le = LabelEncoder()
#     # lab = le.fit_transform(whole_noMissing.music_genre)
#     # class_lab = pd.DataFrame(lab, columns = ["music_genre"])
#     # whole_noMissing['music_genre'] = class_lab.values
#     lbe = LabelEncoder()
#     whole_noMissing['music_genre'] = lbe.fit_transform(
#         whole_noMissing['music_genre'])
#     return whole_noMissing


# def encode_feature_mode(whole_noMissing):
#     '''mode has only major and minor, so use label encoder is fine'''
#     # le = LabelEncoder()
#     # lab = le.fit_transform(whole_noMissing.music_genre)
#     # class_lab = pd.DataFrame(lab, columns = ["mode"])
#     # whole_noMissing['mode'] = class_lab.values
#     lbe = LabelEncoder()
#     whole_noMissing['mode'] = lbe.fit_transform(whole_noMissing['mode'])
#     return whole_noMissing
