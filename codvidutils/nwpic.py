__author__ = '@iRodGit'

import numpy as np
import pandas as pd

def new_pictures(picture, strides=5, kernel=(180,180)):
    "This function generate new picutes from one. Like as Convolutional filter"
    news = []
    for i in range(int((picture.shape[0]-kernel[0])/strides)+1):
        for j in range(int((picture.shape[1]-kernel[1])/strides)+1):
            s,k = strides*i, strides*j
            print(s,k)
            news.append(picture[s:kernel[0]+s,k:kernel[1]+k])
    
    return news

def new_pictures_arrays(pictures, strides=5, kernel=(180,180)):
    
    news = []
    for i in range(int((pictures.shape[1]-kernel[0])/strides)+1):
        for j in range(int((pictures.shape[2]-kernel[1])/strides)+1):
            s,k = strides*i, strides*j
            news.append(pictures[:,s:kernel[0]+s,k:kernel[1]+k])
    news= np.array(news)
    
    return news.reshape(news.shape[0]*news.shape[1], kernel[0],kernel[1],pictures.shape[3])

def underbalance_imgs(diseaseID, X):
    '''
    This function balances the data:
    Inputs: disease_ID: array of the train and test categories.
            X: array of the pics.
    Usage:
        -> disease_ID, X = balance_imgs(disease_ID, X)
    '''
    from collections import Counter
    from imblearn.under_sampling import RandomUnderSampler
    counter = Counter (diseaseID)
    print('Count of classes: ', counter)
    dicto = {2: 4500, 0: 4500, 1: 187}
    X = X.reshape(X.shape[0],-1)
    under = RandomUnderSampler(sampling_strategy =dicto)
    X, diseaseID = under.fit_resample(X, diseaseID)
    # summarize class distribution
    print('New diseaseID shape: ', diseaseID.shape)
    print('New X shape: ', X.shape)
    print('New Count of classes: ', Counter (diseaseID))
    return diseaseID, X