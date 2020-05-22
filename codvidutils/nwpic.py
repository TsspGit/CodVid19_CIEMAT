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

def underbalance_imgs(diseaseID, X,dicto):
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
    X = X.reshape(X.shape[0],-1)
    under = RandomUnderSampler(sampling_strategy =dicto)
    X, diseaseID = under.fit_resample(X, diseaseID)
    # summarize class distribution
    print('New diseaseID shape: ', diseaseID.shape)
    print('New X shape: ', X.shape)
    print('New Count of classes: ', Counter (diseaseID))
    return diseaseID, X

def adding_images(X, Y,strides= 5):#kernel =None ):
    from codvidutils import nwpic as nw
    from numpy import concatenate
    new_X = nw.new_pictures_arrays(X[Y==1],strides)
    X = X[:,10:190,10:190]
    new_Y = np.ones(new_X.shape[0])
    X = concatenate([X,new_X],axis=0)
    Y = concatenate([Y,new_Y],axis=0)
    return X, Y

def load_pictures (data_frame,data_dir,channels):
    import numpy as np
    from PIL import Image
    pics = []
    for img in data_frame['image_path'].values:
        if channels ==3:
            pics.append(np.array(Image.open(data_dir + img))[:, :,:3])
        else:
            pics.append(np.array(Image.open(data_dir + img))[:, :,0])

    return np.array(pics)

