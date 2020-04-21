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