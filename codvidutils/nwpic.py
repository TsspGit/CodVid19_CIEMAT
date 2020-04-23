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

def prob (x,y):
    return x,y

def set_session(ngpu=None, ncpu=None, mode='max'):
    ''' This function sets the number of devices
    you are going to work with.
    Inputs: ngpu: number of GPUs.
            ncpu: number of CPUs.
            mode: "max" (default)
    Examples:
    - max:
        -> set_session()
    - And the manual mode is used if number of gpus and cpus is specified:
        -> set_session(1, 3)
    '''
    import tensorflow as tf
    from keras import backend as K
    import multiprocessing
    gpus = len(K.tensorflow_backend._get_available_gpus())
    cpus = multiprocessing.cpu_count()
    print("Num GPUs Available: ", gpus)
    print("Num CPUs Available: ", cpus)
    if mode == 'max':
        config = tf.ConfigProto(device_count={'GPU': gpus, 'CPU': cpus})
        sess = tf.Session(config=config)
        K.set_session(sess)
        print('---------Keras session created with---------\n - {} GPUs\n - {} CPUs'.format(gpus, cpus))
    elif (ngpu is not None) and (ncpu is not None):
        config = tf.ConfigProto(device_count={'GPU': ngpu, 'CPU': ncpu})
        sess = tf.Session(config=config)
        K.set_session(sess)
        print('---------Keras session created with---------\n - {} GPUs\n - {} CPUs'.format(ngpu, ncpu))
    else:
        raise ValueError('There are only two modes: manual and max.')
        
        