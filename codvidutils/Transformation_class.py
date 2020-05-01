__author__ = '@Tssp'
import numpy as np
import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
# Keras NN:
from keras.utils import to_categorical
# My utils:
from codvidutils.imageproc import map_categorical
from codvidutils.cudasession import set_session
from codvidutils import nwpic as nw

class Transformation:
    '''This class contains all the methods to transform the dataset for its proper input
    to the neural network architectures. We assume that the neural network is going to be
    convolutional, for carry out image classification.
    '''
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        print('You are going to read from: {} and {}'.format(self.train_path, self.test_path))
        
    def read_imgs_paths(self):
        '''Read the data-paths and prepare the categorical
        variables.
        
        
        Parameters
        ----------
        train_path : {string} path to read the train image names.

        test_path : {string} path to read the test image names.


        Returns
        -------
        train_class : {pandas DataFrame} Includes the image_path to read the images and its corresponding categorical class {normal: 0, COVID-19: 1, pneumonia: 2}

        test_class : {pandas DataFrame} Includes the image_path to read the images and its corresponding categorical class {normal: 0, COVID-19: 1, pneumonia: 2}  
       '''
        train_class = pd.read_csv(self.train_path, sep=' ', header=1, names=['image_path', 'class'])
        test_class = pd.read_csv(self.test_path, sep=' ', header=1, names=['image_path', 'class'])
        values_dict = {'pneumonia': 2, 'COVID-19': 1, 'normal': 0}
        test_class['class_categorical'] = test_class['class'].apply(map_categorical, args=(values_dict,))
        train_class['class_categorical'] = train_class['class'].apply(map_categorical, args=(values_dict,))
        return train_class, test_class
        
    def read_imgs(self, train_class, test_class):
        '''This method read the images.
        
        Parameters
        ----------
        train_class : {pandas Dataframe} output form read_imgs_paths()
        test_class : {pandas Dataframe} output from read_imgs_paths()
        
        Returns
        -------
        X_train : {numpy array} train images.
        X_test : {numpy array} test images.
        diseaseID_train : {numpy array} cateogrical column from the train_class
        diseaseID_test : {numpy array} categorical column from the test_class
        '''
        pics = []
        for img in train_class['image_path'].values:
            pics.append(np.array(Image.open('data/train/' + img))[:, :,:3])
        X_train = np.array(pics)
        pics = []
        for img in test_class['image_path'].values:
            pics.append(np.array(Image.open('data/test/' + img))[:, :, :3])
        X_test = np.array(pics)
        print("Total number of images:", len(pics))
        diseaseID_train = np.asarray(train_class["class_categorical"])
        diseaseID_test = np.asarray(test_class["class_categorical"])
        print('shape X: {} {},  disease_ID (Y): {} {}'.format(X_train.shape[0], X_test.shape[0], diseaseID_train.shape[0], diseaseID_test.shape[0] ))
        return X_train, X_test, diseaseID_train, diseaseID_test
    
    def underbalance(self, X_train, X_test, diseaseID_train):
        '''This method underbalance the most populate class in the train dataset.
        
        Parameters
        ----------
        X_train : {numpy array} train images.
        diseaseID_train : {numpy array} cateogrical column from the train_class.
        X_test : {numpy array} test images.
        
        
        Returns
        -------
        X_train : {numpy array} underbalance train images.
        X_test : {numpy array} reshaped test images.
        diseaseID_train : {numpy array} underbalance cateogrical column from the train_class.
        '''
        diseaseID_train, X_train = nw.underbalance_imgs(diseaseID_train, X_train)
        # summarize class distribution
        print('Undersample shapes:\ndiseaseID_train.shape: {}\nX_train.shape: {}'.format(diseaseID_train.shape, X_train.shape))
        X_train = X_train.reshape(X_train.shape[0],200,200,3)
        X_test = X_test.reshape(X_test.shape[0],200,200,3)
        return X_train, X_test, diseaseID_train
    
    def new_imgs(self, X_train, X_test, diseaseID_train, diseaseID_test):
        '''This method create new images on the train and test set.
        
        Parameters
        ----------
        X_train : {numpy array} train images.
        X_test : {numpy array} test images.
        diseaseID_train : {numpy array} cateogrical column from the train_class.
        diseaseID_test : {numpy array} cateogrical column from the test_class.
        
        
        Returns
        -------
        X_train : {numpy array} new train images.
        X_test : {numpy array} new test images.
        diseaseID_train : {numpy array} new cateogrical column from the train_class.
        diseaseID_test : {numpy array} new cateogrical column from the test_class.
        '''
        # News images to train 
        X_train_news = nw.new_pictures_arrays(X_train[diseaseID_train==1])
        print(X_train_news.shape)
        diseaseID_train_news = np.ones(X_train_news.shape[0])
        print(diseaseID_train_news.shape)
        X_train = X_train[:,10:190,10:190]
        X_train = np.concatenate([X_train,X_train_news],axis=0)
        diseaseID_train = np.concatenate([diseaseID_train,diseaseID_train_news],axis=0)
        del X_train_news, diseaseID_train_news
        print('X_train.shape: ', X_train.shape)
        print('diseaseID_train.shape: ', diseaseID_train.shape)

        # News images to test 
        X_test_news = nw.new_pictures_arrays(X_test[diseaseID_test==1])
        print('X_test_news.shape: ', X_test_news.shape)
        diseaseID_test_news = np.ones(X_test_news.shape[0])
        print('diseaseID_test_news.shape: ', diseaseID_test_news.shape)
        X_test = X_test[:,10:190,10:190]
        X_test = np.concatenate([X_test,X_test_news],axis=0)
        diseaseID_test = np.concatenate([diseaseID_test,diseaseID_test_news],axis=0)
        del X_test_news, diseaseID_test_news
        print('X_test.shape: ', X_test.shape)
        print('diseaseID_test.shape: ', diseaseID_test.shape)
        return X_train, X_test, diseaseID_train, diseaseID_test