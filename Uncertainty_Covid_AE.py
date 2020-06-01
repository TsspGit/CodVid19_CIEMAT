''' RNN Bidireccional en Keras '''
__author__ = '@Tssp'
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from codvidutils.Transformation_class import Transformation
import tensorflow as tf
from keras.layers import Conv2D, Input, Dense, MaxPooling2D, UpSampling2D, Flatten, Dropout
from tensorflow.keras.regularizers import l2
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam

trns = Transformation('data/train_split_v4.csv', 'data/test_split_v5.csv')
#------------Read images and prepare dataset------------#
train_class, test_class = trns.read_imgs_paths()
X_train, X_test, diseaseID_train, diseaseID_test = trns.read_imgs(train_class, test_class)
del train_class, test_class
#------------Imbalanced methods------------#
X_train, X_test, diseaseID_train = trns.underbalance(X_train, X_test, diseaseID_train)
print('X_train.shape: {}\nX_test.shape: {}'.format(X_train.shape, X_test.shape))
print("Normal train: ",diseaseID_train[diseaseID_train==0].shape)
print("Pneumonia train: ",diseaseID_train[diseaseID_train==2].shape)
print("COVID train: ",diseaseID_train[diseaseID_train==1].shape)
print("*******************************************************")
print("Normal test: ",diseaseID_test[diseaseID_test==0].shape)
print("Pneumonia test: ",diseaseID_test[diseaseID_test==2].shape)
print("COVID test: ",diseaseID_test[diseaseID_test==1].shape)
X_train, X_test, diseaseID_train, diseaseID_test = trns.new_imgs(X_train, X_test, diseaseID_train, diseaseID_test)
Y_train = np.copy(diseaseID_train)
del diseaseID_train
Y_train[Y_train==2]=0
Y_test = np.copy(diseaseID_test)
Y_test[Y_test==2]=0
X_test = X_test/255
X_train = X_train/255

def autoencoder_model(p_drop, p_l2):
    input_img = Input(shape=X_train.shape[1:])
    encoder = Dropout(rate = p_drop)(input_img)
    encoder = Conv2D(256+25, (3, 3), activation='relu', padding='same', name='Econv2d_1', 
                     bias_regularizer = l2(p_l2), kernel_regularizer=l2(p_l2))(input_img)
    encoder = MaxPooling2D((2,2), padding='same', name='Emaxpool2d_1')(encoder)
    encoder = Dropout(rate = p_drop)(encoder)
    encoder = Conv2D(128+12, (3, 3), activation='relu', padding='same', name='Econv2d_2', 
                     bias_regularizer = l2(p_l2), kernel_regularizer=l2(p_l2))(encoder)
    encoder = MaxPooling2D((2,2), padding='same', name='Emaxpool2d_2')(encoder)
    encoder = Dropout(rate = p_drop)(encoder)
    encoder = Conv2D(64+6, (3, 3), activation='relu', padding='same', name='Econv2d_3', 
                     bias_regularizer = l2(p_l2), kernel_regularizer=l2(p_l2))(encoder)
    encoder = MaxPooling2D((2,2), padding='same', name='Emaxpool2d_3')(encoder)

    decoder = Dropout(rate = p_drop)(encoder)
    decoder = Conv2D(64+6, (3,3), activation='relu', padding='same', name='Dconv2d_1', 
                     bias_regularizer = l2(p_l2), kernel_regularizer=l2(p_l2))(decoder)
    decoder = UpSampling2D((2, 2), name='Dupsamp_1')(decoder)
    decoder = Dropout(rate = p_drop)(decoder)
    decoder = Conv2D(128+12, (3, 3), activation='relu', padding='same', name='Dconv2d_2', 
                     bias_regularizer = l2(p_l2), kernel_regularizer=l2(p_l2))(decoder)
    decoder = UpSampling2D((2, 2), name='Dupsamp_2')(decoder)
    decoder = Dropout(rate = p_drop)(decoder)
    decoder = Conv2D(256+25, (3, 3), activation='relu', name='Dconv2d_3', 
                     bias_regularizer = l2(p_l2), kernel_regularizer=l2(p_l2))(decoder)
    decoder = UpSampling2D((2, 2), name='Dupsamp_3')(decoder)
    decoder = Dropout(rate = p_drop)(decoder)
    decoder = Conv2D(3, (3, 3), activation='sigmoid', padding='same', name='Dconv2d_out')(decoder)

    autoencoder = Model(input_img, decoder)
    autoencoder.summary()
    return autoencoder

p_drop = 0.1
p_l2 = 0.005
for it in range(5):
    tf.keras.backend.clear_session()
    best_model_path = 'hdf_files/Uncertainty_AE_Covid_{}.hdf5'.format(it+1)
    checkpoint = ModelCheckpoint(best_model_path, monitor='loss', verbose=1, save_best_only=True, mode='min')
    amsgrad = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=True)
    autoencoder = autoencoder_model(p_drop, p_l2)
    autoencoder.compile(optimizer=amsgrad, loss='mse', metrics=['acc', 'mse'])
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        history = autoencoder.fit(X_train, X_train, epochs=10, batch_size=64, shuffle=False, callbacks=[checkpoint],
                                  validation_data=(X_test, X_test))