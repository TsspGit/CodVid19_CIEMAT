__author__ = '@Tssp'
from codvidutils.Transformation_class import Transformation
from keras.models import Model, Sequential
from keras.models import load_model
import numpy as np


def Transformation_main(train_path, test_path):
    '''Outputs the final encoded images for training or testing the
    following classification algorithm.
    '''
    trns = Transformation(train_path, test_path)
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
    #------------Use the best Autoencoder architecture------------#
    best_model_path = 'hdf_files/Autoencoder_covid_v5.hdf5'
    model = load_model(best_model_path)
    encoder = Model(model.layers[0].input, model.layers[6].output)
    encoder_test = encoder.predict(X_test)
    decoder_test = model.predict(X_test)
    print('encoder_test.shape', encoder_test.shape)
    encoder_train = encoder.predict(X_train)
    print('encoder_train.shape', encoder_train.shape)
    encoder_train = encoder_train.reshape((encoder_train.shape[0], 23*23*64))
    encoder_test = encoder_test.reshape((encoder_test.shape[0], 23*23*64))
    output = {'encoder_train': encoder_train,
             'encoder_test': encoder_test,
             'Y_test': Y_test,
             'Y_train': Y_train}
    return output