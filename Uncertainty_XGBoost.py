__author__ = '@Tssp'
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import xgboost as xgb
from codvidutils.Autoencoder_Uncertainty_Transformation_main import Transformation_main
import pickle

for it in range(0, 10):
    model = 'hdf_files/Uncertainty_AE_Covid.hdf5'
    outputs = Transformation_main('data/train_split_v4.csv', 'data/test_split_v5.csv', model)
    Y_test = outputs['Y_test']
    Y_train = outputs['Y_train']
    encoder_train = outputs['encoder_train']
    encoder_test = outputs['encoder_test']
    del outputs

    #------------XGBoost------------#
    # Regressor:
    lr = 0.005
    n_trees = 250
    depth = 3
    print('#'*10 +' max depth = {} '.format(depth)+ '#'*10)
    xgbr = xgb.XGBRegressor(objective ='reg:logistic', learning_rate = lr, n_estimators = n_trees, max_depth=depth, n_jobs=-1)
    xgbr.fit(encoder_train, Y_train)
    # save model
    pickle.dump(xgbr, open('log/XGBr_uncmodel_{}.pkl'.format(it+1), "wb"))
    encoder_test = encoder_test.reshape((encoder_test.shape[0], 23*23*70))
    preds = xgbr.predict(encoder_test)
    nocovid = preds[np.where(Y_test == 0)]
    covid = preds[np.where(Y_test == 1)]
    print("\n\n---------- Predictions ----------\n")
    print("preds = ", preds)
    np.savetxt('log/preds_XGBr_Uncertainty_{}.txt'.format(it+1), preds, delimiter=',')
    del encoder_train, encoder_test, xgbr, preds, nocovid, covid, Y_test, Y_train
