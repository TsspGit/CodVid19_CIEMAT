__author__ = '@Tssp'
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from codvidutils.Autoencoder_Transformation_main import Transformation_main
from sklearn.ensemble import RandomForestRegressor

outputs = Transformation_main('data/train_split_v4.csv', 'data/test_split_v4.csv')
Y_test = outputs['Y_test']
encoder_train = outputs['encoder_train']
encoder_test = outputs['encoder_test']
del outputs

#------------Random Forest------------#
# Regressor:
RF = RandomForestRegressor(n_estimators=200, n_jobs=-1)
RF.fit(encoder_RF_train, Y_train)
encoder_imgs = encoder_imgs.reshape((encoder_imgs.shape[0], 23*23*128))
preds = RF.predict(encoder_imgs)
nocovid = preds[np.where(Y_test == 0)]
covid = preds[np.where(Y_test == 1)]
print("\n\n---------- Predictions ----------\n")
print("preds = ", preds)
np.savetxt('preds_RFr_v4.txt', preds, delimiter=',')