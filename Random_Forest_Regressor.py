__author__ = '@Tssp'
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from codvidutils.Autoencoder_Transformation_main import Transformation_main
from sklearn.ensemble import RandomForestRegressor

outputs = Transformation_main('data/train_split_v4.csv', 'data/test_split_v4.csv')
Y_test = outputs['Y_test']
Y_train = outputs['Y_train']
encoder_train = outputs['encoder_train']
encoder_test = outputs['encoder_test']
del outputs

#------------Random Forest------------#
# Regressor:
RF = RandomForestRegressor(n_estimators=200, n_jobs=-1)
RF.fit(encoder_train, Y_train)
encoder_test = encoder_test.reshape((encoder_test.shape[0], 23*23*64))
preds = RF.predict(encoder_test)
nocovid = preds[np.where(Y_test == 0)]
covid = preds[np.where(Y_test == 1)]
print("\n\n---------- Predictions ----------\n")
print("preds = ", preds)
np.savetxt('log/preds_RFr_v4.txt', preds, delimiter=',')
