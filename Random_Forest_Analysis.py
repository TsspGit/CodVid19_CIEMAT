__author__ = '@Tssp'
import numpy as np
import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
# Keras NN:
from keras.utils import to_categorical
from keras.models import Model, Sequential
from keras.models import load_model
# My utils:
from codvidutils.imageproc import map_categorical
from codvidutils.cudasession import set_session
from codvidutils import nwpic as nw

set_session()
train_class = pd.read_csv('data/train_split_v3.csv', sep=' ', header=1, names=['patientID', 'image_path', 'class'])
test_class = pd.read_csv('data/test_split_v3.csv', sep=' ', header=1, names=['patientID', 'image_path', 'class'])
values_dict = {'pneumonia': 2, 'COVID-19': 1, 'normal': 0}
test_class['class_categorical'] = test_class['class'].apply(map_categorical, args=(values_dict,))
train_class['class_categorical'] = train_class['class'].apply(map_categorical, args=(values_dict,))
diseaseID_train = np.asarray(train_class["class_categorical"])
diseaseID_test = np.asarray(test_class["class_categorical"])
diseaseID = np.concatenate([diseaseID_train, diseaseID_test],axis=0)

pics = []
for img in train_class['image_path'].values:
    pics.append(np.array(Image.open('data/train/' + img))[:, :,:3])
for img in test_class['image_path'].values:
    pics.append(np.array(Image.open('data/test/' + img))[:, :, :3])

print("Total number of images:", len(pics))
del train_class, test_class

X = np.array(pics)
del pics
print('shape X: {},  disease_ID (Y): {}'.format(X.shape, diseaseID.shape ))

diseaseID, X = nw.underbalance_imgs(diseaseID, X)
n = np.random.randint(1000,6760)
from sklearn.utils import shuffle
X, diseaseID = shuffle(X, diseaseID, random_state=n)
from sklearn.model_selection import train_test_split
m = np.random.randint(1000,6760)
X_train, X_test, diseaseID_train, diseaseID_test = train_test_split(X, diseaseID, test_size=0.20, random_state=m)
del X, diseaseID

X_train = X_train.reshape(X_train.shape[0],200,200,3)
X_test = X_test.reshape(X_test.shape[0],200,200,3)
print('X_train.shape: {}\nX_test.shape: {}'.format(X_train.shape, X_test.shape))

print("Normal train: ",diseaseID_train[diseaseID_train==0].shape)
print("Pneumonia train: ",diseaseID_train[diseaseID_train==2].shape)
print("COVID train: ",diseaseID_train[diseaseID_train==1].shape)
print("*******************************************************")
print("Normal test: ",diseaseID_test[diseaseID_test==0].shape)
print("Pneumonia test: ",diseaseID_test[diseaseID_test==2].shape)
print("COVID test: ",diseaseID_test[diseaseID_test==1].shape)

"""
News images to train 
"""
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

"""
News images to test 
"""
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

Y_train = np.copy(diseaseID_train)
del diseaseID_train
Y_train[Y_train==2]=0
Y_test = np.copy(diseaseID_test)
Y_test[Y_test==2]=0

X_test = X_test/255
X_train = X_train/255
best_model_path = 'Autoencoder_covid.hdf5'
model = load_model(best_model_path)
encoder = Model(model.layers[0].input, model.layers[6].output)
encoder_imgs = encoder.predict(X_test)
decoder_imgs = model.predict(X_test)
print('encoder_imgs.shape', encoder_imgs.shape)

# Random Forest:
encoder_RF_train = encoder.predict(X_train)
print('encoder_RF_train.shape', encoder_RF_train.shape)
encoder_RF_train = encoder_RF_train.reshape((encoder_RF_train.shape[0], 23*23*128))
# Regressor:
RF = RandomForestRegressor(n_estimators=100, n_jobs=-1)
RF.fit(encoder_RF_train, Y_train)
encoder_imgs = encoder_imgs.reshape((encoder_imgs.shape[0], 23*23*128))
preds = RF.predict(encoder_imgs)
nocovid = preds[np.where(Y_test == 0)]
covid = preds[np.where(Y_test == 1)]
print("\n\n---------- Predictions ----------\n")
print("preds = ", preds)
np.savetxt('preds_RFr.txt', preds, delimiter=',')
# Counting by thresholds
TP_050 = np.count_nonzero(np.where((Y_test==1) & (preds>0.50)))
FN_050 = np.count_nonzero(np.where((Y_test==1) & (preds<0.50)))
FP_050 = np.count_nonzero(np.where((Y_test==0) & (preds>0.50)))
TN_050 = np.count_nonzero(np.where((Y_test==0) & (preds<0.50)))
cm = np.array([[TP_050, TN_050],[FN_050, FP_050]])
print('{} of {} no COVID-19'.format(cm[0,0], np.sum(Y_test == 0)))
print('{} of {} COVID-19'.format(cm[1,1], np.sum(Y_test == 1)))
cm = normalize(cm, 'l1')
labels = ['no COVID-19', 'COVID-19']
fig1, ax = plt.subplots(figsize=(8, 6))
cax = ax.matshow(cm, cmap='YlGnBu', vmin=0, vmax=1)
cbar = fig1.colorbar(cax)
cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
cbar.set_ticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'])
cbar.ax.set_ylabel('Accuracy', rotation=270, labelpad=20, fontsize=16)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
ax.set_ylabel('Expected', fontsize=16)
ax.set_xlabel('Predicted', fontsize=16)
fig1.savefig('Figures/cm_RFregressor.png', dpi=200)

fig2, ax = plt.subplots(figsize=(8,6))
ax.hist(nocovid, 50, histtype='step', lw=2, color=darkorange, label='No COVID')
ax.hist(covid, 50, histtype='step', lw=2, color=darkorange, label='COVID')
plt.grid()
ax.set_ylabel('Number of events /0.01')
plt.set_xlabel('Score')
plt.legend( loc='upper center')
plt.yscale("log")
fig2.savefig('Figures/histogram_RFregressor.png', dpi=200)

# Classifier:
RFc = RandomForestClassifier(n_estimators=100, n_jobs=-1)
RFc.fit(encoder_RF_train, Y_train)
preds = RFc.predict(encoder_imgs)
cm = confusion_matrix(Y_test, preds)
print("Accuracy:", accuracy_score(Y_test, preds))
print('{} of {} no COVID-19'.format(cm[0,0], np.sum(Y_test == 0)))
print('{} of {} COVID-19'.format(cm[1,1], np.sum(Y_test == 1)))
print(cm)
cm = normalize(cm, 'l1')
labels = ['no COVID-19', 'COVID-19']
fig3, ax = plt.subplots(figsize=(8, 6))
cax = ax.matshow(cm, cmap='YlGnBu', vmin=0, vmax=1)
cbar = fig3.colorbar(cax)
cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
cbar.set_ticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'])
cbar.ax.set_ylabel('Accuracy', rotation=270, labelpad=20, fontsize=16)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
ax.set_ylabel('Expected', fontsize=16)
ax.set_xlabel('Predicted', fontsize=16)
fig3.savefig('Figures/cm_RFclassifier.png', dpi=200)
