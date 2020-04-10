__author__ = '@Tssp'

''' RNN Bidireccional en Keras '''

import numpy as np
import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
from keras.layers import Conv2D, Flatten, Dense
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from codvidutils.imageproc import map_categorical

# Load the dataset that includes the classes:
#-------------------------------------------#
train_class = pd.read_csv('data/train_split_v2.txt', sep=' ', header=None, names=['patientID', 'image_path', 'class'])
test_class = pd.read_csv('data/test_split_v2.txt', sep=' ', header=None, names=['patientID', 'image_path', 'class'])
# Create an integer categorical column in the datasets:
values_dict = {'pneumonia': 0, 'COVID-19': 1, 'normal': 2}
test_class['class_categorical'] = test_class['class'].apply(map_categorical, args=(values_dict,))
train_class['class_categorical'] = train_class['class'].apply(map_categorical, args=(values_dict,))

# Load the images:
#----------------#
train_pics = []
for img in train_class['image_path'].values:
    train_pics.append(np.array(Image.open('data/train/' + img))[:, :, :3])
test_pics = []
for img in test_class['image_path'].values:
    test_pics.append(np.array(Image.open('data/test/' + img))[:, :, :3])

# Prepare the data for the NN:
#----------------------------#
X_train = np.array(train_pics)/255
Y_train = to_categorical(train_class['class_categorical'].values.reshape(-1, 1))
X_test = np.array(test_pics)/255
Y_test = to_categorical(test_class['class_categorical'].values.reshape(-1, 1))
assert X_train.shape[0] == len(Y_train)
assert X_test.shape[0] == len(Y_test)
print('Train shape: {}\nTest shape: {}'.format(X_train.shape, X_test.shape))

# Neural Network:
#---------------#
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=X_train.shape[1:], data_format = 'channels_last'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, Y_train, epochs=30, batch_size=20, verbose=0, validation_data=(X_test, Y_test))
pred = model.predict(X_test)
acc_train = np.average(history.history["acc"])
acc_test = np.average(history.history["val_acc"])
print(model.summary())

# Plots:
#------#
# Loss curves:
fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(8,6))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
print('max accuracy: ', round(np.max(history.history['accuracy']), 2))

# Confusion matrix
cm = confusion_matrix(Y_test.argmax(axis=1), pred.argmax(axis=1))
labels = ['pneumonia', 'CODVID-19', 'normal']
fig2, ax2 = plt.subplots(figsize=(8, 6))
cax = ax2.matshow(cm, cmap=plt.cm.Greens)
cbar = fig2.colorbar(cax)
cbar.set_ticks([])
cbar.ax.set_ylabel('Accuracy', rotation=270, labelpad=20, fontsize=16)
ax2.set_xticklabels([''] + labels)
ax2.set_yticklabels([''] + labels)

plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
