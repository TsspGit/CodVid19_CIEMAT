__author__ = '@Tssp'
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd

# Functions:
############
def resize_img(img, newdims):
    # img: image
    # newdims: tuple (width pixels, height pixels)
    out = cv2.resize(img, newdims)
    return out

if __name__ == "__main__":
    # Resize train images:
    ######################
    train_class = pd.read_csv('../train_split_v2.txt', sep=' ', header=None, names=['patientID', 'image_path', 'class'])
    i = 0
    for img_path in train_class['image_path'].values:
        img = (cv2.imread('../train/' + img_path))
        img_resized = resize_img(img, (1, 1))
        plt.imsave('train/' + img_path, img_resized)
        i +=1
        if i%100 == 0:
            print(f"Progress: \n\n {i} of {len(train_class.values)}")

    # Resize test images:
    #####################
    test_class = pd.read_csv('../test_split_v2.txt', sep=' ', header=None, names=['patientID', 'image_path', 'class'])
    i = 0
    for img_path in test_class['image_path'].values:
        img = (cv2.imread('../test/' + img_path))
        img_resized = resize_img(img, (1, 1))
        plt.imsave('test/' + img_path, img_resized)
        i +=1
        if i%100 == 0:
            print(f"Progress: \n\n {i} of {len(test_class.values)}")

    