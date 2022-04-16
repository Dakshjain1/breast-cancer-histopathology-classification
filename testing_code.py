from keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.layers.core import Activation,Flatten,Dense,Dropout
from keras import backend as k
import matplotlib.pyplot as plt
import matplotlib
from keras.preprocessing.image import ImageDataGenerator,img_to_array
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.optimizers import Adam,Nadam,RMSprop,Adamax,SGD
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import os
import cv2
import pickle
import random
from sklearn.metrics import confusion_matrix

## matplotlib.use('TKAgg')
# function for creating an identity residual module
#conda install imutils

#to save images to background
# matplotlib.use('Qt5Agg')
dataset_benign='/mnt/new_data/project-data/benign_tumor/'
dataset_m_ductal='/mnt/new_data/project-data/ductal_carcinoma/'
dataset_m_lobular='/mnt/new_data/project-data/lobular_carcinoma/'
dataset_m_mucinous='/mnt/new_data/project-data/mucinous_carcinoma/'
dataset_m_papillary='/mnt/new_data/project-data/papillary_carcinoma/'
model_path="model.h5"

label_path="/"
plot_path="/"
HP_LR=1e-3
HP_EPOCHS=2
HP_BS=55
HP_IMAGE_DIM=(96,96,3)
data=[]
classes=[]
imagepaths_benign=sorted(list(paths.list_images(dataset_benign)))
imagepaths_ductal=sorted(list(paths.list_images(dataset_m_ductal)))
imagepaths_lobular=sorted(list(paths.list_images(dataset_m_lobular)))
imagepaths_mucinous=sorted(list(paths.list_images(dataset_m_mucinous)))
imagepaths_papillary=sorted(list(paths.list_images(dataset_m_papillary)))
print(len(imagepaths_benign))
print(len(imagepaths_ductal))
print(len(imagepaths_lobular))
print(len(imagepaths_papillary))
print(len(imagepaths_mucinous))
random.seed(42)
imagepaths=imagepaths_ductal+imagepaths_lobular+imagepaths_mucinous+imagepaths_papillary+imagepaths_benign
print(len(imagepaths))
random.shuffle(imagepaths)
print(len(imagepaths))
classes=[]
for imgpath in imagepaths:
    try:
        image=cv2.imread(imgpath)
        image=cv2.resize(image,(96,96))
        image_array=img_to_array(image)
        data.append(image_array)
        print(imgpath)
        label=imgpath.split('/')[-2]
#         temp=label.split(os.path.sep)[-2]
#         print(temp)
        print(label)
        #if label in ['ductal_carcinoma','papillary_carcinoma','lobular_carcinoma','mucinous_carcinoma']:
        #  label='malignant'
        #print(label)
        classes.append(label)    
    except Exception as e:
        print(e)
print(classes)
