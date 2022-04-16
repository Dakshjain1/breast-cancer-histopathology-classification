#import tkinter 
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

matplotlib.use('TKAgg')
# function for creating an identity residual module
#conda install imutils

#to save images to background
matplotlib.use('Qt5Agg')
dataset_benign='/mnt/new_data/project-data/benign_tumor'
dataset_m_ductal='/mnt/new_data/project-data/ductal_carcinoma'
dataset_m_lobular='/mnt/new_data/project-data/lobular_carcinoma'
dataset_m_mucinous='/mnt/new_data/project-data/mucinous_carcinoma'
dataset_m_papillary='/mnt/new_data/project-data/papillary_carcinoma'
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
        label=imgpath.split('/')[-1]
        temp=label.split(os.path.sep)[-2]
        print(temp)
        #print(label)
        #if label in ['ductal_carcinoma','papillary_carcinoma','lobular_carcinoma','mucinous_carcinoma']:
        #  label='malignant'
        #print(label)
        classes.append(temp)    
    except Exception as e:
        print(e)
print(classes)
#normalization
data=np.array(data,dtype=float) 
data=data/255.0
labels=np.array(classes)
lb=LabelBinarizer()
labels=lb.fit_transform(labels)
print(len(data))
print(classes[0])
print(labels[0])
print(classes[6])
print(labels[6])
print(len(data))
print(len(labels))
xtrain,xtest,ytrain,ytest=train_test_split(data,labels,test_size=0.2,random_state=42)
aug=ImageDataGenerator(rotation_range=0.25,width_shift_range=0.25,height_shift_range=0.1,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')
#model=tinyVGG.build(height=96,width=96,depth=3,classes=len(lb.classes_))
model=Sequential()
input_shape=(96,96,3)
channel_dim=-1

#large patterns in smaller images
model.add(Conv2D(32,(3,3),padding='same',input_shape=input_shape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=channel_dim))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Dropout(0.25))
#increase filter and reduce pool size to find better and finer patterns
model.add(Conv2D(64,(3,3),padding='same'))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=channel_dim))
model.add(Conv2D(64,(3,3),padding='same'))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=channel_dim))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,(3,3),padding='same'))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=channel_dim))
model.add(Conv2D(128,(3,3),padding='same'))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=channel_dim))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

#NEWWWWWWWWNWWWWW
model.add(Conv2D(128,(3,3),padding='same'))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=channel_dim))
model.add(Conv2D(128,(3,3),padding='same'))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=channel_dim))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,(3,3),padding='same'))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=channel_dim))
model.add(Conv2D(128,(3,3),padding='same'))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=channel_dim))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(256,(3,3),padding='same'))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=channel_dim))
model.add(Conv2D(256,(3,3),padding='same'))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=channel_dim))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))



######NEWWWWW


model.add(Flatten())#converts into a single dimensional array

model.add(Dense(1024))
model.add(Activation('relu'))

model.add(BatchNormalization()) 
 
model.add(Dropout(0.5))

model.add(Dense(5))
model.add(Activation("softmax"))

print(model.summary())
#aug=ImageDataGenerator(rotation_range=0.25,width_shift_range=0.25,height_shift_range=0.1,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')
opt=RMSprop(lr=HP_LR)
model.compile(loss="binary_crossentropy",optimizer=opt,metrics=['accuracy'])
history=model.fit_generator(aug.flow(xtrain,ytrain,batch_size=HP_BS),validation_data=(xtest,ytest),steps_per_epoch=len(xtrain)//HP_BS,epochs=HP_EPOCHS)
model.save('bcdfinal.h5')


#old code
#loss = history.history['loss']
#accuracy = history.history['accuracy']
gt=[]
pred=[]
predictions=model.predict(xtest)

for i in range(len(xtest)):
    pred.append(np.argmax(predictions[i]))
    
print(pred)
for i in range(len(xtest)):
    for j in range(5):
        if ytest[i][j]==1:
            gt.append(j)
            
            
            
for i in range(20):
    print(np.argmax(predictions[i])+1)
    print(ytest[i])
print(gt)
print(confusion_matrix(pred,gt))

count=0
for i in range(len(xtest)):
    if pred[i]!=gt[i]:
        count+=1
print(count)
print("testing accuracy=",1-((0.25*count)/len(xtest)))
x=[]
y=[]
for i in range(len(predictions)):
    for j in range(5):
        if ytrain[i][j]==1:
             x.append(j)
for i in range(len(predictions)):
    y.append(np.argmax(predictions[i])+1)
    
#plt.plot(y,x)
#plt.show()
