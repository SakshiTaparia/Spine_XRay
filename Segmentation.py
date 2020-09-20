#!/usr/bin/env python
# coding: utf-8

# In[1]:


from os import path
import keras
import matplotlib.pyplot as plt
import cv2
import numpy as np 
import pandas as pd


# In[2]:


import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[3]:


skip_ids=[21, 25, 37, 38, 40, 59, 60, 72, 92, 98,101, 118, 120, 128, 130, 182, 206, 210, 226, 245, 271]


# In[5]:


trainDamagedAP_path = '/kaggle/input/spinedataset/Training/Damaged/ID ({})/AP/AP.jpg'
trainDamagedAP=[]
# shape should be divisible by 16 to make downsampling and upsampling consistent in U-Net
for i in range(1,329):
    imagePath=trainDamagedAP_path.format(i)
    if path.exists(imagePath):
        trainDamagedAP.append(cv2.resize(cv2.imread(imagePath,0),(224,224)))
    else:
        if path.exists('/kaggle/input/spinedataset/Training/Damaged/ID ({})/AP/ap.jpg'.format(i)):
            trainDamagedAP.append(cv2.resize(cv2.imread('/kaggle/input/spinedataset/Training/Damaged/ID ({})/AP/ap.jpg'.format(i),0),(224,224)))
        else:
            trainDamagedAP.append(cv2.resize(cv2.imread('/kaggle/input/spinedataset/Training/Damaged/ID ({})/AP.jpg'.format(i),0),(224,224)))
trainDamagedAP=np.array(trainDamagedAP)


# In[6]:


trainNormalAP_path = '/kaggle/input/spinedataset/Training/Normal/ID ({})/AP/AP.jpg'
trainNormalAP=[]
for i in range(1,351):
    imagePath=trainNormalAP_path.format(i)
    if path.exists(imagePath):
        trainNormalAP.append(cv2.resize(cv2.imread(imagePath,0),(224,224)))
    else:
        if path.exists('/kaggle/input/spinedataset/Training/Normal/ID ({})/AP/ap.jpg'.format(i)):
            trainNormalAP.append(cv2.resize(cv2.imread('/kaggle/input/spinedataset/Training/Normal/ID ({})/AP/ap.jpg'.format(i),0),(224,224)))
        else:
            trainNormalAP.append(cv2.resize(cv2.imread('/kaggle/input/spinedataset/Training/Normal/ID ({})/AP.jpg'.format(i),0),(224,224)))
trainNormalAP=np.array(trainNormalAP)


# In[7]:


trainAP=np.concatenate((trainNormalAP,trainDamagedAP))
del trainNormalAP
del trainDamagedAP
trainAP = np.expand_dims(trainAP, axis=3)


# In[8]:


testAP_path = '/kaggle/input/testdataxrays/TestData(XraysOnly)/Test ({})/AP/AP.jpg'
testAP=[]
for i in range(1,302):
    if i in skip_ids:
        continue
    imagePath=testAP_path.format(i)
    if path.exists(imagePath):
        testAP.append(cv2.resize(cv2.imread(imagePath,0),(224,224)))
    else:
        if path.exists('/kaggle/input/testdataxrays/TestData(XraysOnly)/Test ({})/AP/ap.jpg'.format(i)):
            testAP.append(cv2.resize(cv2.imread('/kaggle/input/testdataxrays/TestData(XraysOnly)/Test ({})/AP/ap.jpg'.format(i),0),(224,224)))
        else:
            if(i==287):
                testAP.append(cv2.resize(cv2.imread('/kaggle/input/testdataxrays/TestData(XraysOnly)/Test (287)/AP/SERIES_001_IMG_0000.jpg',0),(224,224)))
            else:
                testAP.append(cv2.resize(cv2.imread('/kaggle/input/testdataxrays/TestData(XraysOnly)/Test ({})/AP.jpg'.format(i),0),(224,224)))
testAP=np.array(testAP)
testAP = np.expand_dims(testAP, axis=3)


# In[9]:


smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


# In[10]:


def jaccard_distance_loss(y_true, y_pred, smooth=1):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    sum_ = (K.sum(y_true_f) + K.sum(y_pred_f))
    jac = (intersection*intersection*intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


# In[11]:


from keras.models import Model, Input, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint


# In[12]:


def dice_score(y_true, y_pred):
    m = y_true.shape[0]
    total=0
    for i in range(m):
        true=y_true[i,:,:].flatten()
        pred=y_pred[i,:,:].flatten()
        intersection = np.sum(np.multiply(true, pred))
        sum_ = np.sum(true)+np.sum(pred)
        total+= (2*intersection+1)/(sum_+1)
    return total/m


# In[67]:


def save_result(y_pred,name, Results):
#     print(y_pred.shape)
    t=0
    for i in range(y_pred.shape[0]):
        if (i+1+t) in skip_ids:
            t=t+1
        if name=='Ap_Pedicle':
            Results[i+1+t]={}
            Results[i+1+t][name]=np.array(y_pred[i,:,:,0],dtype=bool)
        else:
            Results[i+1+t][name]=np.array(y_pred[i,:,:,0],dtype=bool)
    return Results


# In[60]:


History=[]
Dice_Scores=[]
Results={}
def train_result(x_train,y_train,x_test,y_test,name, Results):
    inputs = Input((224, 224, 1))

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])
    model.compile(optimizer=Adam(lr = 1e-5), loss=jaccard_distance_loss, metrics=[dice_coef])
    history = model.fit(x_train,y_train,batch_size=8,validation_data=(x_test,y_test),shuffle=True,epochs=140)
    History.append(history)
    y_pred=model.predict(x_test)
    print(y_pred.shape)
    score=dice_score(y_test,y_pred)
    print("dice score: ",score)
    Dice_Scores.append((name,score))
    for i in range(y_pred.shape[0]):
        y_pred[i,:,:,:][y_pred[i,:,:,:]>0.5] = 1
        y_pred[i,:,:,:][y_pred[i,:,:,:]<=0.5] = 0
    Results = save_result(y_pred,name,Results)
    return Results


# In[20]:


def load_dataAP(name):
    trainNormal_path = '/kaggle/input/spinedataset/Training/Normal/ID ({})/AP/{}.png'
    trainNormal=[]
    for i in range(1,351):
        imagePath=trainNormal_path.format(i,name)
        trainNormal.append(cv2.resize(cv2.imread(imagePath,0),(224,224)))
    trainNormal=np.array(trainNormal)
    trainNormal = np.expand_dims(trainNormal, axis=3)
    
    trainDamaged_path = '/kaggle/input/spinedataset/Training/Damaged/ID ({})/AP/{}.png'
    trainDamaged=[]
    for i in range(1,329):
        imagePath=trainDamaged_path.format(i,name)
        trainDamaged.append(cv2.resize(cv2.imread(imagePath,0),(224,224)))
    trainDamaged=np.array(trainDamaged)
    trainDamaged = np.expand_dims(trainDamaged, axis=3)
    Y_train=np.concatenate((trainNormal,trainDamaged))
    del trainNormal
    del trainDamaged
    Y_test_path = '/kaggle/input/testmasks/TestResizedMasks/Test ({})/AP/{}.png'
    Y_test=[]
    for i in range(1,302):
        if i in skip_ids:
            continue
        imagePath=Y_test_path.format(i,name)
        Y_test.append(cv2.resize(cv2.imread(imagePath,0),(224,224)))
    Y_test=np.array(Y_test)
    Y_test = np.expand_dims(Y_test, axis=3)
    for i in range(Y_train.shape[0]):
        Y_train[i,:,:,:]=Y_train[i,:,:,:]/255
        Y_train[i,:,:,:][Y_train[i,:,:,:]>0.5] = 1
        Y_train[i,:,:,:][Y_train[i,:,:,:]<=0.5] = 0
    for i in range(Y_test.shape[0]):
        Y_test[i,:,:,:]=Y_test[i,:,:,:]/255
        Y_test[i,:,:,:][Y_test[i,:,:,:]>0.5] = 1
        Y_test[i,:,:,:][Y_test[i,:,:,:]<=0.5] = 0
    return Y_train, Y_test


# In[21]:


def load_dataLAT(name):
    trainNormal_path = '/kaggle/input/spinedataset/Training/Normal/ID ({})/LAT/{}.png'
    trainNormal=[]
    for i in range(1,351):
        imagePath=trainNormal_path.format(i,name)
        if path.exists(imagePath):
            trainNormal.append(cv2.resize(cv2.imread(imagePath,0),(224,224)))
        else:
            imagePath='/kaggle/input/spinedataset/Training/Normal/ID ({})/LAT/{}.png'.format(i,'LAT_Spinous_Process')
            trainNormal.append(cv2.resize(cv2.imread(imagePath,0),(224,224)))

    trainNormal=np.array(trainNormal)
    trainNormal = np.expand_dims(trainNormal, axis=3)
    
    trainDamaged_path = '/kaggle/input/spinedataset/Training/Damaged/ID ({})/LAT/{}.png'
    trainDamaged=[]
    for i in range(1,329):
        imagePath=trainDamaged_path.format(i,name)
        if(i==233):
            if(name=='Lat_Posterior_Vertebral_Line'):
                imagePath=trainDamaged_path.format(i-1,name)
                trainDamaged.append(cv2.resize(cv2.imread(imagePath,0),(224,224)))
                continue
        if(i==169):
            if(name=='Lat_Spinous_Process'):
                imagePath=trainDamaged_path.format(i-1,name)
                trainDamaged.append(cv2.resize(cv2.imread(imagePath,0),(224,224)))
                continue
        trainDamaged.append(cv2.resize(cv2.imread(imagePath,0),(224,224)))
    trainDamaged=np.array(trainDamaged)
    trainDamaged = np.expand_dims(trainDamaged, axis=3)
    Y_train=np.concatenate((trainNormal,trainDamaged))
    del trainNormal
    del trainDamaged
    Y_test_path = '/kaggle/input/testmasks/TestResizedMasks/Test ({})/LAT/{}.png'
    Y_test=[]
    for i in range(1,302):
        if i in skip_ids:
            continue
        imagePath=Y_test_path.format(i,name)
        Y_test.append(cv2.resize(cv2.imread(imagePath,0),(224,224)))
    Y_test=np.array(Y_test)
    Y_test = np.expand_dims(Y_test, axis=3)
    for i in range(Y_train.shape[0]):
        Y_train[i,:,:,:]=Y_train[i,:,:,:]/255
        Y_train[i,:,:,:][Y_train[i,:,:,:]>0.5] = 1
        Y_train[i,:,:,:][Y_train[i,:,:,:]<=0.5] = 0
    for i in range(Y_test.shape[0]):
        Y_test[i,:,:,:]=Y_test[i,:,:,:]/255
        Y_test[i,:,:,:][Y_test[i,:,:,:]>0.5] = 1
        Y_test[i,:,:,:][Y_test[i,:,:,:]<=0.5] = 0
    return Y_train, Y_test


# In[68]:


# namesAP=['Ap_Pedicle','Ap_Spinous_Process','Ap_Vertebra']
namesAP=['Ap_Pedicle']

namesLAT=['Lat_Anterior_Vertebral_Line','Lat_Disk_Height','Lat_Posterior_Vertebral_Line','Lat_Spinous_Process','Lat_Vertebra']
namesLAT=['Lat_Anterior_Vertebral_Line','Lat_Disk_Height','Lat_Posterior_Vertebral_Line','Lat_Spinous_Process','Lat_Vertebra']


# In[23]:


for name in namesAP:
    print(name)
    y_train, y_test = load_dataAP(name)
    Results=train_result(trainAP,y_train,testAP,y_test,name, Results)
    del y_train
    del y_test
del trainAP
del testAP


# In[62]:


trainDamagedLAT_path = '/kaggle/input/spinedataset/Training/Damaged/ID ({})/LAT/LAT.jpg'
trainDamagedLAT=[]
# shLATe should be divisible by 16 to make downsampling and upsampling consistent in U-Net
for i in range(1,329):
    imagePath=trainDamagedLAT_path.format(i)
    if path.exists(imagePath):
        trainDamagedLAT.append(cv2.resize(cv2.imread(imagePath,0),(224,224)))
    else:
        if path.exists('/kaggle/input/spinedataset/Training/Damaged/ID ({})/LAT/lat.jpg'.format(i)):
            trainDamagedLAT.append(cv2.resize(cv2.imread('/kaggle/input/spinedataset/Training/Damaged/ID ({})/LAT/lat.jpg'.format(i),0),(224,224)))
        else:
            trainDamagedLAT.append(cv2.resize(cv2.imread('/kaggle/input/spinedataset/Training/Damaged/ID ({})/LAT.jpg'.format(i),0),(224,224)))
trainDamagedLAT=np.array(trainDamagedLAT)

trainNormalLAT_path = '/kaggle/input/spinedataset/Training/Normal/ID ({})/LAT/LAT.jpg'
trainNormalLAT=[]
for i in range(1,351):
    imagePath=trainNormalLAT_path.format(i)
    if path.exists(imagePath):
        trainNormalLAT.append(cv2.resize(cv2.imread(imagePath,0),(224,224)))
    else:
        if path.exists('/kaggle/input/spinedataset/Training/Normal/ID ({})/LAT/lat.jpg'.format(i)):
            trainNormalLAT.append(cv2.resize(cv2.imread('/kaggle/input/spinedataset/Training/Normal/ID ({})/LAT/lat.jpg'.format(i),0),(224,224)))
        elif path.exists('/kaggle/input/spinedataset/Training/Normal/ID ({})/LAT.jpg'.format(i)):
            trainNormalLAT.append(cv2.resize(cv2.imread('/kaggle/input/spinedataset/Training/Normal/ID ({})/LAT.jpg'.format(i),0),(224,224)))
        else:
            trainNormalLAT.append(cv2.resize(cv2.imread('/kaggle/input/spinedataset/Training/Normal/ID ({})/LAT/LAT .jpg'.format(i),0),(224,224)))
trainNormalLAT=np.array(trainNormalLAT)

trainLAT=np.concatenate((trainNormalLAT,trainDamagedLAT))
del trainNormalLAT
del trainDamagedLAT
trainLAT = np.expand_dims(trainLAT, axis=3)


# In[57]:


testLAT_path = '/kaggle/input/testdataxrays/TestData(XraysOnly)/Test ({})/LAT/LAT.jpg'
testLAT=[]
for i in range(1,302):
    if (i) in skip_ids:
        continue
    imagePath=testLAT_path.format(i)
    if path.exists(imagePath):
        testLAT.append(cv2.resize(cv2.imread(imagePath,0),(224,224)))
    else:
        if path.exists('/kaggle/input/testdataxrays/TestData(XraysOnly)/Test ({})/LAT/lat.jpg'.format(i)):
            testLAT.append(cv2.resize(cv2.imread('/kaggle/input/testdataxrays/TestData(XraysOnly)/Test ({})/LAT/lat.jpg'.format(i),0),(224,224)))
        else:
            if(i==287):
                testLAT.append(cv2.resize(cv2.imread('/kaggle/input/testdataxrays/TestData(XraysOnly)/Test (287)/LAT/SERIES_001_IMG_0001.jpg',0),(224,224)))
            else:
                testLAT.append(cv2.resize(cv2.imread('/kaggle/input/testdataxrays/TestData(XraysOnly)/Test ({})/Lat.jpg'.format(i),0),(224,224)))
testLAT=np.array(testLAT)
testLAT = np.expand_dims(testLAT, axis=3)


# In[69]:


for name in namesLAT:
    print(name)
    y_train, y_test = load_dataLAT(name)
    Results=train_result(trainLAT,y_train,testLAT,y_test,name, Results)

    del y_train
    del y_test
del trainLAT
del testLAT


# In[29]:


import pickle
with open('Segmentation.pickle', 'wb') as handle:
    pickle.dump(Results, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[30]:


i=0
for history in History:
    file_name='history_{}'.format(i)
    with open(filename, 'wb') as file_pi:
            pickle.dump(history.history, file_pi)


# In[31]:


print(Dice_Scores)


# 

# In[50]:


history=History[7]
from matplotlib import pyplot as plt
# summarize history for accuracy
plt.plot(history.history['dice_coef'])
plt.plot(history.history['val_dice_coef'])
# plt.title('model accuracy - batch_size:' + str(batch_size))
plt.ylabel('dice_coef')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')


# In[51]:


history=History[7]
from matplotlib import pyplot as plt
# summarize history for accuracy
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
# plt.title('model accuracy - batch_size:' + str(batch_size))
plt.ylabel('jaccard_loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')

