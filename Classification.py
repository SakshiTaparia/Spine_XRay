#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os.path
from os import path
import os
import glob
import h5py
import shutil
import imgaug as aug
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mimg
import imgaug.augmenters as iaa
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser
from PIL import Image
from pathlib import Path
from skimage.io import imread
from skimage.transform import resize
from keras.models import Sequential, Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, SeparableConv2D
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import cv2
from keras import backend as K
color = sns.color_palette()
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


skip_ids=[21, 25, 37, 38, 40, 59, 60, 72, 92, 98,101, 118, 120, 128, 130, 182, 206, 210, 226, 245, 271]


# In[ ]:


testAP_path = '/kaggle/input/testdataxrays/TestData(XraysOnly)/Test ({})/AP/AP.jpg'
testAP=[]
for i in range(1,302):
    if i in skip_ids:
        continue
    imagePath=testAP_path.format(i)
    if path.exists(imagePath):
        testAP.append(cv2.resize(cv2.imread(imagePath,0),(256,512)))
    else:
        if path.exists('/kaggle/input/testdataxrays/TestData(XraysOnly)/Test ({})/AP/ap.jpg'.format(i)):
            testAP.append(cv2.resize(cv2.imread('/kaggle/input/testdataxrays/TestData(XraysOnly)/Test ({})/AP/ap.jpg'.format(i),0),(256,512)))
        else:
            if(i==287):
                testAP.append(cv2.resize(cv2.imread('/kaggle/input/testdataxrays/TestData(XraysOnly)/Test (287)/AP/SERIES_001_IMG_0000.jpg',0),(256,512)))
            else:
                testAP.append(cv2.resize(cv2.imread('/kaggle/input/testdataxrays/TestData(XraysOnly)/Test ({})/AP.jpg'.format(i),0),(224,224)))
testAP=np.array(testAP)
testAP = np.expand_dims(testAP, axis=3)


# In[ ]:


testAP=np.stack((testAP,testAP,testAP),axis=3)[:,:,:,:,0]


# In[ ]:


testLAT_path = '/kaggle/input/testdataxrays/TestData(XraysOnly)/Test ({})/LAT/LAT.jpg'
testLAT=[]
for i in range(1,302):
    if i in skip_ids:
        continue
    imagePath=testAP_path.format(i)
    if path.exists(imagePath):
        testLAT.append(cv2.resize(cv2.imread(imagePath,0),(256,512)))
    else:
        if path.exists('/kaggle/input/testdataxrays/TestData(XraysOnly)/Test ({})/LAT/lat.jpg'.format(i)):
            testLAT.append(cv2.resize(cv2.imread('/kaggle/input/testdataxrays/TestData(XraysOnly)/Test ({})/LAT/lat.jpg'.format(i),0),(256,512)))
        else:
            if(i==287):
                testLAT.append(cv2.resize(cv2.imread('/kaggle/input/testdataxrays/TestData(XraysOnly)/Test (287)/LAT/SERIES_001_IMG_0001.jpg',0),(256,512)))
            else:
                testLAT.append(cv2.resize(cv2.imread('/kaggle/input/testdataxrays/TestData(XraysOnly)/Test ({})/Lat.jpg'.format(i),0),(256,512)))
testLAT=np.array(testLAT)
testLAT = np.expand_dims(testLAT, axis=3)


# In[ ]:


testLAT=np.stack((testLAT,testLAT,testLAT),axis=3)[:,:,:,:,0]


# In[ ]:


trainDamagedAP_path = '/kaggle/input/spinedataset/Training/Damaged/ID ({})/AP/AP.jpg'
trainDamagedAP=[]
# shape should be divisible by 16 to make downsampling and upsampling consistent in U-Net
for i in range(1,329):
    imagePath=trainDamagedAP_path.format(i)
    if path.exists(imagePath):
        trainDamagedAP.append(cv2.resize(cv2.imread(imagePath,0),(256,512)))
    else:
        if path.exists('/kaggle/input/spinedataset/Training/Damaged/ID ({})/AP/ap.jpg'.format(i)):
            trainDamagedAP.append(cv2.resize(cv2.imread('/kaggle/input/spinedataset/Training/Damaged/ID ({})/AP/ap.jpg'.format(i),0),(256,512)))
        else:
            trainDamagedAP.append(cv2.resize(cv2.imread('/kaggle/input/spinedataset/Training/Damaged/ID ({})/AP.jpg'.format(i),0),(256,512)))
trainDamagedAP=np.array(trainDamagedAP)

trainNormalAP_path = '/kaggle/input/spinedataset/Training/Normal/ID ({})/AP/AP.jpg'
trainNormalAP=[]
for i in range(1,351):
    imagePath=trainNormalAP_path.format(i)
    if path.exists(imagePath):
        trainNormalAP.append(cv2.resize(cv2.imread(imagePath,0),(256,512)))
    else:
        if path.exists('/kaggle/input/spinedataset/Training/Normal/ID ({})/AP/ap.jpg'.format(i)):
            trainNormalAP.append(cv2.resize(cv2.imread('/kaggle/input/spinedataset/Training/Normal/ID ({})/AP/ap.jpg'.format(i),0),(256,512)))
        else:
            trainNormalAP.append(cv2.resize(cv2.imread('/kaggle/input/spinedataset/Training/Normal/ID ({})/AP.jpg'.format(i),0),(256,512)))
trainNormalAP=np.array(trainNormalAP)

trainAP=np.concatenate((trainDamagedAP,trainNormalAP), axis=0)
del trainDamagedAP
del trainNormalAP
trainAP=np.expand_dims(trainAP, axis=3)

trainDamagedLAT_path = '/kaggle/input/spinedataset/Training/Damaged/ID ({})/LAT/LAT.jpg'
trainDamagedLAT=[]
# shLATe should be divisible by 16 to make downsampling and upsampling consistent in U-Net
for i in range(1,329):
    imagePath=trainDamagedLAT_path.format(i)
    if path.exists(imagePath):
        trainDamagedLAT.append(cv2.resize(cv2.imread(imagePath,0),(256,512)))
    else:
        if path.exists('/kaggle/input/spinedataset/Training/Damaged/ID ({})/LAT/lat.jpg'.format(i)):
            trainDamagedLAT.append(cv2.resize(cv2.imread('/kaggle/input/spinedataset/Training/Damaged/ID ({})/LAT/lat.jpg'.format(i),0),(256,512)))
        else:
            trainDamagedLAT.append(cv2.resize(cv2.imread('/kaggle/input/spinedataset/Training/Damaged/ID ({})/LAT.jpg'.format(i),0),(256,512)))
trainDamagedLAT=np.array(trainDamagedLAT)

trainNormalLAT_path = '/kaggle/input/spinedataset/Training/Normal/ID ({})/LAT/LAT.jpg'
trainNormalLAT=[]
for i in range(1,351):
    imagePath=trainNormalLAT_path.format(i)
    if path.exists(imagePath):
        trainNormalLAT.append(cv2.resize(cv2.imread(imagePath,0),(256,512)))
    else:
        if path.exists('/kaggle/input/spinedataset/Training/Normal/ID ({})/LAT/lat.jpg'.format(i)):
            trainNormalLAT.append(cv2.resize(cv2.imread('/kaggle/input/spinedataset/Training/Normal/ID ({})/LAT/lat.jpg'.format(i),0),(256,512)))
        elif path.exists('/kaggle/input/spinedataset/Training/Normal/ID ({})/LAT.jpg'.format(i)):
            trainNormalLAT.append(cv2.resize(cv2.imread('/kaggle/input/spinedataset/Training/Normal/ID ({})/LAT.jpg'.format(i),0),(256,512)))
        else:
            trainNormalLAT.append(cv2.resize(cv2.imread('/kaggle/input/spinedataset/Training/Normal/ID ({})/LAT/LAT .jpg'.format(i),0),(256,512)))
trainNormalLAT=np.array(trainNormalLAT)

trainLAT=np.concatenate((trainDamagedLAT,trainNormalLAT), axis=0)
del trainDamagedLAT
del trainNormalLAT
trainLAT=np.expand_dims(trainLAT, axis=3)


# In[ ]:


x_trainLAT=np.stack((trainLAT,trainLAT,trainLAT),axis=3)[:,:,:,:,0]
x_trainAP=np.stack((trainAP,trainAP,trainAP),axis=3)[:,:,:,:,0]
del trainAP
del trainLAT


# In[ ]:


y_train=np.concatenate((np.ones((328,1)),np.zeros((350,1))),axis=0)


# In[ ]:


#shuffle

indices = np.arange(y_train.shape[0])
np.random.shuffle(indices)

y_train = y_train[indices]
x_trainLAT = x_trainLAT[indices,:,:,:]
x_trainAP = x_trainAP[indices,:,:,:]


# In[ ]:


# def build_model():
#     input_img = Input(shape=(512,256,3), name='ImageInput')
#     x = Conv2D(64, (3,3), activation='relu', padding='same', name='Conv1_1')(input_img)
#     x = Conv2D(64, (3,3), activation='relu', padding='same', name='Conv1_2')(x)
#     x = MaxPooling2D((2,2), name='pool1')(x)
    
#     x = SeparableConv2D(128, (3,3), activation='relu', padding='same', name='Conv2_1')(x)
#     x = SeparableConv2D(128, (3,3), activation='relu', padding='same', name='Conv2_2')(x)
#     x = MaxPooling2D((2,2), name='pool2')(x)
    
#     x = SeparableConv2D(256, (3,3), activation='relu', padding='same', name='Conv3_1')(x)
#     x = BatchNormalization(name='bn1')(x)
#     x = SeparableConv2D(256, (3,3), activation='relu', padding='same', name='Conv3_2')(x)
#     x = BatchNormalization(name='bn2')(x)
#     x = SeparableConv2D(256, (3,3), activation='relu', padding='same', name='Conv3_3')(x)
#     x = MaxPooling2D((2,2), name='pool3')(x)
    
#     x = SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_1')(x)
#     x = BatchNormalization(name='bn3')(x)
#     x = SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_2')(x)
#     x = BatchNormalization(name='bn4')(x)
#     x = SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_3')(x)
#     x = MaxPooling2D((2,2), name='pool4')(x)
    
#     x = Flatten(name='flatten')(x)
#     x = Dense(1024, activation='relu', name='fc1')(x)
#     x = Dropout(0.7, name='dropout1')(x)
#     x = Dense(512, activation='relu', name='fc2')(x)
#     x = Dropout(0.5, name='dropout2')(x)
#     x = Dense(1, activation='relu', name='fc3')(x)
    
#     model = Model(inputs=input_img, outputs=x)
#     return model


# In[ ]:


# model =  build_model()
# model.summary()


# In[ ]:


# # Open the VGG16 weight file
# f = h5py.File('/kaggle/input/vgg16-weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', 'r')

# # Select the layers for which you want to set weight.

# w,b = f['block1_conv1']['block1_conv1_W_1:0'], f['block1_conv1']['block1_conv1_b_1:0']
# model.layers[1].set_weights = [w,b]

# w,b = f['block1_conv2']['block1_conv2_W_1:0'], f['block1_conv2']['block1_conv2_b_1:0']
# model.layers[2].set_weights = [w,b]

# w,b = f['block2_conv1']['block2_conv1_W_1:0'], f['block2_conv1']['block2_conv1_b_1:0']
# model.layers[4].set_weights = [w,b]

# w,b = f['block2_conv2']['block2_conv2_W_1:0'], f['block2_conv2']['block2_conv2_b_1:0']
# model.layers[5].set_weights = [w,b]

# f.close()
# model.summary()


# In[ ]:


# # opt = RMSprop(lr=0.0001, decay=1e-6)
# opt = Adam(lr=0.0001, decay=1e-5)
# es = EarlyStopping(patience=5)
# chkpt = ModelCheckpoint(filepath='best_model_todate', save_best_only=True, save_weights_only=True)
# model.compile(loss='binary_crossentropy', metrics=['accuracy'],optimizer=opt)


# In[ ]:


# datagen = ImageDataGenerator(
#     featurewise_center=True,
#     featurewise_std_normalization=True,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True)
# datagen.fit(x_train)


# In[ ]:


# batch_size = 8
# nb_epochs = 20

# # Get a train data generator
# # train_data_gen = data_gen(data=train_data, batch_size=batch_size)

# # Define the number of training steps
# nb_train_steps = x_train.shape[0]//batch_size


# # feature extraction

# In[ ]:


import keras
import sys
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D
from keras import regularizers, optimizers
import numpy as np
import pandas as pd
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model


# In[ ]:


from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.imagenet_utils import preprocess_input
import numpy as np

model = ResNet50(weights='imagenet', pooling=max, include_top = False) 
model.summary()


# In[ ]:


# # input_img = Input(shape=(512,256,3), name='ImageInput')
# modelLAT = Sequential()
# modelLAT.add(Dense(20, input_dim=262144, activation='relu'))
# modelLAT.add(Dense(12, activation='relu'))
# modelLAT.add(Dense(1, activation='sigmoid'))

# # Prepare model model saving directory.
# save_dir = os.path.join(os.getcwd(), 'saved_models')
# model_name = 'LAT_%s_model.{epoch:03d}.h5'
# if not os.path.isdir(save_dir):
#     os.makedirs(save_dir)
# filepath = os.path.join(save_dir, model_name)

# # Prepare callbacks for model saving and for learning rate adjustment.
# checkpoint = ModelCheckpoint(filepath=filepath,
#                              monitor='val_acc',
#                              verbose=1,
#                              save_best_only=True)
# modelLAT.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


# history = modelLAT.fit(resnet50_feature_listLAT[:600,:], y_train[:600], batch_size=1, epochs=120, shuffle=True,
#                               validation_data=(resnet50_feature_listLAT[600:,:], y_train[600:]),callbacks=[checkpoint])


# In[ ]:


# resnet50_feature_listAP = []
# for i in range(x_trainAP.shape[0]):
#     img_data = preprocess_input(x_trainAP[i,:,:,:])
#     img_data = np.expand_dims(img_data, axis=0)
#     resnet50_feature = model.predict(img_data)
#     resnet50_feature_np = np.array(resnet50_feature)
# #     resnet50_feature_list.append(resnet50_feature_np)
#     resnet50_feature_listAP.append(resnet50_feature_np.flatten())
    
# resnet50_feature_listAP = np.array(resnet50_feature_list)


# In[ ]:


# # input_img = Input(shape=(512,256,3), name='ImageInput')
# modelAP = Sequential()
# modelAP.add(Dense(20, input_dim=262144, activation='relu'))
# modelAP.add(Dense(12, activation='relu'))
# modelAP.add(Dense(1, activation='sigmoid'))

# # Prepare model model saving directory.
# save_dir = os.path.join(os.getcwd(), 'saved_models')
# model_name = 'AP_%s_model.{epoch:03d}.h5'
# if not os.path.isdir(save_dir):
#     os.makedirs(save_dir)
# filepath = os.path.join(save_dir, model_name)

# # Prepare callbacks for model saving and for learning rate adjustment.
# checkpoint = ModelCheckpoint(filepath=filepath,
#                              monitor='val_acc',
#                              verbose=1,
#                              save_best_only=True)
# modelAP.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


# history = modelAP.fit(resnet50_feature_listAP[:600,:], y_train[:600], batch_size=1, epochs=120, shuffle=True,
#                               validation_data=(resnet50_feature_listAP[600:,:], y_train[600:]),callbacks=[checkpoint])


# In[ ]:


resnet50_feature_list = []
for i in range(x_trainAP.shape[0]):
    img_dataAP = preprocess_input(x_trainAP[i,:,:,:])
    img_dataLAT = preprocess_input(x_trainLAT[i,:,:,:])
    img_dataAP = np.expand_dims(img_dataAP, axis=0)
    img_dataLAT = np.expand_dims(img_dataLAT, axis=0)
    resnet50_featureAP = np.array(model.predict(img_dataAP)).flatten()
    resnet50_featureLAT = np.array(model.predict(img_dataLAT)).flatten()
    resnet50_feature_np = np.concatenate((resnet50_featureAP,resnet50_featureLAT),axis=0)
    
#     resnet50_feature_list.append(resnet50_feature_np)
    resnet50_feature_list.append(resnet50_feature_np)
    
resnet50_feature_list = np.array(resnet50_feature_list)


# In[ ]:


resnet50_feature_list = []
for i in range(x_trainAP.shape[0]):
    img_dataAP = preprocess_input(x_trainAP[i,:,:,:])
    img_dataLAT = preprocess_input(x_trainLAT[i,:,:,:])
    img_dataAP = np.expand_dims(img_dataAP, axis=0)
    img_dataLAT = np.expand_dims(img_dataLAT, axis=0)
    resnet50_featureAP = np.array(model.predict(img_dataAP))
    
#     resnet50_feature_list.append(resnet50_feature_np)
    resnet50_feature_list.append(resnet50_featureAP)
    
resnet50_feature_list = np.array(resnet50_feature_list)


# In[ ]:


resnet50_feature_list[0,0,:,:,0].shape


# In[ ]:


plt.imshow((resnet50_feature_list[0,0,:,:,7]))


# In[ ]:


# input_img = Input(shape=(512,256,3), name='ImageInput')
model = Sequential()
model.add(Dense(20, input_dim=524288, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

def lr_schedule(epoch):
    """Learning Rate Schedule
    """
    lr = 1e-3
    if epoch > 100:
        lr *= 0.5e-3
    elif epoch > 50:
        lr *= 1e-3
    elif epoch > 40:
        lr *= 1e-2
    elif epoch > 20:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)


# Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'AP_%s_model.{epoch:03d}.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)
callbacks = [checkpoint, lr_reducer, lr_scheduler]
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


history = model.fit(resnet50_feature_list[:600,:], y_train[:600], batch_size=1, epochs=80,validation_data=(resnet50_feature_list[600:,:], y_train[600:]), shuffle=True,callbacks=callbacks)


# In[ ]:


y_pred=model.predict_classes(resnet50_feature_list)


# In[ ]:


import sklearn 
sklearn.metrics.confusion_matrix(y_train, y_pred, labels=None, sample_weight=None, normalize=None)


# In[ ]:


model.summary()


# In[ ]:


resnet = ResNet50(weights='imagenet', pooling=max, include_top = False) 
model.summary()


# In[ ]:


resnet50_feature_list = []
for i in range(testAP.shape[0]):
    img_dataAP = preprocess_input(testAP[i,:,:,:])
    img_dataLAT = preprocess_input(testLAT[i,:,:,:])
    img_dataAP = np.expand_dims(img_dataAP, axis=0)
    img_dataLAT = np.expand_dims(img_dataLAT, axis=0)
    resnet50_featureAP = np.array(resnet.predict(img_dataAP)).flatten()
    resnet50_featureLAT = np.array(resnet.predict(img_dataLAT)).flatten()
    resnet50_feature_np = np.concatenate((resnet50_featureAP,resnet50_featureLAT),axis=0)
    
#     resnet50_feature_list.append(resnet50_feature_np)
    resnet50_feature_list.append(resnet50_feature_np)
    
resnet50_feature_list = np.array(resnet50_feature_list)


# In[ ]:


y_pred=model.predict_classes(resnet50_feature_list)


# In[ ]:


results={}
t=0
for i in range(y_pred.shape[0]):
    if i+1 in skip_ids:
            t=t+1
    results[i+1+t]=y_pred[i]


# In[ ]:


import pickle
with open('Classification.pickle', 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[ ]:


with open('history.pickle', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)


# In[ ]:


## Plot
from matplotlib import pyplot as plt
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
# summarize history for loss


# In[ ]:


# baseMapNum=32
# weight_decay = 1e-4
# model = Sequential()
# model.add(Conv2D(baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=vgg16_feature_list_np.shape[1:]))
# model.add(Activation('relu'))
# model.add(BatchNormalization())
# model.add(Conv2D(baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
# model.add(Activation('relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.2))

# model.add(Conv2D(2*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
# model.add(Activation('relu'))
# model.add(BatchNormalization())
# model.add(Conv2D(2*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
# model.add(Activation('relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.3))

# # model.add(Conv2D(4*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
# # model.add(Activation('relu'))
# # model.add(BatchNormalization())
# # model.add(Conv2D(4*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
# # model.add(Activation('relu'))
# # model.add(BatchNormalization())
# # model.add(MaxPooling2D(pool_size=(2,2)))
# # model.add(Dropout(0.4))

# model.add(Flatten())
# model.add(Dense(1, activation='softmax'))
# opt_rms = keras.optimizers.rmsprop(lr=0.0005,decay=1e-6)
# model.compile(loss='binary_crossentropy',
#         optimizer=opt_rms,
#         metrics=['accuracy'])
# model.summary()


# In[ ]:


# datagen = ImageDataGenerator(
#     featurewise_center=True,
#     featurewise_std_normalization=True,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True)
# datagen.fit(x_train)


# In[ ]:


# from keras.applications.inception_v3 import InceptionV3
# from keras.preprocessing import image
# from keras.models import Model
# from keras.layers import Dense, GlobalAveragePooling2D
# from keras import backend as K


# In[ ]:


# input_tensor = Input(shape=(512, 256, 3))
# base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)

# # add a global spatial average pooling layer
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# # let's add a fully-connected layer
# x = Dense(1024, activation='relu')(x)
# # and a logistic layer -- let's say we have 1 class
# predictions = Dense(1, activation='sigmoid')(x)

# # this is the model we will train
# model = Model(inputs=base_model.input, outputs=predictions)

# # first: train only the top layers (which were randomly initialized)
# # i.e. freeze all convolutional InceptionV3 layers
# for layer in base_model.layers:
#     layer.trainable = False

# # compile the model (should be done *after* setting layers to non-trainable)
# model.compile(optimizer='rmsprop', loss='binary_crossentropy',metrics=['binary_accuracy'])


# In[ ]:


# batch_size = 32
# epochs = 100
# history = model.fit_generator(datagen.flow(x_train[:610,:,:,:], y_train[:610], batch_size=batch_size),
#                     validation_data=(x_train[610:,:,:,:], y_train[610:]),
#                     epochs=epochs, verbose=1, workers=4)


# In[ ]:


# ## Plot
# from matplotlib import pyplot as plt
# # summarize history for accuracy
# plt.plot(history.history['binary_accuracy'])
# plt.plot(history.history['val_binary_accuracy'])
# plt.title('model accuracy - batch_size:' + str(batch_size))
# plt.ylabel('binary_accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# # summarize history for loss


# In[ ]:


# np.min(model.predict(x_train))


# In[ ]:


# from sklearn.metrics import accuracy_score
# y_pred = model.predict(x_train[0:610,:,:,:])
# np.min(y_pred)


# In[ ]:





# In[ ]:


# model.fit(vgg16_feature_list_np[:600,:,:,:], y_train[:600],
#                   batch_size=8,
#                   epochs=20,
#                   validation_data=(vgg16_feature_list_np[600:,:,:,:], y_train[600:]),
#                   shuffle=True)


# In[ ]:




# # Training parameters
# batch_size = 1  # orig paper trained all networks with batch_size=128
# epochs = 10
# data_augmentation = False
# num_classes = 2
# input_shape=(500,250,6)
# # Subtracting pixel mean improves accuracy
# subtract_pixel_mean = True

# n = 5

# # Model version
# # Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
# version = 2

# # Computed depth from supplied model parameter n
# if version == 1:
#     depth = n * 6 + 2
# elif version == 2:
#     depth = n * 9 + 2

# # Model name, depth and version
# model_type = 'ResNet%dv%d' % (depth, version)

# # Input image dimensions.
# input_shape = x_train.shape[1:]


# print('x_train shape:', x_train.shape)
# print(x_train.shape[0], 'train samples')
# # print(x_test.shape[0], 'test samples')
# print('y_train shape:', y_train.shape)


# def lr_schedule(epoch):
#     """Learning Rate Schedule

#     Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
#     Called automatically every epoch as part of callbacks during training.

#     # Arguments
#         epoch (int): The number of epochs

#     # Returns
#         lr (float32): learning rate
#     """
#     lr = 1e-3
#     if epoch > 180:
#         lr *= 0.5e-3
#     elif epoch > 160:
#         lr *= 1e-3
#     elif epoch > 120:
#         lr *= 1e-2
#     elif epoch > 80:
#         lr *= 1e-1
#     print('Learning rate: ', lr)
#     return lr


# def resnet_layer(inputs,
#                  num_filters=16,
#                  kernel_size=3,
#                  strides=1,
#                  activation='relu',
#                  batch_normalization=True,
#                  conv_first=True):
#     """2D Convolution-Batch Normalization-Activation stack builder

#     # Arguments
#         inputs (tensor): input tensor from input image or previous layer
#         num_filters (int): Conv2D number of filters
#         kernel_size (int): Conv2D square kernel dimensions
#         strides (int): Conv2D square stride dimensions
#         activation (string): activation name
#         batch_normalization (bool): whether to include batch normalization
#         conv_first (bool): conv-bn-activation (True) or
#             bn-activation-conv (False)

#     # Returns
#         x (tensor): tensor as input to the next layer
#     """
#     conv = Conv2D(num_filters,
#                   kernel_size=kernel_size,
#                   strides=strides,
#                   padding='same',
#                   kernel_initializer='he_normal',
#                   kernel_regularizer=l2(1e-4))

#     x = inputs
#     if conv_first:
#         x = conv(x)
#         if batch_normalization:
#             x = BatchNormalization()(x)
#         if activation is not None:
#             x = Activation(activation)(x)
#     else:
#         if batch_normalization:
#             x = BatchNormalization()(x)
#         if activation is not None:
#             x = Activation(activation)(x)
#         x = conv(x)
#     return x


# def resnet_v1(input_shape, depth, num_classes=100):
#     """ResNet Version 1 Model builder [a]

#     Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
#     Last ReLU is after the shortcut connection.
#     At the beginning of each stage, the feature map size is halved (downsampled)
#     by a convolutional layer with strides=2, while the number of filters is
#     doubled. Within each stage, the layers have the same number filters and the
#     same number of filters.
#     Features maps sizes:
#     stage 0: 32x32, 16
#     stage 1: 16x16, 32
#     stage 2:  8x8,  64
#     The Number of parameters is approx the same as Table 6 of [a]:
#     ResNet20 0.27M
#     ResNet32 0.46M
#     ResNet44 0.66M
#     ResNet56 0.85M
#     ResNet110 1.7M

#     # Arguments
#         input_shape (tensor): shape of input image tensor
#         depth (int): number of core convolutional layers
#         num_classes (int): number of classes (CIFAR10 has 10)

#     # Returns
#         model (Model): Keras model instance
#     """
#     if (depth - 2) % 6 != 0:
#         raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
#     # Start model definition.
#     num_filters = 16
#     num_res_blocks = int((depth - 2) / 6)

#     inputs = Input(shape=input_shape)
#     x = resnet_layer(inputs=inputs)
#     # Instantiate the stack of residual units
#     for stack in range(3):
#         for res_block in range(num_res_blocks):
#             strides = 1
#             if stack > 0 and res_block == 0:  # first layer but not first stack
#                 strides = 2  # downsample
#             y = resnet_layer(inputs=x,
#                              num_filters=num_filters,
#                              strides=strides)
#             y = resnet_layer(inputs=y,
#                              num_filters=num_filters,
#                              activation=None)
#             if stack > 0 and res_block == 0:  # first layer but not first stack
#                 # linear projection residual shortcut connection to match
#                 # changed dims
#                 x = resnet_layer(inputs=x,
#                                  num_filters=num_filters,
#                                  kernel_size=1,
#                                  strides=strides,
#                                  activation=None,
#                                  batch_normalization=False)
#             x = keras.layers.add([x, y])
#             x = Activation('relu')(x)
#         num_filters *= 2

#     # Add classifier on top.
#     # v1 does not use BN after last shortcut connection-ReLU
#     x = AveragePooling2D(pool_size=8)(x)
#     y = Flatten()(x)
#     outputs = Dense(1,
#                     activation='softmax',
#                     kernel_initializer='he_normal')(y)

#     # Instantiate model.
#     model = Model(inputs=inputs, outputs=outputs)
#     return model


# def resnet_v2(input_shape, depth, num_classes=100):
#     """ResNet Version 2 Model builder [b]

#     Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
#     bottleneck layer
#     First shortcut connection per layer is 1 x 1 Conv2D.
#     Second and onwards shortcut connection is identity.
#     At the beginning of each stage, the feature map size is halved (downsampled)
#     by a convolutional layer with strides=2, while the number of filter maps is
#     doubled. Within each stage, the layers have the same number filters and the
#     same filter map sizes.
#     Features maps sizes:
#     conv1  : 32x32,  16
#     stage 0: 32x32,  64
#     stage 1: 16x16, 128
#     stage 2:  8x8,  256

#     # Arguments
#         input_shape (tensor): shape of input image tensor
#         depth (int): number of core convolutional layers
#         num_classes (int): number of classes (CIFAR10 has 10)

#     # Returns
#         model (Model): Keras model instance
#     """
#     if (depth - 2) % 9 != 0:
#         raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
#     # Start model definition.
#     num_filters_in = 16
#     num_res_blocks = int((depth - 2) / 9)

#     inputs = Input(shape=input_shape)
#     # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
#     x = resnet_layer(inputs=inputs,
#                      num_filters=num_filters_in,
#                      conv_first=True)

#     # Instantiate the stack of residual units
#     for stage in range(3):
#         for res_block in range(num_res_blocks):
#             activation = 'relu'
#             batch_normalization = True
#             strides = 1
#             if stage == 0:
#                 num_filters_out = num_filters_in * 4
#                 if res_block == 0:  # first layer and first stage
#                     activation = None
#                     batch_normalization = False
#             else:
#                 num_filters_out = num_filters_in * 2
#                 if res_block == 0:  # first layer but not first stage
#                     strides = 2    # downsample

#             # bottleneck residual unit
#             y = resnet_layer(inputs=x,
#                              num_filters=num_filters_in,
#                              kernel_size=1,
#                              strides=strides,
#                              activation=activation,
#                              batch_normalization=batch_normalization,
#                              conv_first=False)
#             y = resnet_layer(inputs=y,
#                              num_filters=num_filters_in,
#                              conv_first=False)
#             y = resnet_layer(inputs=y,
#                              num_filters=num_filters_out,
#                              kernel_size=1,
#                              conv_first=False)
#             if res_block == 0:
#                 # linear projection residual shortcut connection to match
#                 # changed dims
#                 x = resnet_layer(inputs=x,
#                                  num_filters=num_filters_out,
#                                  kernel_size=1,
#                                  strides=strides,
#                                  activation=None,
#                                  batch_normalization=False)
#             x = keras.layers.add([x, y])

#         num_filters_in = num_filters_out

#     # Add classifier on top.
#     # v2 has BN-ReLU before Pooling
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = AveragePooling2D(pool_size=8)(x)
#     y = Flatten()(x)
#     outputs = Dense(1,
#                     activation='softmax',
#                     kernel_initializer='he_normal')(y)

#     # Instantiate model.
#     model = Model(inputs=inputs, outputs=outputs)
#     return model


# if version == 2:
#     model = resnet_v2(input_shape=input_shape, depth=depth)
# else:
#     model = resnet_v1(input_shape=input_shape, depth=depth)

# model.compile(loss='binary_crossentropy',
#               optimizer=Adam(learning_rate=lr_schedule(0)),
#               metrics=['accuracy'])
# # Prepare model model saving directory.
# save_dir = os.path.join(os.getcwd(), 'saved_models')
# model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type
# if not os.path.isdir(save_dir):
#     os.makedirs(save_dir)
# filepath = os.path.join(save_dir, model_name)

# # Prepare callbacks for model saving and for learning rate adjustment.
# checkpoint = ModelCheckpoint(filepath=filepath,
#                              monitor='val_acc',
#                              verbose=1,
#                              save_best_only=True)

# lr_scheduler = LearningRateScheduler(lr_schedule)

# lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
#                                cooldown=0,
#                                patience=5,
#                                min_lr=0.5e-6)

# callbacks = [checkpoint, lr_reducer, lr_scheduler]

# # Run training, with or without data augmentation.


# In[ ]:


# if not data_augmentation:
#     print('Not using data augmentation.')
#     Hist=[]
#     for i in range(6):
#         X_val=x_train[i*113:(i+1)*113,:,:,:]
#         Y_val=y_train[i*113:(i+1)*113]
#         indexes=list(range(i*113))+list(range((i+1)*113,678))
#         history = model.fit(x_train[indexes,:,:,:], y_train[indexes],
#                   batch_size=batch_size,
#                   epochs=8,
#                   validation_data=(X_val, Y_val),
#                   shuffle=True,
#                   callbacks=callbacks)
#         Hist.append(history)
# else:
#     print('Using real-time data augmentation.')
#     # This will do preprocessing and realtime data augmentation:
#     datagen = ImageDataGenerator(
#         # set input mean to 0 over the dataset
#         featurewise_center=False,
#         # set each sample mean to 0
#         samplewise_center=False,
#         # divide inputs by std of dataset
#         featurewise_std_normalization=False,
#         # divide each input by its std
#         samplewise_std_normalization=False,
#         # apply ZCA whitening
#         zca_whitening=False,
#         # epsilon for ZCA whitening
#         zca_epsilon=1e-06,
#         # randomly rotate images in the range (deg 0 to 180)
#         rotation_range=0,
#         # randomly shift images horizontally
#         width_shift_range=0.1,
#         # randomly shift images vertically
#         height_shift_range=0.1,
#         # set range for random shear
#         shear_range=0.,
#         # set range for random zoom
#         zoom_range=0.,
#         # set range for random channel shifts
#         channel_shift_range=0.,
#         # set mode for filling points outside the input boundaries
#         fill_mode='nearest',
#         # value used for fill_mode = "constant"
#         cval=0.,
#         # randomly flip images
#         horizontal_flip=True,
#         # randomly flip images
#         vertical_flip=False,
#         # set rescaling factor (applied before any other transformation)
#         rescale=None,
#         # set function that will be applied on each input
#         preprocessing_function=None,
#         # image data format, either "channels_first" or "channels_last"
#         data_format=None,
#         # fraction of images reserved for validation (strictly between 0 and 1)
#         validation_split=0.0)

#     # Compute quantities required for featurewise normalization
#     # (std, mean, and principal components if ZCA whitening is applied).
#     datagen.fit(x_train)

#     # Fit the model on the batches generated by datagen.flow().
#     history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
#                         epochs=epochs, verbose=1, workers=4,
#                         callbacks=callbacks)


# In[ ]:


# from keras.applications.resnet50 import ResNet50
# model = ResNet50(weights='imagenet', pooling=max, include_top = False)

