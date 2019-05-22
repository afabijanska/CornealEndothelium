import numpy as np
import configparser
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, core, Dropout, concatenate
from keras.callbacks import ModelCheckpoint,TensorBoard

from helpers import load_hdf5

#---------------------------------------------------
#read config
config = configparser.RawConfigParser()
config.read('configuration.txt')

#train data
train_imgs = config.get('data paths', 'train_imgs')
train_gt = config.get('data paths', 'train_gt')

#training settings
N_epochs = int(config.get('training settings', 'N_epochs'))
batch_size = int(config.get('training settings', 'batch_size'))

#patch properties
patch_height = int(config.get('data attributes','patch_height'))
patch_width = int(config.get('data attributes','patch_width'))
patch_num = int(config.get('data attributes','patch_num'))

#----------------------------------------------------
def masks_reshape(masks):
    
    im_h = masks.shape[1]
    im_w = masks.shape[2]
    
    masks = np.reshape(masks,(masks.shape[0],im_h*im_w))
    new_masks = np.empty((masks.shape[0],im_h*im_w,2), dtype = 'float16')
    for i in range(masks.shape[0]):
        for j in range(im_h*im_w):
            if  masks[i,j] == 0:
                new_masks[i,j,0]=1
                new_masks[i,j,1]=0
            else:
                new_masks[i,j,0]=0
                new_masks[i,j,1]=1
                
    return new_masks 

#---------------------------------------------------
def getShallowUnet(patch_height,patch_width,n_ch):
    #
    inputs = Input((patch_height, patch_width,n_ch))
    #
    conv1 = Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), padding="same", activation="relu")(conv1)
    print(conv1.shape)
    #
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    print(pool1.shape)
    #
    conv2 = Conv2D(64, (3, 3), padding="same", activation="relu")(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), padding="same", activation="relu")(conv2)
    print(conv2.shape)
    #
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    print(pool2.shape)
    #
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    print(conv3.shape)
    #
    up1 = concatenate([UpSampling2D(size=(2, 2))(conv3), conv2], axis=-1)
    print(up1.shape)
    #
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)
    print(conv4.shape)
    #
    up2 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv1], axis=-1)
    print(up2.shape)
    #
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)
    print(conv5.shape)
    #
    conv6 = Conv2D(2, (1, 1), activation='relu', padding='same')(conv5)
    print(conv6.shape)
    conv6 = core.Reshape((2,patch_height*patch_width))(conv6)
    print(conv6.shape)
    conv6 = core.Permute((2,1))(conv6)
    print(conv6.shape)
    #
    conv7 = core.Activation('softmax')(conv6)
    print(conv7.shape)

    model = Model(inputs=inputs, outputs=conv7)

    model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
    model.summary()

    return model

#load data
X_train = load_hdf5(train_imgs)
X_train = X_train.reshape((patch_num, patch_height, patch_width, 1))

Y_train = load_hdf5(train_gt)
Y_train = Y_train.reshape((patch_num, patch_height, patch_width,1))
Y_train = masks_reshape(Y_train)

#train
model = getShallowUnet(patch_width, patch_height,1)
json_string = model.to_json()
open('model.json', 'w').write(json_string)
checkpointer = ModelCheckpoint('bestWeights.h5', verbose=1, monitor='val_loss', mode='auto', save_best_only=True) 
tbCallback = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)

model.fit(X_train, Y_train, epochs = N_epochs, batch_size = batch_size, verbose=2, shuffle=True, validation_split=0.2, callbacks=[checkpointer,tbCallback])
model.save_weights('lastWeights.h5', overwrite=True)