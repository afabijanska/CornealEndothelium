# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 12:00:37 2017

@author: Ania
"""
import os
import numpy as np
import configparser
from keras.models import model_from_json

from skimage import io
from skimage.color import rgb2gray

#---------------------------------------------------------------------------------------------

#read config
config = configparser.RawConfigParser()
config.read('configuration.txt')

#data location info
original_imgs_test = config.get('data paths', 'test_data')
predictions_test = config.get('data paths', 'test_preds')

#patches info
patch_height = int(config.get('data attributes', 'patch_height'))
patch_width = int(config.get('data attributes', 'patch_width'))
stride_height = int(config.get('testing settings', 'stride_height'))
stride_width = int(config.get('testing settings', 'stride_width'))

assert (stride_height < patch_height and stride_width < patch_width)

#load trained model
model = model_from_json(open('model.json').read())
model.load_weights('bestWeights.h5')

#---------------------------------------------------------------------------------------------
def get_patches(img, patch_h, patch_w, stride_h, stride_w):
    
    h = img.shape[0] 
    w = img.shape[1] 
    
    assert ((h-patch_h)%stride_h==0 and (w-patch_w)%stride_w==0)
    
    H = (h-patch_h)//stride_h+1
    W = (w-patch_w)//stride_w+1
   
    patches = np.empty((W*H,patch_h,patch_w,1))
    iter_tot = 0
    
    for h in range(H):
        for w in range(W):
            patch = img[h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w,:]
            patches[iter_tot]=patch
            iter_tot +=1
    
    return patches

#---------------------------------------------------------------------------------------------
    
def pred_to_img(pred, patch_h, patch_w):
    
    assert (len(pred.shape)==3)
    assert (pred.shape[2]==2 )
    pred_image = np.empty((pred.shape[0],pred.shape[1])) 
    
    for i in range(pred.shape[0]):
        for pix in range(pred.shape[1]):
            pred_image[i,pix]=pred[i,pix,1]
        
    pred_image = np.reshape(pred_image,(pred_image.shape[0], patch_h, patch_w, 1))
    
    return pred_image

#---------------------------------------------------------------------------------------------

def build_img_from_patches(preds, img_h, img_w, stride_h, stride_w):
    
    patch_h = preds.shape[1]
    patch_w = preds.shape[2]
    
    H = (img_h-patch_h)//stride_h+1
    W = (img_w-patch_w)//stride_w+1
  
    prob = np.zeros((img_h, img_w, 1)) 
    _sum = np.zeros((img_h, img_w, 1))

    k = 0
    
    for h in range(H):
        for w in range(W):
            prob[h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w,:]+=preds[k]
            _sum[h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w,:]+=1
            k+=1
    
    final_avg = prob/_sum
    print (final_avg.shape)
    
    return final_avg

#---------------------------------------------------------------------------------------------
    
def add_outline(img, patch_h, patch_w, stride_h, stride_w):
    
    print(img.shape)
    img_h = img.shape[0]               
    img_w = img.shape[1]               
    leftover_h = (img_h-patch_h)%stride_h   
    leftover_w = (img_w-patch_w)%stride_w   
    
    if (leftover_h != 0):
        tmp = np.zeros((img_h+(stride_h-leftover_h),img_w,1))
        tmp[0:img_h,0:img_w,:] = img
        img = tmp
    if (leftover_w != 0):
        tmp = np.zeros((img.shape[0],img_w+(stride_w - leftover_w)))
        tmp[0:img.shape[0],0:img_w] = img
        img = tmp
        
    print ("new image shape: \n" +str(img.shape))
    
    return img

#---------------------------------------------------------------------------------------------
    
def predict_img(org_path):
    
   org = io.imread(org_path)
   org = rgb2gray(org)
   org = np.asarray(org, dtype='float16')
    
   print ('original image: ' + org_path)
    
   height = org.shape[0]
   width = org.shape[1]
    
   print ('image dims: (%d x %d)' % (height, width))
    
   org = np.reshape(org,(height, width, 1))
   assert(org.shape == (height, width, 1))
    
   org = add_outline(org, patch_height, patch_width, stride_height, stride_width)
   
   new_height = org.shape[0]
   new_width = org.shape[1]
   
   print ('new image dims: (%d x %d)' % (new_height, new_width))
   
   org = np.reshape(org,(new_height, new_width, 1))
   assert(org.shape == (new_height, new_width, 1))
   
   patches = get_patches(org, patch_height, patch_width, stride_height, stride_width)

   predictions = model.predict(patches, batch_size=32, verbose=2)
   print ("predicted images size :")
   print (predictions.shape)
    
   pred_patches = pred_to_img(predictions, patch_height, patch_width)
   pred_img = build_img_from_patches(pred_patches, new_height, new_width, stride_height, stride_width)
 
   return pred_img

#================================================

for path, subdirs, files in os.walk(original_imgs_test):
    for i in range(len(files)):
        
        org_path = original_imgs_test + files[i]
        pred_path = predictions_test + files[i]
        
        print(org_path)
        prediction = predict_img(org_path)
        prediction2 = np.reshape(prediction, (prediction.shape[0], prediction.shape[1]))
        prediction3 = 255*prediction2/np.max(prediction2)
        io.imsave(pred_path, prediction3.astype(np.uint8))
        
        