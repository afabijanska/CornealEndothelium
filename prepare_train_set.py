# -*- coding: utf-8 -*-
"""
Created on Tue May 30 19:40:00 2017

@author: Ania
"""

import os
import random
import numpy as np
import configparser
import matplotlib.pyplot as plt

from helpers import write_hdf5

from skimage import io
from skimage.color import rgb2gray

#------------Path of the images --------------------------------------------------------------

config = configparser.RawConfigParser()
config.read('configuration.txt')

dataset_path = config.get('data paths', 'path_local')
original_imgs_train = config.get('data paths', 'path_org')
groundTruth_imgs_train = config.get('data paths', 'path_gt')

train_imgs = config.get('data paths', 'train_imgs')
train_gt = config.get('data paths', 'train_gt')

#-----------Properties of train data --------------------------------------------------

N_train = int(config.get('image props', 'N_train'))
n_channels = int(config.get('image props', 'n_channels'))
height = int(config.get('image props', 'height'))
width = int(config.get('image props', 'width'))

patch_height = int(config.get('data attributes','patch_height'))
patch_width = int(config.get('data attributes','patch_width'))
patch_num = int(config.get('data attributes','patch_num'))


def get_dataset(img_dir, gt_dir, n_imgs):
    
    imgs = np.empty((n_imgs, height, width)) 
    groundTruth = np.empty((n_imgs, height, width))
    
    for path, subdirs, files in os.walk(img_dir):
        for i in range(len(files)):
            
            org_path = img_dir + files[i]
            org = io.imread(org_path)
            org = rgb2gray(org)
            
            imgs[i,:,:] = org;
            plt.imshow(org, cmap='gray')
            
            gt_path = gt_dir + files [i]
            gt = io.imread(gt_path)
            gt = gt[:,:,0]
            groundTruth[i,:,:] = gt;
        
    imgs = np.reshape(imgs,(n_imgs, height, width))    
    gts = np.reshape(groundTruth,(n_imgs, height, width))
    gts = gts/255
   
    return imgs, gts
            
#Load the original data and return the extracted patches for training/testing

def extract_random(full_imgs, full_masks, patch_h, patch_w, N_patches):
    
    if (N_patches%full_imgs.shape[0] != 0):
        print ("N_patches: plase enter a multiple of train images")
        print (N_patches)
        print(full_imgs.shape[0])
        exit()

    patches = np.empty((N_patches,patch_h,patch_w))
    patches_masks = np.empty((N_patches,patch_h,patch_w))
    img_h = full_imgs.shape[1] 
    img_w = full_imgs.shape[2] 

    patch_per_img = int(N_patches/full_imgs.shape[0])
    print ("patches per full image: " +str(patch_per_img))
    iter_tot = 0   
    for i in range(full_imgs.shape[0]): 
        k=0
        while k <patch_per_img:
            x_center = random.randint(0+int(patch_w/2),img_w-int(patch_w/2))
            y_center = random.randint(0+int(patch_h/2),img_h-int(patch_h/2))
            patch = full_imgs[i,y_center-int(patch_h/2):y_center+int(patch_h/2),x_center-int(patch_w/2):x_center+int(patch_w/2)]
            patch_mask = full_masks[i,y_center-int(patch_h/2):y_center+int(patch_h/2),x_center-int(patch_w/2):x_center+int(patch_w/2)]
            patches[iter_tot]=patch
            patches_masks[iter_tot]=patch_mask
            iter_tot +=1
            k+=1
            
    return patches, patches_masks

#getting the training data
imgs_train, groundTruth_train = get_dataset(original_imgs_train, groundTruth_imgs_train, N_train)
patches_imgs_train, patches_masks_train = extract_random(imgs_train, groundTruth_train, patch_height, patch_width, patch_num)
 
print ("saving train datasets")
write_hdf5(patches_imgs_train, train_imgs)
write_hdf5(patches_masks_train, train_gt)              
            
            
            
    
    