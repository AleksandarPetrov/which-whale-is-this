# -*- coding: utf-8 -*-
"""
Created on Sat May 19 17:22:25 2018

@author: Rohan

References: 
    -> https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
"""

from data_aug import aug_para, img_loading, createFolder, img_data_aug

# Set up augmentation parameters and equal them to a variable:
augment_para = aug_para(rot_deg = 30, width = 0.1, height = 0.1, shear = 0.1, zoom = 0.1)

# Load image and convert it to array:
x = img_loading('00a29f63.jpg')

print(x.shape)

# Augmentation part:
augmentation_dir = "preview" # change later
createFolder(augmentation_dir) # creates folder, even if exists or not
random_max = 20 # maximum number of images for the random generator
augmentation_name = "whale" # main name for the augmented file

img_data_aug(random_max, augment_para, x, augmentation_dir, augmentation_name)