# -*- coding: utf-8 -*-
"""
Created on Sat May 19 19:44:11 2018

@author: Rohan
References: 
    -> https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
"""

import sys
import os
import numpy as np
import random
import keras

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


def aug_para(rot_deg, width, height, shear, zoom):
    a = ImageDataGenerator(
        rotation_range = rot_deg, # random rotation -> from 0 to rotation_range (must be 0-180)
        width_shift_range = width, # random translation with fraction of total image width
        height_shift_range = height, # random translation with fraction of total image height
        rescale = 1./255, # random rescaling -> multiplies the image by this value
        shear_range = shear, # "Shear angle in counter-clockwise direction in degrees"
        zoom_range = zoom, # random zoom inside picture
        horizontal_flip = True, # "randomly flipping half of the images horizontally"
        fill_mode = 'nearest' # strategy used for filling in newly created pixels
        )
    return a

def img_loading(img_path):
    # load one PIL - Python Imaging Library - image
    img  = load_img(img_path)
    x = img_to_array(img)
    # Convert the image to an array of numbers
    x = img_to_array(img) # convert image to array -> im_to_array
    # The goal now is to reshape the image to a rank 4 array
    if len(x.shape) == 2: # if we are working with grayscale images -> shape is 2
        x = x.reshape((1,1,) + x.shape)  
    elif len(x.shape) == 3: # if we are working with RGB images -> shape is 3 (because there are 3 channels)
        x = x.reshape((1,) + x.shape)
    # elif len(x.shape) == 4: # this is for rank 4 tensor
    #    return x
    else:
        print('Error! The shape of the image is ', len(x.shape))
        sys.exit()
    return x

# Taken from: https://gist.github.com/keithweaver/562d3caa8650eefe7f84fa074e9ca949
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


def img_data_aug(random_max, augmentation_parameters, image_array, augmentation_dir, augmentation_name):
    rand_int = random.randint(5, random_max)
    # print(rand_int)
    # generates rand_int images: 
    i = 1
    # print(aug_folder_name)
    for batch in augmentation_parameters.flow(image_array, batch_size = 1,
                          save_to_dir = augmentation_dir, save_prefix = augmentation_name, save_format = 'jpeg'):
        i += 1
        if rand_int > i:
            break  # otherwise the generator would loop indefinitely
