#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 15:00:28 2018

@author: isabelle
"""

import cv2
import glob
import csv
import numpy as np
from pathlib import Path
import sys
import os

####################################################################################

dataDir = sys.argv[1]

trainLabels_dir = os.path.join(dataDir,"train.csv")

trainFiles = glob.glob (os.path.join(dataDir,"trainUniformBW/*.jpg"))

####################################################################################

def gen_npy_data(Files):
    data = []
    for myFile in Files:
        print(myFile)
        image = cv2.imread (myFile)
        data.append(image)  
    data = np.array(data)
    return data

def gen_npy_labels(trainLabels_dir):
    labels = []
    with open(trainLabels_dir, 'rb') as csvfile:
         labelreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
         idx = 0
         for row in labelreader:         
             string =  row[0] 
             if idx != 0: # forget about first line
                 label = string.split(',')[1]
                 labels.append(label) 
             idx = idx + 1             
    return labels

####################################################################################
#################### Training data #################################################
####################################################################################

my_file = Path(os.path.join(dataDir,"trainData.npy"))
if my_file.is_file() == 0:
    ## if trainData.npy does no exists
    trainData = gen_npy_data(trainFiles)

    
else:
    # open the existing file
    trainData = np.load("/home/isabelle/Documents/Education/Masters/Fourth_year/Q4/Deep_learning/data/trainData.npy")

####################################################################################

labels = gen_npy_labels(trainLabels_dir)

####################################################################################

## check number of labels same as number of images
print np.shape(labels)[0] == np.shape(trainData)[0]


#train_data_array = np.array(train_data)
#
#Data_n_Labels_dict = {}
#
#Data_n_Labels_dict['data'] = train_data_array
#                  
#Data_n_Labels_dict['label'] = labels 

print('trainData shape:', np.array(trainData).shape) # shape will be (no_images, width, height, depth)


target_labels_file = os.path.join(dataDir,"trainImagesLabels.npy")
target_train_data_file = os.path.join(dataDir,"trainData.npy")
#np.save('/home/isabelle/Documents/Education/Masters/Fourth_year/Q4/Deep_learning/data/trainUniformBW/trainImagesLabels.npy', labels)

np.save(target_labels_file, labels)
np.save(target_train_data_file, trainData)