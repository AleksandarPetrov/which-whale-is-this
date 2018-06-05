import numpy as np
import keras
from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten, MaxPooling2D
from keras.models import Model, Sequential, load_model
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD, Adam
from keras.losses import binary_crossentropy
import numpy.random as rng
import sys
import os
import glob
from sklearn.preprocessing import LabelEncoder
from numpy import zeros
import pandas as pd
from keras.callbacks import ModelCheckpoint
from gen_imageName_dict import gen_imageName_dict
from collections import Counter
import matplotlib.pyplot as plt
from data_aug import img_data_aug_array, aug_para

class SiameseDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, parent_dir,X_dataset,y_labels, batch_size=32, dim=(64, 64), n_channels=1, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.shown = 0
        self.parent_dir = parent_dir
        self.X_dataset = X_dataset
        self.y_labels = np.array(y_labels)
        self.indexes = np.arange(0,np.size(y_labels,0))
        self.augParam = aug_para(rot_deg=45,
                            width=0,
                            height=0,
                            shear=0,
                            zoom=0)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.y_labels) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of IDs
        y_labels = [self.y_labels[k] for k in indexes]
        # Generate data
        
        X, y = self.__data_generation(y_labels,indexes)
        return X, y



    def __data_generation(self, y_labels,indexes):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X1 = np.empty((self.batch_size, *self.dim, self.n_channels))
        X2 = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)
        total_indexes = np.arange(0,np.size(self.y_labels))
        i = 0       
        # Generate data
        for label in y_labels:
            # Store sample 1
            ind1 = indexes[i]
            img = self.X_dataset[ind1]
            img = img[:, :, np.newaxis]
            X1[i,] = img
            
            # For the second one take one from the same class is i is even, otherwise one with a different class,
            # also checks if the class is not new_whale and if there is at least one other picture of the same whale
            ###DATA AUGMENTATION SHOULD BE PUT HERE AND A LOT OF THINGS ADJUSTED
            if i%2==0:
                
                positive_indexes = np.where(self.y_labels == label)
                positive_indexes = positive_indexes[0]
                ind2 = np.random.choice(positive_indexes,size=1)
                img = self.X_dataset[ind2]
                
                # Augment:
                img = img_data_aug_array(self.augParam, img[0, :, :])
                img = img[:, :, np.newaxis]
                X2[i,] = img
                # Store class
                y[i] = 1
                i += 1
            else:
                negative_indexes = np.delete(total_indexes,positive_indexes)
                ind2 = np.random.choice(negative_indexes,size=1)
                img = self.X_dataset[ind2]
    
                # Augment:
                img = img_data_aug_array(self.augParam, img[0, :, :])
                img = img[:, :, np.newaxis]
                X2[i,] = img
                # Store class
                y[i] = 0
                i += 1
            
        return [X1, X2], y