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

    def __init__(self, list_IDs, labels, parent_dir, batch_size=32, dim=(32, 32), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.shown = 0
        self.parent_dir = parent_dir

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X1 = np.empty((self.batch_size, *self.dim, self.n_channels))
        X2 = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample 1
            img = np.load(os.path.join(self.parent_dir, 'train_npy/' + ID + '.npy'))
            img = img[:, :, np.newaxis]
            X1[i,] = img
            X1_label = self.labels[ID]

            listOfPicturesOfSameWhale = [k for k in list(self.labels.keys()) if self.labels[k] == X1_label]
            # For the second one take one from the same class is i is even, otherwise one with a different class,
            # also checks if the class is not new_whale and if there is at least one other picture of the same whale
            ###DATA AUGMENTATION SHOULD BE PUT HERE AND A LOT OF THINGS ADJUSTED
            if i%2==0 and X1_label!=0:
                X2_ID = np.random.choice(listOfPicturesOfSameWhale, 1)[0]
                img = np.load(os.path.join(self.parent_dir, 'train_npy/' + X2_ID + '.npy'))
                # Augment:
                augParam = aug_para(rot_deg=min(10, max(-10, np.random.normal(loc=0.0, scale=5))),
                                    width=min(0.1, max(-0.1, np.random.normal(loc=0.0, scale=0.05))),
                                    height=min(0.1, max(-0.1, np.random.normal(loc=0.0, scale=0.05))),
                                    shear=min(0.1, max(-0.1, np.random.normal(loc=0.0, scale=0.05))),
                                    zoom=min(0.1, max(-0.1, np.random.normal(loc=0.0, scale=0.05))))
                img = img_data_aug_array(augParam, img)
                #if(self.shown<30):
                #    plt.imshow(img)
                #    plt.savefig('foo'+str(self.shown)+'.png')
                #    self.shown=self.shown+1
                img = img[:, :, np.newaxis]
                X2[i,] = img
                # Store class
                y[i] = 1
            else:
                listOfPicturesOfDifferentWhales = [k for k in list(self.labels.keys()) if (self.labels[k] != X1_label or self.labels[k]!=0) and k != ID]
                X2_ID = np.random.choice(listOfPicturesOfDifferentWhales, 1)[0]
                img = np.load(os.path.join(self.parent_dir, 'train_npy/' + X2_ID + '.npy'))
                # Augment:
                augParam = aug_para(rot_deg=min(10, max(-10, np.random.normal(loc=0.0, scale=5))),
                                    width=min(0.1, max(-0.1, np.random.normal(loc=0.0, scale=0.05))),
                                    height=min(0.1, max(-0.1, np.random.normal(loc=0.0, scale=0.05))),
                                    shear=min(0.1, max(-0.1, np.random.normal(loc=0.0, scale=0.05))),
                                    zoom=min(0.1, max(-0.1, np.random.normal(loc=0.0, scale=0.05))))
                img = img_data_aug_array(augParam, img)
                #if (self.shown < 30):
                #    plt.imshow(img)
                #    plt.savefig('foo'+str(self.shown)+'.png')
                #    self.shown=self.shown+1
                img = img[:, :, np.newaxis]
                X2[i,] = img
                # Store class
                y[i] = 0



        return [X1, X2], y