# -*- coding: utf-8 -*-
"""
Created on Tue May  8 10:00:55 2018

@author: Kuba
"""
#from my_classes import DataGenerator
from dataGeneratorSimpleCNN import DataGenerator
from gen_imageName_dict import gen_imageName_dict
from simple_cnn import gen_model
from sklearn.preprocessing import LabelEncoder
from numpy import zeros
from tf_cnnvis import *
import pandas as pd
from keras.callbacks import ModelCheckpoint


# Some useful directories
test_dir = '../DATA/test_npy'
train_dir = '../DATA/train_npy'
labels_dir = '../DATA/train.csv'

# Reading of labels and corresponding image names
classes = pd.read_csv(labels_dir)
list_ids = list(classes.Id) # whale ids not image ids
file_names = list(classes.Image)
n_files = len(file_names)
file_names = [file[:-4] for file in file_names]

# Label encoder, changes the label names to integers for use in generator
le = LabelEncoder()
le.fit(list_ids) # whale ids not image ids
n_classes = len(le.classes_)
labels_int = list(le.fit_transform(list_ids))

# Parameters for Generator
partition = gen_imageName_dict(test_dir,train_dir, 0.2)
imageName_ID_dict = dict(zip(file_names,labels_int))

params = {'dim': (250,500),
          'batch_size': 64,
          'n_classes': n_classes,
          'n_channels': 1,
          'shuffle': True}


# Generator
training_generator = DataGenerator(partition['train'], imageName_ID_dict, **params)
validation_generator = DataGenerator(partition['validation'], imageName_ID_dict, **params)

# Network parameters
conv_param = zeros((2,2))
dense_param = zeros((1,1))
in_shape = (250,500,1)
        
conv_param[0,0], conv_param[0,1] = 8, 5
conv_param[1,0], conv_param[1,1] = 4, 15     
dense_param[0] = 1000

#Saving callback
filepath="../DATA/weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# Model generation
modelz = gen_model(conv_param,dense_param,in_shape,n_classes)

modelz.fit_generator(generator = training_generator,
                     use_multiprocessing=True,
                     epochs=3,
                     verbose = 1,
                     callbacks=callbacks_list)


# Save final
modelz.save('../DATA/simpleCNN.h5')
