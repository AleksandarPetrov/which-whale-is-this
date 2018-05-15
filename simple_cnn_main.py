# -*- coding: utf-8 -*-
"""
Created on Tue May  8 10:00:55 2018

@author: Kuba
"""
#from my_classes import DataGenerator
from dataGeneratorSimpleCNN import DataGenerator
from gen_id_dict import gen_id_dict
from simple_cnn import gen_model
from sklearn.preprocessing import LabelEncoder
from numpy import zeros
import pandas as pd

# Some useful directories
test_dir = '../DATA/test_npy'
train_dir = '../DATA/train_npy'
labels_dir = '../DATA/train.csv'

# Reading of labels and corresponding image names
classes = pd.read_csv(labels_dir)
list_ids = list(classes.Id)
file_name = list(classes.Image)
n_files = len(file_name)
file_name = [file[:-4] for file in file_name]

# Label encoder, changes the label names to integers for use in generator
le = LabelEncoder()
le.fit(list_ids)
n_classes = len(le.classes_)
labels_int = list(le.fit_transform(list_ids))

# Parameters for Generator
partition = gen_id_dict(test_dir,train_dir)
labels = dict(zip(file_name,labels_int))

params = {'dim': (250,500),
          'batch_size': 64,
          'n_classes': n_classes,
          'n_channels': 1,
          'shuffle': True}


# Generator
training_generator = DataGenerator(partition['train'], labels, **params)

# Network parameters
conv_param = zeros((2,2))
dense_param = zeros((1,1))
in_shape = (250,500,1)
        
conv_param[0,0], conv_param[0,1] = 8, 5
conv_param[1,0], conv_param[1,1] = 4, 15     
dense_param[0] = 1000

# Model generation
modelz = gen_model(conv_param,dense_param,in_shape,n_classes)
modelz.fit_generator(generator = training_generator, use_multiprocessing=True,verbose = 1)

modelz.save('simpleCNN.h5')