import tensorflow as tf
import numpy as np
import keras
from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten, MaxPooling2D, Dropout
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


def W_init(shape, name=None):
    """Initialize weights as in paper"""
    values = rng.normal(loc=0, scale=1e-2, size=shape)
    return K.variable(values, name=name)


def b_init(shape, name=None):
    """Initialize bias as in paper"""
    values = rng.normal(loc=0.5, scale=1e-2, size=shape)
    return K.variable(values, name=name)


def alternate_network():
    input_shape = (64, 64, 1)
    #build convnet to use in each siamese 'leg'
    convnet = Sequential()
    convnet.add(Conv2D(64,(7,7),activation='relu',input_shape=input_shape))
    convnet.add(MaxPooling2D())
    convnet.add(Conv2D(128,(4,4),activation='relu'))
    convnet.add(MaxPooling2D())
    convnet.add(Conv2D(128,(4,4),activation='relu'))
    convnet.add(MaxPooling2D())
    convnet.add(Conv2D(256,(2,2),activation='relu'))
    convnet.add(Flatten())
    convnet.add(Dense(4096,activation="sigmoid"))
    return convnet

def basicSiameseGenerator(parent_dir,trainable,alternate = True):
    input_shape = (64, 64, 1)

    test_input = Input(input_shape)  # this is where we feed the image we want to test if is the same as the known image
    known_input = Input(input_shape)  # this is where we feed the known image
    # It doesn't matter which is which, the network is completely symmetrical
    
    # Load kaggle model and drop the last layer
    if alternate:
        model = alternate_network()
    else:        
        model = load_model(os.path.join(parent_dir, 'keras_model.h5'))
        model.layers.pop() # remove the last layer
        model.layers.pop()
        model.layers.pop()
        model.trainable = trainable # fix the weights

        
        
    model.summary() #print the architecture

    # BUILDING THE LEGS OF THE SIAMESE NETWORK
    convnet = Sequential()
    convnet.add(model)
    convnet.add(Dense(units=1000,
                       activation='sigmoid'))
    convnet.summary()

    # Add the two inputs to the leg (passing the two inputs through the same network is effectively the same as having
    # two legs with shared weights
    encoded_test = convnet(test_input)
    encoded_known = convnet(known_input)

    # Get the absolute difference between the two vectors
    L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_test, encoded_known])
    
    # Add the final layer that connects all of the  distances on the previous layer to the single output
    prediction = Dense(units=2,
                       activation='softmax')(L1_distance)
    siamese_net = Model(input=[test_input, known_input], output=prediction)

    optimizer = Adam(0.00008)
    siamese_net.compile(loss="binary_crossentropy",
                        optimizer=optimizer,
                        metrics=['binary_accuracy'])

    siamese_net.summary()
    
    return siamese_net


