
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


def W_init(shape, name=None):
    """Initialize weights as in paper"""
    values = rng.normal(loc=0, scale=1e-2, size=shape)
    return K.variable(values, name=name)


def b_init(shape, name=None):
    """Initialize bias as in paper"""
    values = rng.normal(loc=0.5, scale=1e-2, size=shape)
    return K.variable(values, name=name)


def basicSiameseGenerator(parent_dir):
    input_shape = (250, 500, 1)

    test_input = Input(input_shape)  # this is where we feed the image we want to test if is the same as the known image
    known_input = Input(input_shape)  # this is where we feed the known image
    # It doesn't matter which is which, the network is completely symmetrical

    # Load kaggle model
    kaggleModel = load_model(os.path.join(parent_dir, 'keras_model.h5'))
    kaggleModel.summary()

    # BUILDING THE LEGS OF THE SIAMESE NETWORK
    convnet = Sequential()

    convnet.add(Conv2D(filters=16,
                       kernel_size=(16, 16),
                       activation='relu',
                       input_shape=input_shape,
                       kernel_initializer=W_init,
                       kernel_regularizer=l2(2e-4),
                       use_bias=True)
                )
    convnet.add(MaxPooling2D())

    convnet.add(Conv2D(filters=32,
                       kernel_size=(13, 13),
                       activation='relu',
                       kernel_regularizer=l2(2e-4),
                       kernel_initializer=W_init,
                       bias_initializer=b_init,
                       use_bias=False)
                )
    convnet.add(MaxPooling2D())

    convnet.add(Conv2D(filters=32,
                       kernel_size=(10, 10),
                       activation='relu',
                       kernel_regularizer=l2(2e-4),
                       kernel_initializer=W_init,
                       use_bias=True)
                )
    convnet.add(MaxPooling2D())

    convnet.add(Conv2D(filters=64,
                       kernel_size=(7, 7),
                       activation='relu',
                       kernel_initializer=W_init,
                       kernel_regularizer=l2(2e-4),
                       bias_initializer=b_init,
                       use_bias=False)
                )
    convnet.add(MaxPooling2D())

    convnet.add(Conv2D(filters=64,
                       kernel_size=(4, 4),
                       activation='relu',
                       kernel_initializer=W_init,
                       kernel_regularizer=l2(2e-4),
                       bias_initializer=b_init,
                       use_bias=True)
                )

    convnet.add(Flatten())

    convnet.add(Dense(units=4096,
                      activation="sigmoid",
                      kernel_regularizer=l2(1e-3),
                      kernel_initializer=W_init,
                      bias_initializer=b_init)
                )
    convnet.summary()

    # Add the two inputs to the leg (passing the two inputs through the same network is effectively the same as having
    # two legs with shared weights
    encoded_test = convnet(test_input)
    encoded_known = convnet(known_input)

    # Get the absolute difference between the two vectors
    L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_test, encoded_known])

    # Add the final layer that connects all of the  distances on the previous layer to the single output
    prediction = Dense(units=1,
                       activation='sigmoid',
                       bias_initializer=b_init
                       )(L1_distance)

    siamese_net = Model(inputs=[test_input, known_input], outputs=prediction)

    optimizer = Adam(0.00006)
    siamese_net.compile(loss="binary_crossentropy",
                        optimizer=optimizer,
                        metrics=['accuracy'])

    return siamese_net
