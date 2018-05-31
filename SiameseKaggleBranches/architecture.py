import tensorflow as tf
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


# margin for the triplet loss
margin = 1.0


def W_init(shape, name=None):
    """Initialize weights as in paper"""
    values = rng.normal(loc=0, scale=1e-2, size=shape)
    return K.variable(values, name=name)


def b_init(shape, name=None):
    """Initialize bias as in paper"""
    values = rng.normal(loc=0.5, scale=1e-2, size=shape)
    return K.variable(values, name=name)


def basicSiameseGenerator(parent_dir):
    input_shape = (64, 64, 1)

    anchor_input = Input(input_shape)  # this is where we feed the image we want to test if is the same as the known image
    positive_input = Input(input_shape)  # this is where we feed the known image
    negative_input = Input(input_shape)
    # It doesn't matter which is which, the network is completely symmetrical

    # Load kaggle model and drop the last layer
    
    kaggleModel = load_model(os.path.join(parent_dir, 'keras_model.h5'))
    kaggleModel.pop() # remove the last layer
    kaggleModel.trainable = False # fix the weights
    print("Kaggle network (without output layer):")
    kaggleModel.summary() #print the architecture

    # BUILDING THE LEGS OF THE SIAMESE NETWORK
    convnet = Sequential()
    convnet.add(kaggleModel)
    convnet.add(Dense(units=4096,
                      activation="sigmoid"))
    print("Single Siamese branch:")
    convnet.summary()

    # Add the two inputs to the leg (passing the two inputs through the same network is effectively the same as having
    # three legs with shared weights

    # encoded_test = convnet(test_input)
    # encoded_known = convnet(known_input)

    anchor_output = convnet(anchor_input)
    positive_output = convnet(positive_input)
    negative_output = convnet(negative_input)

    def compute_triplet_loss(anchor_output,positive_output,negative_output):
        d_pos = tf.reduce_sum(tf.square(anchor_output - positive_output), 1)
        d_neg = tf.reduce_sum(tf.square(anchor_output - negative_output), 1)

        _square = Lambda(lambda x: x**2)
        _squarert = Lambda(lambda x: K.sqrt(x))

        def sum_all(x):
            _sum = Dense(1, kernel_initializer='ones', bias_initializer='zeros')
            return _sum(x)

        # Normalize the differences
        n1 = _squarert(sum_all(_square(d_pos)))
        n2 = _squarert(sum_all(_square(d_neg)))

        loss = tf.maximum(0., margin + n1 - n2)
        loss = tf.reduce_mean(loss)
        return loss

    # Get the absolute difference between the two vectors
    L1_layer = Lambda(lambda tensors: compute_triplet_loss(tensors[0],tensors[1], tensors[2])) #lambda tensors: K.abs(tensors[0] - tensors[1]
    L1_distance = L1_layer([anchor_output, positive_output,negative_output])


    # check the outputs of L1_distance
    # print("here",tf.shape(L1_layer))

    # Add the final layer that connects all of the  distances on the previous layer to the single output
    prediction = Dense(units=1,
                       activation='sigmoid')(L1_distance)
    siamese_net = Model(inputs=[anchor_input, positive_input, negative_input], outputs=prediction)

    optimizer = Adam(0.0006)
    siamese_net.compile(loss="binary_crossentropy",
                        optimizer=optimizer,
                        metrics=['accuracy'])

    print("Full Siamese network:")
    siamese_net.summary()
    
    return siamese_net


parent_dir = './../../DATA/'#sys.argv[1]

basicSiameseGenerator(parent_dir)