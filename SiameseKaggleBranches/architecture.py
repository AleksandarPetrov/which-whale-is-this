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

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1) #shape1 = shape=(2,) so shape1[0] = 2 ? i guess

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    #tf.Print(y_pred, tf.shape(y_pred)) #[Note: try to print this]
    margin = accuracy(y_true, y_pred)#1
    a = K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    for d in range(num_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)

def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    print('halo isabelle here')
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    print('halo rohan here')
    print(y_true.dtype)
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))




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

<<<<<<< HEAD
    # # Get the absolute difference between the two vectors
    # L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    # L1_distance = L1_layer([encoded_test, encoded_known])
    #
    # # Add the final layer that connects all of the  distances on the previous layer to the single output
    # prediction = Dense(units=1,
    #                    activation='sigmoid')(L1_distance)
    siamese_net = Model(inputs=[test_input, known_input], outputs=[encoded_test, encoded_known])

    # def compute_euclidean_distance(x, y):
    #     """
    #     Computes the euclidean distance between two tensorflow variables
    #     """
    #
    #     with tf.name_scope('euclidean_distance') as scope:
    #         # d = tf.square(tf.sub(x, y))
    #         # d = tf.sqrt(tf.reduce_sum(d)) # What about the axis ???
    #         d = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(x, y)), 1))
    #         return d
    #
    # def contrastive_loss(left_feature, right_feature, label, margin, is_target_set_train=True):
    #     """
    #     Compute the contrastive loss as in
    #     http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    #     L = 0.5 * (Y) * D^2 + 0.5 * (1-Y) * {max(0, margin - D)}^2
    #     OR MAYBE THAT
    #     L = 0.5 * (1-Y) * D^2 + 0.5 * (Y) * {max(0, margin - D)}^2
    #     **Parameters**
    #      left_feature: First element of the pair
    #      right_feature: Second element of the pair
    #      label: Label of the pair (0 or 1)
    #      margin: Contrastive margin
    #     **Returns**
    #      Return the loss operation
    #     """
    #
    #     with tf.name_scope("contrastive_loss"):
    #         label = tf.to_float(label)
    #         one = tf.constant(1.0)
    #
    #         d = compute_euclidean_distance(left_feature, right_feature)
    #         # first_part = tf.mul(one - label, tf.square(d))  # (Y-1)*(d^2)
    #         # first_part = tf.mul(label, tf.square(d))  # (Y-1)*(d^2)
    #         between_class = tf.exp(tf.mul(one - label, tf.square(d)))  # (1-Y)*(d^2)
    #         max_part = tf.square(tf.maximum(margin - d, 0))
    #
    #         within_class = tf.mul(label, max_part)  # (Y) * max((margin - d)^2, 0)
    #         # second_part = tf.mul(one-label, max_part)  # (Y) * max((margin - d)^2, 0)
    #
    #         loss = 0.5 * tf.reduce_mean(within_class + between_class)
    #
    #         return loss, tf.reduce_mean(within_class), tf.reduce_mean(between_class)



    optimizer = Adam(0.0006)
    siamese_net.compile(loss=contrastive_loss, #(encoded_known, encoded_test, labels)
                        optimizer=optimizer,
                        metrics=[accuracy])
=======
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
>>>>>>> ed6bddc11f22bbd04a222a25ba8f223b7acfe471

    siamese_net.summary()
    
    return siamese_net


