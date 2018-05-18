#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 12:38:40 2018

@author: isabelle
"""

## Keras ##
from keras.optimizers import RMSprop, Adagrad, Adadelta, Adam # optimizers
from keras.models import Sequential # Model
from keras.layers import Dense,Conv2D,Flatten,MaxPooling2D # Layers
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, TensorBoard # Callbacks
from keras.utils import np_utils
## Sklearn ##
from sklearn.model_selection import TimeSeriesSplit
## Hyperas ##
from hyperas.distributions import uniform
## Hyperopt ##
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
## Datasets ##
from keras.datasets import mnist

## Matplotlib ##
from matplotlib import pyplot as plt

import numpy as np
import sys
import os
import math

from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import cross_val_score


###########################################################
################# Other data ##############################
###########################################################
# Some useful directories
trainData = np.load('/home/isabelle/Documents/Education/Masters/Fourth_year/Q4/Deep_learning/data/trainData.npy')
trainLabels = np.load('/home/isabelle/Documents/Education/Masters/Fourth_year/Q4/Deep_learning/data/trainLabels.npy')

n_classes = len(set(trainLabels)) #4251

#############################################################
######################## Mnist data ###########################
#############################################################
#(X_train, y_train), (X_test, y_test) = mnist.load_data()
## reshape the data from  (n, width, height) to (n, depth, width, height).
#X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
#X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
## convert data to float32 and normalize our data values to range [0,1]
#X_train = X_train.astype('float32')
#X_test = X_test.astype('float32')
#X_train /= 255
#X_test /= 255
## Convert 1-dimensional class arrays to 10-dimensional class matrices
#Y_train = np_utils.to_categorical(y_train, 10)
#Y_test = np_utils.to_categorical(y_test, 10)

############################################################
###### Create the folds for cross-validation ###############
############################################################
n_folds = 10
tscv = TimeSeriesSplit(n_splits =n_folds)

############################################################
############## Hyperparameter Optimization #################
############################################################

# the callbacks
earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001, cooldown = 2)


# Hyperparameters to tune:     
#1) no_conv_layers
#2) no_filters_conv
#3) row_filter_conv_kernel
#4) cols_filter_conv_kernel
#5) activation
#6) poolSize
#7) Dropout  
#8) dense layer output size   
#9) optimizer
#10) no_epochs for training
    
    
space = {   'n_conv_layers': hp.uniform('n_conv_layers', 2, 10),
            'no_filters_conv': hp.uniform('no_filters_conv', 12, 32),
            'dim_conv_kernel': hp.uniform('dim_conv_kernel', 32, 64),
            'activation': hp.choice('activation',['tanh','sigmoid','relu']),
            'poolSize': hp.uniform('poolSize', .25,.5),                      
            'dropout': hp.uniform('dropout', .25,.5),
            'outputDense': hp.uniform('outputDense', .25,.5),
            'optimizer': hp.choice('optimizer',['adadelta','adam','rmsprop','sgd']),
            'nb_epochs' : hp.uniform('nb_epochs', 40, 70),
            'batch_size' : hp.uniform('batch_size', 28,128),           
            'n_classes': n_classes,
        }


def gen_model(params):
    #All conv. filters. of square size
    #conv param: array of size (n_conv,2),(:,0) - n_filters, (:,1) - kernel_dim
    #dense param: array of size (n_dense,1), (:,0) - neurons in dense layers
    
    model = Sequential()
    
    for i in range(params['n_conv_layers']):
        if i != 0:            
            model.add(Conv2D(params['no_filters_conv'],(params['dim_conv_kernel'],\
                             params['dim_conv_kernel']), activation=params['activation']))
        else:
            model.add(Conv2D(params['no_filters_conv'],(params['dim_conv_kernel'],\
                             params['dim_conv_kernel']), input_shape = in_shape, activation='relu',data_format="channels_last"))
        model.add(MaxPooling2D(pool_size=(params['poolSize'], params['poolSize']), \
                                   strides=None, padding='valid', data_format=None))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
        model.add(Dense(params['outputDense'], activation=params['activation']))
        model.add(Dropout(0.5))
        model.add(Dense(params['n_classes'], activation=params['activation']))
        
    def getModel():
        #build_fn should construct, compile and return a Keras model, which will then be used to fit/predict
        model.compile(loss='mean_squared_error', optimizer=params['optimizer']) #, metrics='mean_squared_error'
        return model
    estimator = KerasRegressor(build_fn=getModel,epochs= int(params['nb_epochs']), batch_size= int(params['batch_size']) )
    results = cross_val_score(estimator, trainData, trainLabels, cv=tscv, fit_params={'callbacks': [reduce_lr]})
    mean_cv_score = results.mean()
    if math.isnan(mean_cv_score):
        mean_cv_score = 100000
    print('Mean cross-validation score', mean_cv_score)
    sys.stdout.flush() 
    return {'loss': mean_cv_score, 'status': STATUS_OK}    
    
    

print(1)

trials = Trials()
best = fmin(gen_model, space, algo=tpe.suggest, max_evals=15, trials=trials)
print('best: ')
print(best)
np.save('CnnModelHyperparameters.npy',best)

