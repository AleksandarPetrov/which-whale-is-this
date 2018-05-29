# -*- coding: utf-8 -*-
"""
Created on Sun May 27 16:52:20 2018

@author: Kuba
"""
import numpy as np
import numpy.random as rng
import pandas as pd
from glob import glob
from PIL import Image
import matplotlib.pylab as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

import random
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Dropout, Input, Lambda, Flatten,MaxPooling2D
from keras.optimizers import RMSprop
from keras import backend as K



from subprocess import check_output


class LabelOneHotEncoder():
    def __init__(self):
        self.ohe = OneHotEncoder()
        self.le = LabelEncoder()
    def fit_transform(self, x):
        features = self.le.fit_transform( x)
        return self.ohe.fit_transform( features.reshape(-1,1))
    def transform( self, x):
        return self.ohe.transform( self.la.transform( x.reshape(-1,1)))
    def inverse_tranform( self, x):
        return self.le.inverse_transform( self.ohe.inverse_tranform( x))
    def inverse_labels( self, x):
        return self.le.inverse_transform( x)
    
def compute_distances_no_loops(X, Y):
    dists = -2 * np.dot(X, Y) + np.sum(X**2, axis=1) + np.sum(Y**2, axis=1)[:, np.newaxis]
    return dists
      
    
def ImportImage( filename,SIZE):
    #image are imported with a resizing and a black and white conversion
    img = Image.open(filename).convert("LA").resize( (SIZE,SIZE))
    return np.array(img)[:,:,0]

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)




