# -*- coding: utf-8 -*-
"""
Created on Tue May  1 09:36:28 2018

@author: Kuba
"""
from keras.models import Sequential
from keras.layers import Dense,Conv2D
import numpy as np

def model(conv_param,dense_param,in_shape):
    #All conv. filters. of square size
    #conv param: array of size (n_conv,2),(:,0) - n_filters, (:,1) - kernel_dim
    #dense param: array of size (n_dense,1), (:,0) - neurons in dense layers
    
    n_conv,n_dense = conv_param.shape[0], dense_param.shape[0]
    
    model = Sequential()
    
    for i in range(n_conv):
        if i != 0:            
            model.add(Conv2D(conv_param[i,0],(conv_param[i,1],conv_param[i,1]), activation='relu'))
        else:
            model.add(Conv2D(conv_param[i,0],(conv_param[i,1],conv_param[i,1]), input_shape = in_shape, activation='relu',data_format="channels_last"))
            
    for i in range(n_dense):
        model.add(Dense(dense_param[i,0], activation='relu'))
        
    model.add(Dense(2000))
    model.summary()
    return model
    
    
    
'''TESTING BELOW'''    
    
    
conv_param = np.zeros((2,2))
dense_param = np.zeros((1,1))
in_shape = (500,250,1)
        
conv_param[0,0], conv_param[0,1] = 4, 30
conv_param[1,0], conv_param[1,1] = 16, 10     
dense_param[0] = 1000


conv_param = conv_param.astype(int)
dense_param = dense_param.astype(int)


model = model(conv_param,dense_param,in_shape)



    
     
    


