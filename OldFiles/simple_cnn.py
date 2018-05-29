# -*- coding: utf-8 -*-
"""
Created on Tue May  1 09:36:28 2018

@author: Kuba
"""
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten,MaxPooling2D

def gen_model(conv_param,dense_param,in_shape,n_classes):
    #All conv. filters. of square size
    #conv param: array of size (n_conv,2),(:,0) - n_filters, (:,1) - kernel_dim
    #dense param: array of size (n_dense,1), (:,0) - neurons in dense layers
    conv_param = conv_param.astype(int)
    dense_param = dense_param.astype(int)

    n_conv,n_dense = conv_param.shape[0], dense_param.shape[0]
    
    model = Sequential()
    
    for i in range(n_conv):
        if i != 0:            
            model.add(Conv2D(conv_param[i,0],(conv_param[i,1],conv_param[i,1]), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
        else:
            model.add(Conv2D(conv_param[i,0],(conv_param[i,1],conv_param[i,1]), input_shape = in_shape, activation='relu',data_format="channels_last"))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
    
    model.add(Flatten())
    if n_dense != 0:
        for i in range(n_dense):
            model.add(Dense(dense_param[i,0], activation='relu'))
        
    model.add(Dense(n_classes, activation='softmax')) # added softmax here, check if suitable?
    model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return model
    
    
    




    
     
    


