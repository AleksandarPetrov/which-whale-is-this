# -*- coding: utf-8 -*-
"""
Created on Sun May 20 13:56:01 2018

@author: Kuba
"""


import matplotlib.pyplot as plt
from simple_cnn import gen_model
from keras.datasets import mnist
import numpy as np
from keras.utils import np_utils
import time

'''Defining some functions'''
def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()
    
    
    
def accuracy(test_x, test_y, model):
    result = model.predict(test_x)
    predicted_class = np.argmax(result, axis=1)
    true_class = np.argmax(test_y, axis=1)
    num_correct = np.sum(predicted_class == true_class) 
    accuracy = float(num_correct)/result.shape[0]
    return (accuracy * 100)    
    

'''Importing and processing the data'''
(train_features, train_labels), (test_features, test_labels) = mnist.load_data()
_, img_rows, img_cols =  train_features.shape
num_classes = len(np.unique(train_labels))
num_input_nodes = img_rows*img_cols


train_features = train_features.reshape(train_features.shape[0], 1, img_rows, img_cols).astype('float32')
test_features = test_features.reshape(test_features.shape[0], 1, img_rows, img_cols).astype('float32')
train_features /= 255
test_features /= 255
# convert class labels to binary class labels
#train_labels_bin = np_utils.to_categorical(train_labels, num_classes)
#test_labels_bin = np_utils.to_categorical(test_labels, num_classes)

'''Generate the model'''
dataformat = "channels_first"
in_shape = (1, 28, 28)
conv_param = np.zeros((1,2)) # what exactly are those?
dense_param = np.zeros((1,1)) 
conv_param[0,0], conv_param[0,1] = 32, 5
n_classes = 10

samples_per_class = [1,10,100,300,600,900]

class_indices = [0]*10
for i in range(n_classes):
    class_indices[i] = np.where(train_labels == i)

sample_model_info = []
for samples in samples_per_class:
    mask = np.arange(0,samples)
    indices = np.array([])
    
    for i in range(n_classes):
        class_indices_i = class_indices[i][0]
        masked_indices = class_indices_i[mask]
        indices = np.hstack((indices,masked_indices))
    indices = indices.astype(int)
    train_feature_sample = train_features[indices]
    train_labels_sample = np_utils.to_categorical(train_labels[indices], num_classes)
    test_labels_bin = np_utils.to_categorical(test_labels, num_classes)
    
    modelz = gen_model(conv_param,dense_param,in_shape,n_classes,dataformat)
    
    modelz_info = modelz.fit(train_feature_sample, train_labels_sample, batch_size=128, \
                             nb_epoch=20, verbose=1, validation_split=0.2)

    score = modelz.evaluate(test_features, test_labels_bin, batch_size=128)
    sample_model_info.append(score)
    
