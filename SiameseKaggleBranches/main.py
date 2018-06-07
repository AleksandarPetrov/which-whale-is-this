import numpy as np
import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten, MaxPooling2D
from keras.models import Model, Sequential, load_model
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD, Adam
from keras.losses import binary_crossentropy
import os

from sklearn.preprocessing import LabelEncoder
import pandas as pd
from keras.callbacks import ModelCheckpoint
# from gen_imageName_dict import gen_imageName_dict
from collections import Counter
import matplotlib.pyplot as plt
from data_aug import img_data_aug_array, aug_para
from dataGenerator import SiameseDataGenerator
from architecture import basicSiameseGenerator
import h5py

N_EPOCHS = 50

# Some useful directories
parent_dir = './../../DATA/'#sys.argv[1]
test_dir = os.path.join(parent_dir, 'test_npy')
train_dir = os.path.join(parent_dir, 'train_npy')
labels_dir = os.path.join(parent_dir, 'train.csv')

dataset = h5py.File(os.path.join(parent_dir, 'tr_gr_64.h5'), 'r')
X_dataset = np.array(dataset['x'])
y_labels = np.array(dataset['y'])
y_labels = y_labels.astype('str')

training_generator = SiameseDataGenerator(parent_dir,X_dataset,y_labels, stochastic = True)

# Saving callback
filepath = os.path.join(parent_dir, 'weights.Siamese.best.binary_accuracy.training.hdf5')
checkpoint = ModelCheckpoint(filepath, monitor='val_binary_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# Model generation
model = basicSiameseGenerator(parent_dir = parent_dir,
                              trainable = True)
history = model.fit_generator(generator = SiameseDataGenerator(parent_dir,X_dataset,y_labels, stochastic = True),
                              validation_data = SiameseDataGenerator(parent_dir,X_dataset,y_labels, stochastic = False),
                              use_multiprocessing=True,
                              epochs= N_EPOCHS,
                              callbacks = callbacks_list,
                              verbose=2)


# Save final
model.save(os.path.join(parent_dir, 'weights.Siamese.final.hdf5'))

# Plot the training and validation loss and accuracies
try:
    plt.figure()
    plt.plot(history.history['binary_accuracy'])
    plt.plot(history.history['val_binary_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # "Loss"
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
except:
    print("Plotting couldn't be done. Probably running on Cloud without graphical interface.")


