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
from gen_imageName_dict import gen_imageName_dict
from collections import Counter
import matplotlib.pyplot as plt
from data_aug import img_data_aug_array, aug_para
from dataGenerator import SiameseDataGenerator
from architecture import basicSiameseGenerator
import h5py

TRAIN_TOP_N_WHALES = False
N = 20
N_EPOCHS = 15
LOAD_WEIGHTS = True



# Some useful directories
parent_dir = './DATA/'#sys.argv[1]
test_dir = os.path.join(parent_dir, 'test_npy')
train_dir = os.path.join(parent_dir, 'train_npy')
labels_dir = os.path.join(parent_dir, 'train.csv')

dataset = h5py.File('tr_gr_64.h5', 'r')
X_dataset = np.array(dataset['x'])
y_labels = np.array(dataset['y'])
y_labels = y_labels.astype('str')
# Reading of labels and corresponding image names
#    classes = pd.read_csv(labels_dir)
#    list_ids = list(classes.Id)
#    file_names = list(classes.Image)
#    n_files = len(file_names)
#    file_names = [file[:-4] for file in file_names]
#
#    # Label encoder, changes the label names to integers for use in generator
#    le = LabelEncoder()
#    le.fit(list_ids)
#    n_classes = len(le.classes_)
#    labels_int = list(le.fit_transform(list_ids))
#
#    if TRAIN_TOP_N_WHALES:
#        # Count
#        whale_counts = Counter(list_ids)
#        whale_counts_most_data = whale_counts.most_common(N)  # whale IDs for n most common whales
#        number_images = sum([element[1] for element in whale_counts_most_data])
#        whale_IDs_most_data = [element[0] for element in whale_counts_most_data]  # get the whale_Ids only
#
#        # get indexes of these whales in the entire training dataset
#        list_ids_arr = np.array(list_ids)
#        idx_whale_IDs_most_data = []
#        for i in range(len(whale_IDs_most_data)):
#            indexes = np.where(list_ids_arr == whale_IDs_most_data[i])[0]
#            idx_whale_IDs_most_data.extend(list(indexes))
#        idx_whale_IDs_most_data.sort()
#
#        # find image names corresponding to these whales
#        file_names_sub = [file_names[i] for i in idx_whale_IDs_most_data]
#        labels_int_sub = [labels_int[i] for i in idx_whale_IDs_most_data]
#        # should this be refit??
#        labels_int_sub_refit = list(le.fit_transform(labels_int_sub))  # refit it to the N classes
#
#        # labels_sub = [list_ids[i] for i in idx_whale_IDs_most_data]
#        # labels_sub == whale_IDs_most_data
#
#        # Parameters for Generator
#        partition = gen_imageName_dict(test_dir, train_dir, 0.2, file_names_sub)
#        imageName_ID_dict = dict(zip(file_names_sub, labels_int_sub_refit))
#        n_classes = N
#    else:
#        # Parameters for Generator
#        partition = gen_imageName_dict(test_dir, train_dir, 0.2)
#        imageName_ID_dict = dict(zip(file_names, labels_int))
#        n_classes = len(le.classes_)
#
#    # Parameters for Generator [OLD, BEFORE THE TRAIN TOP WHALES]
#    #partition = gen_imageName_dict(test_dir, train_dir, validation_fraction=0.2)
#    #labels = dict(zip(file_names, labels_int))
#
#    params = {'dim': (250, 500),
#              'batch_size': 32,
#              'n_classes': n_classes,
#              'n_channels': 1,
#              'shuffle': True}

training_generator = SiameseDataGenerator(parent_dir,X_dataset,y_labels)

# Saving callback
filepath = os.path.join(parent_dir, 'weights.best.basicSiamese.hdf5')
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# Model generation
model = basicSiameseGenerator(parent_dir = parent_dir)
history = model.fit_generator(generator=training_generator,use_multiprocessing=True,epochs= N_EPOCHS,verbose=1)

# Save final
model.save(os.path.join(parent_dir, 'weights.final.basicSiamese.hdf5'))

# Plot the training and validation loss and accuracies

#  "Accuracy"
plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
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


