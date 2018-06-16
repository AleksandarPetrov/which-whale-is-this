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
import random
from random import randint, uniform
import pickle
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
# from gen_imageName_dict import gen_imageName_dict
from collections import Counter
import matplotlib.pyplot as plt
from data_aug import img_data_aug_array, aug_para
from dataGenerator import SiameseDataGenerator
from architecture import basicSiameseGenerator
import h5py
import datetime
import time
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import csv
from collections import Counter

from collections import *


########### BOOLS TO FILL IN ##############
checking_effect_amount_data = True
augment_data = True
N = 2 # number of images to augment
###########################################
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H:%M:%S')

N_EPOCHS = 25

# Some useful directories
parent_dir = './../../DATA/'#sys.argv[1]
test_dir = os.path.join(parent_dir, 'test_npy')
train_dir = os.path.join(parent_dir, 'train_npy')
labels_dir = os.path.join(parent_dir, 'train.csv')

dataset = h5py.File(os.path.join(parent_dir, 'tr_gr_64.h5'), 'r')
X_dataset = np.array(dataset['x'])
y_labels = np.array(dataset['y'])
y_labels = y_labels.astype('str')
# print(set(y_labels))
# NORMALIZE
X_dataset_original = X_dataset
X_dataset_flattened = X_dataset
X_dataset_flattened.flatten()
average_value = np.mean(X_dataset_flattened)
std_dev = np.std(X_dataset_flattened)
# print(average_value)

X_dataset = (X_dataset - np.ones(np.shape(X_dataset))*average_value)/std_dev
# print(X_dataset.shape)
# Is it okay if aug is done after normalization?

if augment_data:
    # X_dataset_augmented = X_dataset
    # y_labels_augmented = y_labels
    file_path1 = os.path.join(parent_dir, 'augmented_to_' + str(N)+'_tr_gr_64.hdf5')
    file_path2 = os.path.join(parent_dir, 'augmented_to_' + str(N)+'_val_gr_64.hdf5')

    whale_counts = Counter(y_labels)
    extra_images = []
    extra_labels = []

    l = len(set(y_labels))
    for i in range(len(set(y_labels))):
        current_label = y_labels[i]
        indexes = np.where(y_labels == current_label)
        n_indexes = len(indexes)

        if n_indexes <= N:
            n_images_to_augment = N - n_indexes
            indexes_for_augmentation = indexes
            a = indexes
            random.shuffle(a)
        else:
            continue # no augmentation if already a lot of images

        for i in range(n_images_to_augment):
            augParam = aug_para(rot_deg=uniform(20, 60),
                                width=uniform(0, 1),
                                height=uniform(0, 1),
                                shear=uniform(10, 30),
                                zoom=uniform(0, 1))
            if i < n_indexes:
                image_to_augment = X_dataset[indexes_for_augmentation[i]]
                image_augmented = img_data_aug_array(augParam, image_to_augment[0, :])
                # plt.imshow(image_to_augment, interpolation='nearest')
                # plt.show()
                # print(np.shape(image_augmented))

            else:
                # print(i, n_indexes)
                idx = min(i - n_indexes,len(a)-1) # so we don't exceed idx in a
                # print(a, idx, len(a))
                image_to_augment = X_dataset[a[idx]]
                print(image_to_augment)
                image_augmented = img_data_aug_array(augParam, image_to_augment[0, :])

            X_dataset = np.vstack((X_dataset, np.reshape(image_augmented, (1, 64, 64))))
            # print('this',y_labels_augmented.shape)
            y_labels = np.hstack((y_labels, current_label))
            # print('here', X_dataset_augmented.shape)
else:

    file_path1 = os.path.join(parent_dir, 'tr_gr_64.hdf5')
    file_path2 = os.path.join(parent_dir, 'val_gr_64.hdf5')

bool1 = os.path.exists(file_path1)
bool2 = os.path.exists(file_path2)

if bool1 == False:
    whale_ids = set(y_labels)
    random.shuffle(list(whale_ids))
    whale_ids_list = list(whale_ids)
    whales_training = whale_ids_list[0::2]
    idx_training_whales = [i for i, x in enumerate(y_labels) if any(thing in x for thing in whales_training)]
    X_dataset_training = X_dataset[idx_training_whales]
    y_labels_training = y_labels[idx_training_whales]
    print('here')
    f = h5py.File(file_path1,"w")
    f.create_dataset('training_data', data = X_dataset_training)
    asciiList = [n.encode("ascii", "ignore") for n in y_labels_training]
    f.create_dataset('training_data_labels', (len(asciiList),1),'S10', asciiList)
    f.close()
else:
    dataset = h5py.File(file_path1, 'r')
    X_dataset_training = np.array(dataset['training_data'])
    y_labels_training = np.array(dataset['training_data_labels'])
    y_labels_training = y_labels_training.astype('str')


if bool2 == False:
    whale_ids = set(y_labels)
    random.shuffle(list(whale_ids))

    whale_ids_list = list(whale_ids)
    whales_validation = whale_ids_list[1::2]
    print(whales_validation)
    idx_validation_whales = [i for i, x in enumerate(y_labels) if any(thing in x for thing in whales_validation)]
    X_dataset_validation = X_dataset[idx_validation_whales]
    y_labels_validation = y_labels[idx_validation_whales]

    print('there')
    f = h5py.File(os.path.join(parent_dir,'val_gr_64.hdf5'),"w")
    f.create_dataset('validation_data', data = X_dataset_validation)
    asciiList = [n.encode("ascii", "ignore") for n in y_labels_validation]
    f.create_dataset('validation_data_labels', (len(asciiList),1),'S10', asciiList)

    f.close()

else:
    dataset = h5py.File(file_path2, 'r')
    X_dataset_validation = np.array(dataset['validation_data'])
    y_labels_validation = np.array(dataset['validation_data_labels'])
    y_labels_validation = y_labels_validation.astype('str')

print(X_dataset_training[1])



training_generator = SiameseDataGenerator(parent_dir,X_dataset_training,y_labels_training, stochastic = True)


# Saving callback
filepath = os.path.join(parent_dir, st+'weights.Siamese.best.binary_acc.tr.hdf5')
checkpoint = ModelCheckpoint(filepath, monitor='val_binary_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# Model generation

# load the weights which give best results for far
# model = load_model(filepath)
model = basicSiameseGenerator(parent_dir = parent_dir,trainable = True)



history = model.fit_generator(generator = SiameseDataGenerator(parent_dir,X_dataset_training,y_labels_training, stochastic = True), # change the input datasets to be based on certain number of whales
                              validation_data = SiameseDataGenerator(parent_dir,X_dataset_validation,y_labels_validation, stochastic = False), # change the input datasets to be based on certain number of whales
                              use_multiprocessing=True,
                              epochs= N_EPOCHS,
                              callbacks = callbacks_list,
                              verbose=2)

d = {'loss': history.history['loss'], 'acc': history.history['binary_accuracy'], 'val_loss':history.history['val_loss'],'val_acc': history.history['val_binary_accuracy']}
df = pd.DataFrame.from_dict(data=d)

if checking_effect_amount_data:
    sub_path = os.path.join(parent_dir,'effect_amount_data/')
else:
    sub_path = parent_dir
filepath_history = os.path.join(sub_path,'history_'+st+'.csv')

pd.DataFrame.to_csv(df, filepath_history)

# loss = np.array(history_file['loss'])
# acc = np.array(history_file['acc'])
# val_loss = np.array(history_file['val_loss'])
# val_acc = np.array(history_file['val_acc'])






# Save final
model.save(os.path.join(parent_dir, st+'weights.Siamese.final.hdf5'))

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


