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


TRAIN_TOP_N_WHALES = True
N = 20
N_EPOCHS = 15
LOAD_WEIGHTS = True

def W_init(shape, name=None):
    """Initialize weights as in paper"""
    values = rng.normal(loc=0, scale=1e-2, size=shape)
    return K.variable(values, name=name)


def b_init(shape, name=None):
    """Initialize bias as in paper"""
    values = rng.normal(loc=0.5, scale=1e-2, size=shape)
    return K.variable(values, name=name)


def basicSiameseGenerator():
    input_shape = (250, 500, 1)

    test_input = Input(input_shape)  # this is where we feed the image we want to test if is the same as the known image
    known_input = Input(input_shape)  # this is where we feed the known image
    # It doesn't matter which is which, the network is completely symmetrical

    # Load kaggle model
    kaggleModel = load_model(os.path.join(parent_dir, 'keras_model.h5'))
    kaggleModel.summary()

    # BUILDING THE LEGS OF THE SIAMESE NETWORK
    convnet = Sequential()

    convnet.add(Conv2D(filters=16,
                       kernel_size=(16, 16),
                       activation='relu',
                       input_shape=input_shape,
                       kernel_initializer=W_init,
                       kernel_regularizer=l2(2e-4),
                       use_bias=True)
                )
    convnet.add(MaxPooling2D())

    convnet.add(Conv2D(filters=32,
                       kernel_size=(13, 13),
                       activation='relu',
                       kernel_regularizer=l2(2e-4),
                       kernel_initializer=W_init,
                       bias_initializer=b_init,
                       use_bias=False)
                )
    convnet.add(MaxPooling2D())

    convnet.add(Conv2D(filters=32,
                       kernel_size=(10, 10),
                       activation='relu',
                       kernel_regularizer=l2(2e-4),
                       kernel_initializer=W_init,
                       use_bias=True)
                )
    convnet.add(MaxPooling2D())

    convnet.add(Conv2D(filters=64,
                       kernel_size=(7, 7),
                       activation='relu',
                       kernel_initializer=W_init,
                       kernel_regularizer=l2(2e-4),
                       bias_initializer=b_init,
                       use_bias=False)
                )
    convnet.add(MaxPooling2D())

    convnet.add(Conv2D(filters=64,
                       kernel_size=(4, 4),
                       activation='relu',
                       kernel_initializer=W_init,
                       kernel_regularizer=l2(2e-4),
                       bias_initializer=b_init,
                       use_bias=True)
                )

    convnet.add(Flatten())

    convnet.add(Dense(units=4096,
                      activation="sigmoid",
                      kernel_regularizer=l2(1e-3),
                      kernel_initializer=W_init,
                      bias_initializer=b_init)
                )
    convnet.summary()

    # Add the two inputs to the leg (passing the two inputs through the same network is effectively the same as having
    # two legs with shared weights
    encoded_test = convnet(test_input)
    encoded_known = convnet(known_input)

    # Get the absolute difference between the two vectors
    L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_test, encoded_known])

    # Add the final layer that connects all of the  distances on the previous layer to the single output
    prediction = Dense(units=1,
                       activation='sigmoid',
                       bias_initializer=b_init
                       )(L1_distance)

    siamese_net = Model(inputs=[test_input, known_input], outputs=prediction)

    optimizer = Adam(0.00006)
    siamese_net.compile(loss="binary_crossentropy",
                        optimizer=optimizer,
                        metrics=['accuracy'])

    return siamese_net


class SiameseDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, labels, batch_size=32, dim=(32, 32), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.shown = 0

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X1 = np.empty((self.batch_size, *self.dim, self.n_channels))
        X2 = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample 1
            img = np.load(os.path.join(parent_dir, 'train_npy/' + ID + '.npy'))
            img = img[:, :, np.newaxis]
            X1[i,] = img
            X1_label = self.labels[ID]

            listOfPicturesOfSameWhale = [k for k in list(self.labels.keys()) if self.labels[k] == X1_label]
            # For the second one take one from the same class is i is even, otherwise one with a different class,
            # also checks if the class is not new_whale and if there is at least one other picture of the same whale
            ###DATA AUGMENTATION SHOULD BE PUT HERE AND A LOT OF THINGS ADJUSTED
            if i%2==0 and X1_label!=0:
                X2_ID = np.random.choice(listOfPicturesOfSameWhale, 1)[0]
                img = np.load(os.path.join(parent_dir, 'train_npy/' + X2_ID + '.npy'))
                # Augment:
                augParam = aug_para(rot_deg=min(10, max(-10, np.random.normal(loc=0.0, scale=5))),
                                    width=min(0.1, max(-0.1, np.random.normal(loc=0.0, scale=0.05))),
                                    height=min(0.1, max(-0.1, np.random.normal(loc=0.0, scale=0.05))),
                                    shear=min(0.1, max(-0.1, np.random.normal(loc=0.0, scale=0.05))),
                                    zoom=min(0.1, max(-0.1, np.random.normal(loc=0.0, scale=0.05))))
                img = img_data_aug_array(augParam, img)
                #if(self.shown<30):
                #    plt.imshow(img)
                #    plt.savefig('foo'+str(self.shown)+'.png')
                #    self.shown=self.shown+1
                img = img[:, :, np.newaxis]
                X2[i,] = img
                # Store class
                y[i] = 1
            else:
                listOfPicturesOfDifferentWhales = [k for k in list(self.labels.keys()) if (self.labels[k] != X1_label or self.labels[k]!=0) and k != ID]
                X2_ID = np.random.choice(listOfPicturesOfDifferentWhales, 1)[0]
                img = np.load(os.path.join(parent_dir, 'train_npy/' + X2_ID + '.npy'))
                # Augment:
                augParam = aug_para(rot_deg=min(10, max(-10, np.random.normal(loc=0.0, scale=5))),
                                    width=min(0.1, max(-0.1, np.random.normal(loc=0.0, scale=0.05))),
                                    height=min(0.1, max(-0.1, np.random.normal(loc=0.0, scale=0.05))),
                                    shear=min(0.1, max(-0.1, np.random.normal(loc=0.0, scale=0.05))),
                                    zoom=min(0.1, max(-0.1, np.random.normal(loc=0.0, scale=0.05))))
                img = img_data_aug_array(augParam, img)
                #if (self.shown < 30):
                #    plt.imshow(img)
                #    plt.savefig('foo'+str(self.shown)+'.png')
                #    self.shown=self.shown+1
                img = img[:, :, np.newaxis]
                X2[i,] = img
                # Store class
                y[i] = 0



        return [X1, X2], y


if __name__ == "__main__":

    # Some useful directories
    parent_dir = '../../DATA/'#sys.argv[1]
    test_dir = os.path.join(parent_dir, 'test_npy')
    train_dir = os.path.join(parent_dir, 'train_npy')
    labels_dir = os.path.join(parent_dir, 'train.csv')

    # Reading of labels and corresponding image names
    classes = pd.read_csv(labels_dir)
    list_ids = list(classes.Id)
    file_names = list(classes.Image)
    n_files = len(file_names)
    file_names = [file[:-4] for file in file_names]

    # Label encoder, changes the label names to integers for use in generator
    le = LabelEncoder()
    le.fit(list_ids)
    n_classes = len(le.classes_)
    labels_int = list(le.fit_transform(list_ids))

    if TRAIN_TOP_N_WHALES:
        # Count
        whale_counts = Counter(list_ids)
        whale_counts_most_data = whale_counts.most_common(N)  # whale IDs for n most common whales
        number_images = sum([element[1] for element in whale_counts_most_data])
        whale_IDs_most_data = [element[0] for element in whale_counts_most_data]  # get the whale_Ids only

        # get indexes of these whales in the entire training dataset
        list_ids_arr = np.array(list_ids)
        idx_whale_IDs_most_data = []
        for i in range(len(whale_IDs_most_data)):
            indexes = np.where(list_ids_arr == whale_IDs_most_data[i])[0]
            idx_whale_IDs_most_data.extend(list(indexes))
        idx_whale_IDs_most_data.sort()

        # find image names corresponding to these whales
        file_names_sub = [file_names[i] for i in idx_whale_IDs_most_data]
        labels_int_sub = [labels_int[i] for i in idx_whale_IDs_most_data]
        # should this be refit??
        labels_int_sub_refit = list(le.fit_transform(labels_int_sub))  # refit it to the N classes

        # labels_sub = [list_ids[i] for i in idx_whale_IDs_most_data]
        # labels_sub == whale_IDs_most_data

        # Parameters for Generator
        partition = gen_imageName_dict(test_dir, train_dir, 0.2, file_names_sub)
        imageName_ID_dict = dict(zip(file_names_sub, labels_int_sub_refit))
        n_classes = N
    else:
        # Parameters for Generator
        partition = gen_imageName_dict(test_dir, train_dir, 0.2)
        imageName_ID_dict = dict(zip(file_names, labels_int))
        n_classes = len(le.classes_)

    # Parameters for Generator [OLD, BEFORE THE TRAIN TOP WHALES]
    #partition = gen_imageName_dict(test_dir, train_dir, validation_fraction=0.2)
    #labels = dict(zip(file_names, labels_int))

    params = {'dim': (250, 500),
              'batch_size': 32,
              'n_classes': n_classes,
              'n_channels': 1,
              'shuffle': True}


    training_generator = SiameseDataGenerator(list_IDs = partition['train'],
                                              labels = imageName_ID_dict,
                                              **params)
    validation_generator = SiameseDataGenerator(list_IDs = partition['validation'],
                                                labels = imageName_ID_dict,
                                                **params)

    # Saving callback
    filepath = os.path.join(parent_dir, 'weights.best.basicSiamese.hdf5')
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    # Model generation
    model = basicSiameseGenerator()
    model.summary()
    history = model.fit_generator(generator=training_generator,
                                  validation_data = validation_generator,
                                  use_multiprocessing=True,
                                  epochs= N_EPOCHS,
                                  verbose=1,
                                  callbacks=callbacks_list)

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


