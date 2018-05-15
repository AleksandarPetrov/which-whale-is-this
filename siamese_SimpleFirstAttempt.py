import numpy as np
from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten, MaxPooling2D
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD, Adam
from keras.losses import binary_crossentropy
import numpy.random as rng


def W_init(shape, name=None):
    """Initialize weights as in paper"""
    values = rng.normal(loc=0, scale=1e-2, size=shape)
    return K.variable(values, name=name)


def b_init(shape, name=None):
    """Initialize bias as in paper"""
    values = rng.normal(loc=0.5, scale=1e-2, size=shape)
    return K.variable(values, name=name)


def basicSiameseGenerator():
    input_shape = (500, 250, 1)

    test_input = Input(input_shape)  # this is where we feed the image we want to test if is the same as the known image
    known_input = Input(input_shape)  # this is where we feed the known image
    # It doesn't matter which is which, the network is completely symmetrical

    # BUILDING THE LEGS OF THE SIAMESE NETWORK
    convnet = Sequential()

    convnet.add(Conv2D(filters=32,
                       kernel_size=(16, 16),
                       activation='relu',
                       input_shape=input_shape,
                       kernel_initializer=W_init,
                       kernel_regularizer=l2(2e-4),
                       use_bias=False)
                )
    convnet.add(MaxPooling2D())

    convnet.add(Conv2D(filters=64,
                       kernel_size=(13, 13),
                       activation='relu',
                       kernel_regularizer=l2(2e-4),
                       kernel_initializer=W_init,
                       bias_initializer=b_init)
                )
    convnet.add(MaxPooling2D())

    convnet.add(Conv2D(filters=128,
                       kernel_size=(10, 10),
                       activation='relu',
                       kernel_regularizer=l2(2e-4),
                       kernel_initializer=W_init,
                       use_bias=False)
                )
    convnet.add(MaxPooling2D())

    convnet.add(Conv2D(filters=128,
                       kernel_size=(7, 7),
                       activation='relu',
                       kernel_initializer=W_init,
                       kernel_regularizer=l2(2e-4),
                       bias_initializer=b_init)
                )
    convnet.add(MaxPooling2D())

    convnet.add(Conv2D(filters=256,
                       kernel_size=(4, 4),
                       activation='relu',
                       kernel_initializer=W_init,
                       kernel_regularizer=l2(2e-4),
                       bias_initializer=b_init,
                       use_bias=False)
                )

    convnet.add(Flatten())

    convnet.add(Dense(units=4096,
                      activation="sigmoid",
                      kernel_regularizer=l2(1e-3),
                      kernel_initializer=W_init,
                      bias_initializer=b_init)
                )

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
                        optimizer=optimizer)

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
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            img = np.load('data/train_npy/' + ID + '.npy')
            img = img[:, :, np.newaxis]
            X[i,] = img
            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


if __name__ == "__main__":
    testNet = basicSiameseGenerator()
    testNet.summary()
