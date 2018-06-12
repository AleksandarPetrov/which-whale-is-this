from keras.models import Sequential, Model
from keras.initializers import TruncatedNormal
from dataGenerator import SiameseDataGenerator
from keras.layers import Dense,Conv2D,MaxPooling2D, Input, Lambda, Flatten
from keras import backend as K
from keras.optimizers import Adam
import numpy as np
import h5py
import os

def modelEvaluator(params, trainEpochs = 25, parent_dir = './../../DATA/', iterations = 5):
    print("yo")
    networkPerformanceList = []

    for i in range(iterations):
    # BUILD THE COMPLETE NETWORK

        input_shape = (64, 64, 1)

        test_input = Input(input_shape)  # this is where we feed the image we want to test if is the same as the known image
        known_input = Input(input_shape)

        legInstance = legNetwork(params)
        test_leg = legInstance(test_input)
        known_leg = legInstance(known_input)

        L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
        L1_distance = L1_layer([test_leg, known_leg])

        # Add the final layer that connects all of the  distances on the previous layer to the single output
        prediction = Dense(units=2,
                           activation='softmax',
                           name='predictionLayer')(L1_distance)
        legInstance.summary()
        trialNet = Model(input=[test_input, known_input], output=prediction)

        optimizer = Adam(0.00008)
        trialNet.compile(loss="binary_crossentropy",
                            optimizer=optimizer,
                            metrics=['binary_accuracy'])


        trialNetPerformance = checkPerformance(trialNet, trainEpochs, parent_dir)

        networkPerformanceList.append(trialNetPerformance)

    trialNetPerformanceAvg = np.average(networkPerformanceList)

    trialNetSize = trialNet.count_params()
    sizePenalty = 0
    if trialNetSize > 10000000:
        sizePenalty = sizePenalty + 0.05
    if trialNetSize > 25000000:
        sizePenalty = sizePenalty + 0.05
    if trialNetSize > 50000000:
        sizePenalty = sizePenalty + 0.1

    return trialNetPerformanceAvg - sizePenalty

def legNetwork(params):

    model = Sequential(name='siameseLeg')

    input_shape = (64, 64, 1)

    for i in range(int(params['n_conv_layers'])):

        numberOfFilters = params['initial_number_filters'] * 2**i
        kernelSize = params['initial_kernel_size'] - (i * 2)

        if numberOfFilters > 256:
            numberOfFilters  = 256
        if kernelSize < 3:
            kernelSize = 3

        if i != 0:
            model.add(Conv2D(numberOfFilters, (kernelSize, kernelSize),
                             activation = 'relu',
                             kernel_initializer='random_uniform',
                             #name="legConvLayer_" + str(i + 1),
                             trainable = False
                             )
                      )
        else:
            model.add(Conv2D(numberOfFilters, (kernelSize, kernelSize),
                             activation='relu',
                             input_shape=input_shape,
                             kernel_initializer = 'random_uniform',
                             #name="legConvLayer_" + str(i + 1),
                             trainable=False
                             )
                      )

        model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))

    model.add(Flatten())
    model.add(Dense(1024,activation="sigmoid", name="legOutput"))
    return model

def checkPerformance(network, trainEpochs, parent_dir):

    dataset = h5py.File(os.path.join(parent_dir, 'tr_gr_64.h5'), 'r')
    X_dataset = np.array(dataset['x'])
    y_labels = np.array(dataset['y'])
    y_labels = y_labels.astype('str')

    randomizedIndices = np.arange(len(y_labels))
    np.random.shuffle(randomizedIndices)


    training_generator = SiameseDataGenerator(parent_dir = parent_dir,
                                              X_dataset = X_dataset[randomizedIndices[0:int(0.8*len(y_labels))], :, :],
                                              y_labels = y_labels[randomizedIndices[0:int(0.8*len(y_labels))]],
                                              batch_size=32,
                                              dim=(64, 64),
                                              n_channels=1,
                                              shuffle=True,
                                              stochastic = True)
    validation_generator = SiameseDataGenerator(parent_dir = parent_dir,
                                              X_dataset = X_dataset[randomizedIndices[int(0.8*len(y_labels)):], :, :],
                                              y_labels = y_labels[randomizedIndices[int(0.8*len(y_labels)):]],
                                              batch_size=32,
                                              dim=(64, 64),
                                              n_channels=1,
                                              shuffle=True,
                                              stochastic = False)

    history = network.fit_generator(generator=training_generator,
                                    validation_data=validation_generator,
                                    use_multiprocessing=True,
                                    epochs=trainEpochs,
                                    verbose=0)

    return max(history.history['val_binary_accuracy'])