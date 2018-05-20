## Keras ##
from keras.models import Sequential # Model
from keras.backend import categorical_crossentropy, constant
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten,MaxPooling2D
## Hyperopt ##
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from gen_imageName_dict import gen_imageName_dict
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from collections import Counter
from keras.layers import Dense,Conv2D,MaxPooling2D, Dropout, Flatten # Layers
from keras.utils.np_utils import to_categorical
import tensorflow as tf



from keras.initializers import RandomUniform
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import cross_val_score

###########################################################
################# Other data ##############################
###########################################################
# # Some useful directories
# trainData = np.load('/home/isabelle/Documents/Education/Masters/Fourth_year/Q4/Deep_learning/data/trainData.npy')
# trainLabels = np.load('/home/isabelle/Documents/Education/Masters/Fourth_year/Q4/Deep_learning/data/trainLabels.npy')
TRAIN_TOP_N_WHALES = True
N = 3
N_EVALUATIONS = 2


in_shape = (250,500,1)

# Some useful directories
test_dir = '../DATA/test_npy'
train_dir = '../DATA/train_npy'
labels_dir = '../DATA/train.csv'

# Reading of labels and corresponding image names
classes = pd.read_csv(labels_dir)
list_ids = list(classes.Id) # whale ids not image ids
file_names = list(classes.Image)
n_files = len(file_names)
file_names = [file[:-4] for file in file_names]

# Label encoder, changes the label names to integers for use in generator
le = LabelEncoder()
le.fit(list_ids) # whale ids not image ids

labels_int = list(le.fit_transform(list_ids))

if TRAIN_TOP_N_WHALES:
    # Count
    whale_counts = Counter(list_ids)
    whale_counts_most_data = whale_counts.most_common(N) # whale IDs for n most common whales
    number_images = sum([element[1] for element in whale_counts_most_data])
    whale_IDs_most_data = [element[0] for element in whale_counts_most_data]# get the whale_Ids only

    # get indexes of these whales in the entire training dataset
    list_ids_arr = np.array(list_ids)
    idx_whale_IDs_most_data = []
    for i in range(len(whale_IDs_most_data)):
        indexes = np.where(list_ids_arr == whale_IDs_most_data[i])[0]
        idx_whale_IDs_most_data.extend(list(indexes))

    # find image names corresponding to these whales
    file_names_sub = [file_names[i] for i in idx_whale_IDs_most_data]
    labels_int_sub = [labels_int[i] for i in idx_whale_IDs_most_data]
    # should this be refit??
    labels_int_sub_refit = list(le.fit_transform(labels_int_sub)) # refit it to the N classes


    #labels_sub = [list_ids[i] for i in idx_whale_IDs_most_data]
    #labels_sub == whale_IDs_most_data

    # Parameters for Generator
    partition = gen_imageName_dict(test_dir,train_dir, 0.2, file_names_sub)
    imageName_ID_dict = dict(zip(file_names_sub, labels_int_sub_refit))
    n_classes = N
else:
    # Parameters for Generator
    partition = gen_imageName_dict(test_dir,train_dir, 0.2)
    imageName_ID_dict = dict(zip(file_names,labels_int))
    n_classes = len(le.classes_)



architecture_space = {   'n_conv_layers': hp.uniform('n_conv_layers', 2, 4), # note: max number of MaxPooling2D layers is 7 since 2^8 is 256 and 250 is dimension
            'no_filters_conv': hp.uniform('no_filters_conv', 12, 32),
            'dim_conv_kernel': hp.uniform('dim_conv_kernel', 32, 64),
            'poolSize': hp.uniform('poolSize', .25 ,.5),
            'dropout': hp.uniform('dropout', .25 ,.5),
            'outputDense': hp.uniform('outputDense', 100, 150),
            'n_classes': n_classes,
            }


def gen_model(params):
    # All conv. filters. of square size
    # conv param: array of size (n_conv,2),(:,0) - n_filters, (:,1) - kernel_dim
    # dense param: array of size (n_dense,1), (:,0) - neurons in dense layers

    model = Sequential()
    # my_init = RandomUniform(minval=-0.05, maxval=0.05, seed=None)
    # print(my_init)
    for i in range(int(params['n_conv_layers'])):
        if i != 0:
            model.add(Conv2D(int(params['no_filters_conv']), (int(params['dim_conv_kernel']),
                                                              int(params['dim_conv_kernel'])), activation = 'relu'))
        else:
            model.add(Conv2D(int(params['no_filters_conv']), (int(params['dim_conv_kernel']),
                                                              int(params['dim_conv_kernel'])), input_shape=in_shape, kernel_initializer = 'random_uniform',
                             activation='relu', data_format="channels_last")) #
        model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
        # model.add(Dense(params['outputDense'])) #, kernel_initializer = 'random_uniform', bias_initializer='zeros', activation = 'relu'
    model.add(Dense(int(params['outputDense']), activation='relu'))
    model.add(Flatten())
    model.add(Dense(int(params['n_classes']), activation= 'softmax')) # for classification??



    # build_fn should construct, compile and return a Keras model, which will then be used to fit/predict
    model.compile(optimizer='rmsprop',
          loss='categorical_crossentropy',
          metrics=['accuracy'])  # compiling and with training parameters although not going to train + of course need it for predict_classes later on
    #  predict on the training data
    predictions = []

    categorical_labels = []

    ids = list(imageName_ID_dict.values())
    for i, ID in enumerate(imageName_ID_dict):
        # Store sample
        image = np.load('../DATA/train_npy/' + ID + '.npy')
        image = np.reshape(image, [1, 250, 500, 1])
        prediction = model.predict_classes(image)
        print(prediction)
        predictions.append(prediction)
        id = ids[i]
        print(id)

        categorical_label = to_categorical(id, num_classes = n_classes)
        categorical_labels.append(categorical_label)
        print(categorical_label)


    predictions_arr = np.array(predictions)
    categorical_labels_arr = np.array(categorical_labels)
    acc = np.sum(predictions_arr == categorical_labels_arr) / np.size(predictions_arr)
    # convert to tensors (as required by categorical_crossentropy)
    predictions_tensor = constant(predictions_arr)
    categorical_labels_tensor = constant(categorical_labels_arr)

    loss = categorical_crossentropy(predictions_tensor, categorical_labels_tensor)
    sess = tf.Session()
    result = sess.run(loss)
    return acc

# Note downgrade networkx, run this: pip3 install networkx==1.11
trials = Trials()
best = fmin(fn = gen_model, space = architecture_space, algo=tpe.suggest, max_evals= N_EVALUATIONS, trials=trials)
print('best: ')
print(best)
np.save('best_arch_params.npy', best)