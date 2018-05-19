## Keras ##
from keras.models import Sequential # Model
from keras.backend import categorical_crossentropy
## Hyperopt ##
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from gen_imageName_dict import gen_imageName_dict
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from collections import Counter
from keras.layers import Dense,Conv2D,MaxPooling2D # Layers


from keras.initializers import RandomUniform
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import cross_val_score

###########################################################
################# Other data ##############################
###########################################################
# # Some useful directories
# trainData = np.load('/home/isabelle/Documents/Education/Masters/Fourth_year/Q4/Deep_learning/data/trainData.npy')
# trainLabels = np.load('/home/isabelle/Documents/Education/Masters/Fourth_year/Q4/Deep_learning/data/trainLabels.npy')
TRAIN_TOP_N_WHALES = False
N = 20

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
n_classes = len(le.classes_)
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
    #labels_sub = [list_ids[i] for i in idx_whale_IDs_most_data]
    #labels_sub == whale_IDs_most_data

    # Parameters for Generator
    partition = gen_imageName_dict(test_dir,train_dir, 0.2, file_names_sub)
    imageName_ID_dict = dict(zip(file_names_sub, labels_int_sub))
    n_classes = N
else:
    # Parameters for Generator
    partition = gen_imageName_dict(test_dir,train_dir, 0.2)
    imageName_ID_dict = dict(zip(file_names,labels_int))




architecture_space = {   'n_conv_layers': hp.uniform('n_conv_layers', 2, 10),
            'no_filters_conv': hp.uniform('no_filters_conv', 12, 32),
            'dim_conv_kernel': hp.uniform('dim_conv_kernel', 32, 64),
            'poolSize': hp.uniform('poolSize', .25 ,.5),
            'dropout': hp.uniform('dropout', .25 ,.5),
            'outputDense': hp.uniform('outputDense', .25 ,.5),
            'n_classes': n_classes,
            }
params = {   'n_conv_layers': 2,
            'no_filters_conv': 12,
            'dim_conv_kernel': 10,
            'poolSize': 0.75,
            'dropout': 0.5,
            'outputDense': 0.5,
            'n_classes': n_classes,
            }

def gen_model(params):
    # All conv. filters. of square size
    # conv param: array of size (n_conv,2),(:,0) - n_filters, (:,1) - kernel_dim
    # dense param: array of size (n_dense,1), (:,0) - neurons in dense layers

    model = Sequential()
    my_init = RandomUniform(minval=-0.05, maxval=0.05, seed=None)
    print(my_init)
    for i in range(int(params['n_conv_layers'])):
        if i != 0:
            model.add(Conv2D(params['no_filters_conv'], (params['dim_conv_kernel'],
                                                         params['dim_conv_kernel']), kernel_initializer = 'random_uniform', activation = 'relu'))
        else:
            model.add(Conv2D(params['no_filters_conv'], (params['dim_conv_kernel'],
                                                         params['dim_conv_kernel']), kernel_initializer = 'random_uniform', input_shape=in_shape,
                             activation='relu', data_format="channels_last"))
        # model.add(MaxPooling2D(pool_size=(params['poolSize'], params['poolSize']),
        #                        strides=None, padding='valid', data_format=None))
        model.add(Dense(params['outputDense'], kernel_initializer = 'random_uniform', bias_initializer='zeros', activation = 'relu'))
        model.add(Dropout(0.5))
        model.add(Dense(params['n_classes'], activation= 'softmax')) # for classification??




    # build_fn should construct, compile and return a Keras model, which will then be used to fit/predict
    model.compile(optimizer='rmsprop',
          loss='categorical_crossentropy',
          metrics=['accuracy'])  # compiling and with training parameters although not going to train + of course need it for predict_classes later on
    #  predict on the training data
    predictions = []
    for i, ID in enumerate(imageName_ID_dict):
        # Store sample
        image = np.load('../DATA/train_npy/' + ID + '.npy')
        prediction = model.predict_classes(image)
        predictions.append(prediction)
    loss = categorical_crossentropy(predictions, imageName_ID_dict.values())
    return loss

# Note downgrade networkx, run this: pip3 install networkx==1.11
trials = Trials()
best = fmin(fn = gen_model, space = architecture_space, algo=tpe.suggest, max_evals=15, trials=trials)
print('best: ')
print(best)