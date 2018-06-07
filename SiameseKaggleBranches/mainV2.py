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
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from keras.callbacks import ModelCheckpoint
# from gen_imageName_dict import gen_imageName_dict
from collections import Counter
import matplotlib.pyplot as plt
from data_aug import img_data_aug_array, aug_para
from dataGenerator import SiameseDataGenerator
from architectureV2 import basicSiameseGenerator
from gen_imageName_dict import gen_imageName_dict
import h5py

TRAIN_TOP_N_WHALES = False
N = 20
N_EPOCHS = 15
LOAD_WEIGHTS = True

input_shape = (64, 64, 1)

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1) #shape1 = shape=(2,) so shape1[0] = 2 ? i guess

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    #tf.Print(y_pred, tf.shape(y_pred)) #[Note: try to print this]
    margin = accuracy(y_true, y_pred)#1
    a = K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    # list_a = [len(digit_indices[d]) for d in range(n_classes)]
    # n = min(list_a) - 1#

    # print(n)
    for d in range(n_classes):

        for i in range(len(digit_indices[d])): # perhaps so as to get a balanced set
            # print(d)
            try:
                z1_same, z2_same = digit_indices[d][i], digit_indices[d][i + 1]
                inc = random.randrange(1, n_classes)
                dn = (d + inc) % n_classes
                z1_opp, z2_opp = digit_indices[d][i], digit_indices[dn][i]
                print(z1_opp,z2_opp)
                prod = z1_same*z2_same*z1_opp*z2_same
                if prod!= 0:
                    pairs += [[x[z1_same], x[z2_same]]]
                    pairs += [[x[z1_opp], x[z2_opp]]]
                    print('a', np.shape([[x[z1_same], x[z2_same]]]))
                    print('b',np.shape([[x[z1_opp], x[z2_opp]]]))
                    print(x[z1_same])
                labels += [1, 0]
                print('after')
            except:
                continue

    print(np.shape(labels))
    return np.array(pairs), np.array(labels)





def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    print(y_true.dtype)
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

# Some useful directories
parent_dir = './../../DATA/'#sys.argv[1]
test_dir = os.path.join(parent_dir, 'test_npy')
train_dir = os.path.join(parent_dir, 'train_npy')
labels_dir = os.path.join(parent_dir, 'train.csv')

dataset = h5py.File(os.path.join(parent_dir, 'tr_gr_64.h5'), 'r')
X_dataset = np.array(dataset['x'])/255.
y_labels = np.array(dataset['y'])


training_generator = SiameseDataGenerator(parent_dir,X_dataset,y_labels, stochastic = True)


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



x_train = training_generator.X_dataset
y_train = training_generator.y_labels

# create training+test positive and negative pairs

print('a',np.size(le.classes_))
print('b',np.size(y_train))
digit_indices = [np.where(y_train == id)[0] for id in le.classes_]
tr_pairs, tr_y = create_pairs(x_train, digit_indices)


# Saving callback
filepath = os.path.join(parent_dir, 'weights.best.basicSiameseV2.hdf5')
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# Model generation
base_network = basicSiameseGenerator(parent_dir = parent_dir,trainable = True)


input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([processed_a, processed_b])
print(K.shape(distance))
# print(K.eval(distance)([input_a,input_b]))

model = Model([input_a, input_b], distance)

# reshape the tr_pairs
tr_pairs = np.reshape(tr_pairs, np.shape(tr_pairs) + (1,) )
print(np.shape(tr_pairs[:, 1]))
print(np.shape(tr_y))
# train
optimizer = Adam(0.008)
model.compile(loss=contrastive_loss, optimizer=optimizer, metrics=[accuracy])
model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
          batch_size=128,
          epochs=N_EPOCHS)

# compute final accuracy on training and test sets
y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
print([y for y in y_pred>np.ones(np.size(y_pred))])
print(tr_y)

tr_acc = compute_accuracy(tr_y, y_pred)

# print(tr_acc)
# y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
# te_acc = compute_accuracy(te_y, y_pred)

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))





# history = model.fit_generator(generator = training_generator,use_multiprocessing=True,epochs= N_EPOCHS,verbose=1)
#
#
# # Save final
# model.save(os.path.join(parent_dir, 'weights.final.basicSiamese.hdf5'))
#
# # Plot the training and validation loss and accuracies
#
# #  "Accuracy"
# plt.figure()
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()
# # "Loss"
# plt.figure()
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()
#
#
