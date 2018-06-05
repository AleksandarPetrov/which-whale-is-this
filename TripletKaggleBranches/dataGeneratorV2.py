

## load the data

# tr_gr_64.h5
import h5py
import keras
import numpy as np
from sklearn.preprocessing import LabelEncoder
from numpy import zeros
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from collections import Counter
import matplotlib.pyplot as plt

# Some useful directories
test_dir = '../../DATA/test_npy'
train_dir = '../../DATA/train_npy'
labels_dir = '../../DATA/train.csv'

####### h5 ##############################
filename = '../../DATA/tr_gr_64.h5'
data = h5py.File(filename, 'r')

#  images, imageNames, imageLabels
X = data['x'] # get first x_value X.value[0]
y = data['y'] # labels

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
#print(y.value)
print(file_names)

dict_ID_imageNames = {}
for i,filename in enumerate(file_names):
    if filename in dict_ID_imageNames.keys():
        print(i)
        dict_ID_imageNames[list_ids[i]].append(filename)
    else:
        dict_ID_imageNames[list_ids[i]] = filename
print(len(dict_ID_imageNames['new_whale']))


class SiameseDataGeneratorV2(keras.utils.Sequence):
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
        X1 = np.empty((self.batch_size, *self.dim, self.n_channels)) # Anchor
        X2 = np.empty((self.batch_size, *self.dim, self.n_channels)) # Positive
        X3 = np.empty((self.batch_size, *self.dim, self.n_channels)) # Negative
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            ### Get the anchor ###
            # Store sample 1
            img1 = np.load(os.path.join(parent_dir, 'train_npy/' + ID + '.npy'))
            img1 = img1[:, :, np.newaxis]
            X1[i,] = img1
            X1_label = self.labels[ID]


            ### Get the positive ###
            listOfPicturesOfSameWhale = [k for k in list(self.labels.keys()) if self.labels[k] == X1_label]
            # For the second one take one from the same class is i is even, otherwise one with a different class,
            # also checks if the class is not new_whale and if there is at least one other picture of the same whale
            ###DATA AUGMENTATION SHOULD BE PUT HERE AND A LOT OF THINGS ADJUSTED
            # if i%2==0 and X1_label!=0:
            X2_ID = np.random.choice(listOfPicturesOfSameWhale, 1)[0]
            img2 = np.load(os.path.join(parent_dir, 'train_npy/' + X2_ID + '.npy'))
            # Augment:
            augParam = aug_para(rot_deg=min(10, max(-10, np.random.normal(loc=0.0, scale=5))),
                                width=min(0.1, max(-0.1, np.random.normal(loc=0.0, scale=0.05))),
                                height=min(0.1, max(-0.1, np.random.normal(loc=0.0, scale=0.05))),
                                shear=min(0.1, max(-0.1, np.random.normal(loc=0.0, scale=0.05))),
                                zoom=min(0.1, max(-0.1, np.random.normal(loc=0.0, scale=0.05))))
            img2 = img_data_aug_array(augParam, img2)
            #if(self.shown<30):
            #    plt.imshow(img)
            #    plt.savefig('foo'+str(self.shown)+'.png')
            #    self.shown=self.shown+1
            img2 = img2[:, :, np.newaxis]
            X2[i,] = img2
            # Store class
            y[i] = X1_label # ADD LABEL HERES


            ### Get the negative ###
            listOfPicturesOfDifferentWhales = [k for k in list(self.labels.keys()) if (self.labels[k] != X1_label or self.labels[k]!=0) and k != ID]
            X3_ID = np.random.choice(listOfPicturesOfDifferentWhales, 1)[0]
            img3 = np.load(os.path.join(parent_dir, 'train_npy/' + X2_ID + '.npy'))
            # Augment:
            augParam = aug_para(rot_deg=min(10, max(-10, np.random.normal(loc=0.0, scale=5))),
                                width=min(0.1, max(-0.1, np.random.normal(loc=0.0, scale=0.05))),
                                height=min(0.1, max(-0.1, np.random.normal(loc=0.0, scale=0.05))),
                                shear=min(0.1, max(-0.1, np.random.normal(loc=0.0, scale=0.05))),
                                zoom=min(0.1, max(-0.1, np.random.normal(loc=0.0, scale=0.05))))
            img3 = img_data_aug_array(augParam, img3)
            #if (self.shown < 30):
            #    plt.imshow(img)
            #    plt.savefig('foo'+str(self.shown)+'.png')
            #    self.shown=self.shown+1
            img3 = img3[:, :, np.newaxis]
            X3[i,] = img3



        return [X1, X2, X3], y
