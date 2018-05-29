import numpy as np
from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten, MaxPooling2D
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD, Adam
from keras.losses import binary_crossentropy
import numpy.random as rng
from keras.datasets import mnist


def W_init(shape, name=None):
    """Initialize weights as in paper"""
    values = rng.normal(loc=0, scale=1e-2, size=shape)
    return K.variable(values, name=name)


def b_init(shape, name=None):
    """Initialize bias as in paper"""
    values = rng.normal(loc=0.5, scale=1e-2, size=shape)
    return K.variable(values, name=name)

def basicSiameseGenerator():
    input_shape = (28, 28, 1)

    test_input = Input(input_shape)  # this is where we feed the image we want to test if is the same as the known image
    known_input = Input(input_shape)  # this is where we feed the known image
    # It doesn't matter which is which, the network is completely symmetrical

    # BUILDING THE LEGS OF THE SIAMESE NETWORK
    convnet = Sequential()

    convnet.add(Conv2D(filters=8,
                       kernel_size=(3, 3),
                       activation='relu',
                       input_shape=input_shape,
                       kernel_initializer=W_init,
                       kernel_regularizer=l2(2e-4),
                       use_bias=False)
                )

    convnet.add(Conv2D(filters=8,
                       kernel_size=(7, 7),
                       activation='relu',
                       kernel_regularizer=l2(2e-4),
                       kernel_initializer=W_init,
                       bias_initializer=b_init)
                )


    convnet.add(Flatten())

    convnet.add(Dense(units=10,
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
    convnet.summary()
    return siamese_net



class Siamese_Loader:
    """For loading batches and testing tasks to a siamese net"""
    def __init__(self,Xtrain,Xval):
        self.Xval = Xval
        self.Xtrain = Xtrain
        self.n_classes,self.n_examples,self.w,self.h = Xtrain.shape
        self.n_val,self.n_ex_val,_,_ = Xval.shape

    def get_batch(self,n):
        """Create batch of n pairs, half same class, half different class"""
        categories = rng.choice(self.n_classes,size=(n,),replace=False)
        pairs=[np.zeros((n, self.h, self.w,1)) for i in range(2)]
        targets=np.zeros((n,))
        targets[n//2:] = 1
        for i in range(n):
            category = categories[i]
            idx_1 = rng.randint(0,self.n_examples)
            pairs[0][i,:,:,:] = self.Xtrain[category,idx_1].reshape(self.w,self.h,1)
            idx_2 = rng.randint(0,self.n_examples)
            #pick images of same class for 1st half, different for 2nd
            category_2 = category if i >= n//2 else (category + rng.randint(1,self.n_classes)) % self.n_classes
            pairs[1][i,:,:,:] = self.Xtrain[category_2,idx_2].reshape(self.w,self.h,1)
        return pairs, targets

    def make_oneshot_task(self,N):
        """Create pairs of test image, support set for testing N way one-shot learning. """
        categories = rng.choice(self.n_val,size=(N,),replace=False)
        indices = rng.randint(0,self.n_ex_val,size=(N,))
        true_category = categories[0]
        ex1, ex2 = rng.choice(self.n_examples,replace=False,size=(2,))
        test_image = np.asarray([self.Xval[true_category,ex1,:,:]]*N).reshape(N,self.w,self.h,1)
        support_set = self.Xval[categories,indices,:,:]
        support_set[0,:,:] = self.Xval[true_category,ex2]
        support_set = support_set.reshape(N,self.w,self.h,1)
        pairs = [test_image,support_set]
        targets = np.zeros((N,))
        targets[0] = 1
        return pairs, targets

    def test_oneshot(self,model,N,k,verbose=0):
        """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
        pass
        n_correct = 0
        if verbose:
            print("Evaluating model on {} unique {} way one-shot learning tasks ...".format(k,N))
        for i in range(k):
            inputs, targets = self.make_oneshot_task(N)
            probs = model.predict(inputs)
            if np.argmax(probs) == 0:
                n_correct+=1
        percent_correct = (100.0*n_correct / k)
        if verbose:
            print("Got an average of {}% {} way one-shot learning accuracy".format(percent_correct,N))
        return percent_correct

(train_features, train_labels), (test_features, test_labels) = mnist.load_data()
_, img_rows, img_cols =  train_features.shape
num_classes = len(np.unique(train_labels))
num_input_nodes = img_rows*img_cols


train_features = train_features.reshape(train_features.shape[0], img_rows, img_cols,1).astype('float32')
test_features = test_features.reshape(test_features.shape[0], img_rows, img_cols, 1).astype('float32')
train_features /= 255
test_features /= 255    
    
class_indices_train = [0]*10
class_indices_test = [0]*10
for i in range(num_classes):
    class_indices_train[i] = np.where(train_labels == i)   
    class_indices_test[i] = np.where(test_labels == i)  
    
    
n_samples = 100
n_test = 800
dummy_train_features = np.zeros((10,n_samples,28,28))
dummy_test_features = np.zeros((10,n_test,28,28))

for i in range(10):
    mask1 = class_indices_train[i][0][:n_samples]
    mask2 = class_indices_test[i][0][:n_test]
    dummy_train_features[i] = train_features[mask1,:,:,0]
    dummy_test_features[i] = test_features[mask2,:,:,0]
    
    
    
loss_every=300
batch_size = 5
N_way = 20
n_val = 550


siamese_net = basicSiameseGenerator()
loader = Siamese_Loader(dummy_train_features,dummy_test_features)


for i in range(900000):
    (inputs,targets)=loader.get_batch(batch_size)
    loss=siamese_net.train_on_batch(inputs,targets)
