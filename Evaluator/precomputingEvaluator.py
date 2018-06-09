"""
"""
import argparse
import keras
import h5py
import os
import keras.backend as K
from keras.models import Sequential
from keras.layers import Input, Lambda
import numpy as np
import time


K.clear_session()

parser = argparse.ArgumentParser(description='Evaluate a model on the test data and prepare a Kaggle output file')
parser.add_argument('--path', help= 'paste path to the model file')
parser.add_argument('--data', help= 'paste path to the data folder')
parser.add_argument('--output', help= 'name for the output file (in the data folder), defaults to KaggleTestPredictions.txt', default = 'KaggleTestPredictions.txt')


args = parser.parse_args()

print("Loading the model from " + args.path)


model = keras.models.load_model(args.path)
model.summary()

#Make the pre-compute model
precomputeModel = Sequential()
for layer in model.layers:
    if layer.name == "input_1":
        precomputeModel.add(layer)
        print("Input layer added to the precompute network")
    if layer.name == "sequential_1":
        precomputeModel.add(layer)
        print("Sequential layer added to the precompute network")
print("Precompute network input:  " + str(precomputeModel.input.shape))
print("Precompute network output: " + str(precomputeModel.output.shape))
precomputeModel.summary()
print("------------------")


#Make the comparison model
for layer in model.layers:
    if layer.name == "dense_3":
        print("Comparison function input:  " + str(layer.input.shape))
        print("Comparison function output: " + str(layer.output.shape))
        newOutputs = layer
        comparisonFunction = K.function([layer.input],
                                        [layer.output])
        print("Comparison function is generated")
        print("------------------")

# Load the test and data
trainDataset = h5py.File(os.path.join(args.data, 'tr_gr_64.h5'), 'r')
testDataset = h5py.File(os.path.join(args.data, 'tst_gr_64.h5'), 'r')

trainX = np.array(trainDataset['x'])
trainY = np.array(trainDataset['y']).astype('str')
testX = np.array(testDataset['test_data'])
testFileNames = np.array(testDataset['test_labels']).astype('str')[:, 0]

# Do the precomputation
print("Pre-computing the training dataset...")
start = time.time()
trainXprecomp = precomputeModel.predict(x = trainX[:, :, :, np.newaxis])
print("Pre-computing the training dataset took " + str(int((time.time()-start))) + " seconds")

print("Pre-computing the test dataset...")
start = time.time()
testXprecomp = precomputeModel.predict(x = testX[:, :, :, np.newaxis])
print("Pre-computing the test dataset took " + str(int((time.time()-start))) + " seconds")


# Set up the output dictionary
guesses = {}
outputFile = open(os.path.join(args.data, args.output),'w')
print("Saving output in: " + str(os.path.join(args.data, args.output)))
print("------------------")
outputFile.write("Image,Id")

averageProcessingTime = 0
i = 0

# Iterate over the test dataset
for testImage, testName in zip(testXprecomp, testFileNames):
    i=i+1
    start = time.time()
    print("Lookin' up the whale in image " + testName+" [" + str(i).zfill(5) + "/" + str(len(testFileNames)).zfill(5) + " ]. ", end='', flush=True)
    # See how similar the new image is to all the images in the train set.
    #predictions = model.predict(x = [np.repeat(testImage[ np.newaxis, :, :, np.newaxis], trainX.shape[0], axis=0), trainX[:, :, :, np.newaxis]])

    predictions = comparisonFunction([np.abs(trainXprecomp - testImage[np.newaxis,:])])
    predictions = predictions[0]

    # Find the 4 most similar images (based on the SECOND output which is how dissimilar they are. Hence we are looking for the FIRST entries
    ranks = np.argsort(predictions[:,1])
    sortedLabels = trainY[ranks]
    guesses[testName] = []
    for sortedLabel in sortedLabels: #needed as to make sure there are no dublicates
        if len(guesses[testName]) < 4 and sortedLabel not in guesses[testName]:
            guesses[testName].append(sortedLabel)
    print("Probably one of ", end='', flush=True)
    print(guesses[testName], end='', flush=True)

    # Save to the output file
    outputFile.write("\n"+testName + ",")
    outputFile.write("new_whale")
    for label in guesses[testName]:
        outputFile.write(" " + label)

    outputFile.flush()

    averageProcessingTime = (averageProcessingTime*(i-1)+ time.time() - start)/i
    print(". Search took " + str(int((time.time() - start)*1000)) + "ms. Remaining time: " + str(int((averageProcessingTime*(len(testFileNames)-i))/60)) + " min.")

outputFile.close()
