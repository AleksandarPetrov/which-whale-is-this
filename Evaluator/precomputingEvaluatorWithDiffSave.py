"""
Run with python3 precomputingEvaluator.py --path ../../DATA/weights.Siamese.best.binary_accuracy.training.hdf5
        --data ../../DATA --output June09_fixedSiamese_KaggleTestPredictions.txt  > ../../DATA/June09_fixedSiamese_KaggleTestPredictions.log
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
parser.add_argument('--layer_input', help= 'name of one of the input layers', default = 'input_1')
parser.add_argument('--layer_leg', help= 'name of the whole leg layer', default = 'sequential_1')
parser.add_argument('--layer_dense', help= 'name of one of the dense layer that computes the final output', default = 'dense_3')



args = parser.parse_args()

print("Loading the model from " + args.path)


model = keras.models.load_model(args.path)
model.summary()

print("Model loaded")
print("------------------")
print("Input layer name: " + args.layer_input)
print("Leg layer name  : " + args.layer_leg)
print("Dense layer name: " + args.layer_dense)
print("------------------")


#Make the pre-compute model
precomputeModel = Sequential()
for layer in model.layers:
    if layer.name == args.layer_input:
        precomputeModel.add(layer)
        print("Input layer added to the precompute network")
    if layer.name == args.layer_leg:
        precomputeModel.add(layer)
        print("Sequential layer added to the precompute network")
print("Precompute network input:  " + str(precomputeModel.input.shape))
print("Precompute network output: " + str(precomputeModel.output.shape))
precomputeModel.summary()
print("------------------")


#Make the comparison model
for layer in model.layers:
    if layer.name == args.layer_dense:
        print("Comparison function input:  " + str(layer.input.shape))
        print("Comparison function output: " + str(layer.output.shape))
        newOutputs = layer
        comparisonFunction = K.function([layer.input],
                                        [layer.output])
        print("Comparison function is generated")
        print("------------------")

# Set up the prediction values file
NofPredictions = 50
outputPredDiffFile = open(os.path.join(args.data, "predictionDifferences.csv"),'w')
print("Saving prediction values in: " + str(os.path.join(args.data, "predictionDifferences.csv")))
print("------------------")
print(', '.join((np.array(range(NofPredictions-1))+1).astype(str)) + "\n")
outputPredDiffFile.write(', '.join((np.array(range(NofPredictions))+1).astype(str)) + "\n")

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
    print("Lookin' up the whale in image " + testName+" [" + str(i).zfill(5) + "/" + str(len(testFileNames)).zfill(5) + "]. ", end='', flush=True)
    # See how similar the new image is to all the images in the train set.
    #predictions = model.predict(x = [np.repeat(testImage[ np.newaxis, :, :, np.newaxis], trainX.shape[0], axis=0), trainX[:, :, :, np.newaxis]])

    predictions = comparisonFunction([np.abs(trainXprecomp - testImage[np.newaxis,:])])
    predictions = predictions[0]

    # Find the 4 most similar images (based on the SECOND output which is how dissimilar they are. Hence we are looking for the FIRST entries
    ranks = np.argsort(predictions[:,1])
    ranksWorstToBest = np.argsort(predictions[:,0])
    sortedLabels = trainY[ranks]
    guesses[testName] = []
    for sortedLabel in sortedLabels: #needed as to make sure there are no dublicates
        if len(guesses[testName]) < 4 and sortedLabel not in guesses[testName]:
            guesses[testName].append(sortedLabel)
    print("Probably one of ", end='', flush=True)
    print(guesses[testName], end='', flush=True)

    # Save the predictions
    topNpredictionValues = 1-predictions[ranks[0:NofPredictions],1]
    bottomNpredictionValues = np.flip(1-predictions[ranksWorstToBest[0:NofPredictions],1], 0)
    outputPredDiffFile.write('best, ')
    outputPredDiffFile.write(', '.join(topNpredictionValues.astype(str))+"\n")    
    outputPredDiffFile.write('worst, ')
    outputPredDiffFile.write(', '.join(bottomNpredictionValues.astype(str))+"\n")

    # Save to the output file
    outputFile.write("\n"+testName + ",")
    outputFile.write("new_whale")
    for label in guesses[testName]:
        outputFile.write(" " + label)

    outputFile.flush()

    averageProcessingTime = (averageProcessingTime*(i-1)+ time.time() - start)/i
    print(". Search took " + str(int((time.time() - start)*1000)) + "ms. Remaining time: " + str(int((averageProcessingTime*(len(testFileNames)-i))/60)) + " min.")

outputFile.close()
