import argparse
import keras
import h5py
import os
import keras.backend as K
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

# Load the test data
trainDataset = h5py.File(os.path.join(args.data, 'tr_gr_64.h5'), 'r')
testDataset = h5py.File(os.path.join(args.data, 'tst_gr_64.h5'), 'r')


trainX = np.array(trainDataset['x'])
trainY = np.array(trainDataset['y']).astype('str')
testX = np.array(testDataset['test_data'])
testFileNames = np.array(testDataset['test_labels']).astype('str')[:, 0]

# Set up the output dictionary
guesses = {}
outputFile = open(os.path.join(args.data, args.output),'w')
outputFile.write("Image, Id")

averageProcessingTime = 0
i = 0

# Iterate over the test dataset
for testImage, testName in zip(testX, testFileNames):
    i=i+1
    start = time.time()

    print("Lookin' up the whale in image " + testName+". ", end='', flush=True)
    # See how similar the new image is to all the images in the train set.
    predictions = model.predict(x = [np.repeat(testImage[ np.newaxis, :, :, np.newaxis], trainX.shape[0], axis=0), trainX[:, :, :, np.newaxis]])

    # Find the 4 most similar images (based on the SECOND output which is how dissimilar they are. Hence we are looking for the FIRST entries
    ranks = np.argsort(predictions[:,1])
    sortedLabels = trainY[ranks]
    guesses[testName] = []
    for sortedLabel in sortedLabels: #needed as to make sure there are no dublicates
        if len(guesses[testName]) <4 and sortedLabel not in guesses[testName]:
            guesses[testName].append(sortedLabel)
    print("Probably one of ", end='', flush=True)
    print(guesses[testName], end='', flush=True)

    # Save to the output file
    outputFile.write("\n"+testName + ",")
    outputFile.write("new_whale")
    for label in guesses[testName]:
        outputFile.write(" " + label)

    averageProcessingTime = (averageProcessingTime*(i-1)+ time.time() - start)/i
    print(". Search took " + str(int(time.time() - start)) + "s. Remaining time: " + str(int((averageProcessingTime*len(testFileNames)-i)*averageProcessingTime/60 )) + " min.")

outputFile.close()