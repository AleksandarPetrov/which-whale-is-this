#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 15:00:28 2018

@author: isabelle
"""

import cv2
import glob
import csv
import numpy as np
from pathlib import Path
import sys
import os

####################################################################################

dataDir = sys.argv[1]

testFiles = glob.glob (os.path.join(dataDir,"testUniformBW/*.jpg"))

####################################################################################

def gen_npy_data_separate(Files):
    for myFile in Files:
       print(myFile)
       image = cv2.imread (myFile)
       print(np.shape(set(image)))
       print(image[:,:,0])
       print(image[:,:,1])
       print(image[:,:,2])
       print(myFile)
	
       ID = myFile.split('/')[-1].split('.')[0]       
       np.save(os.path.join(dataDir,'test_npy/' + ID   + '.npy'), image)
       print(os.path.join(dataDir,'test_npy/' + ID   + '.npy'))
####################################################################################

gen_npy_data_separate(testFiles)
