#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 6 12:46:52 2019
Updated on Thurs Mar 7
@author: sbbrown
"""

# Imports
from __future__ import division 
import sys
import numpy as np
#import matplotlib.pyplot as plt

# Import data -----------------------------------------------------------------
try:
    pred = np.load(sys.argv[1])             # y_pred_test.npy for example
    true = np.squeeze(np.load(sys.argv[2])) # y_test.npy for example
except:
    print('Failed to load data')

# Main ------------------------------------------------------------------------
numPredPositives    = 0 # Total number predicted as "1"
numTruePositives    = 0 # Total number actually true "1"
numCorrectPositives = 0 # Of numbers identified as "1", these were actually "1"
numFalsePositives   = 0 # Of numbers identified as "1", these were actually "0"  
numTruePred         = 0 # Of "1"s in true, how many we correctly predicted
numFalsePred        = 0 # Of "1"s in true, how many we incorrectly predicted

# Loop over pred (prediction) array
for index in range(pred.shape[1]):
    if pred[0][index] == 1:
        numPredPositives +=1
        if true[index] == 1: 
            numCorrectPositives +=1
        else:
            numFalsePositives +=1

    if true[index] == 1:      
        numTruePositives +=1
        if pred[0][index] == 1: 
            numTruePred +=1
        else:
            numFalsePred +=1   

# Calculate precision: 
# Of data we identified as "1", how many were actually "1"? -------------------
precision = float(numCorrectPositives/numPredPositives)           # %
print('Precision [%] = ', precision)  
        
# Calculate recall: 
# How many actual "1"s from the total set did we identify as "1"? -------------  
recall = float(numTruePred/numTruePositives)  
print('Recall [%] = ', recall)    

# Calculate f1 score: 
# Harmonic mean of precision and recall --------------------------------------- 
f1 = 2/((1/precision)+(1/recall))
print('f1  = ', f1) 