#Option 2: Binary Classification using SKLearn's SVM and TensorFlow
#If I can't get a model just in TF, this can be one that uses both
#still working on it

from sklearn import svm
from sklearn.model_selection import train_test_split #optional
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

#for test purposes with available data
from sklearn import datasets
iris = datasets.load_iris()

#LOAD OUR DATA
#...

#Preprocessing options
#should the normalization and randomization of the data go here?
#we also need to get even numbers of y=0 and y=1 (many more y=0 in the data than y=1)
X = irisdata.drop('Class', axis=1) #we can use this to cut off the "y" column that shows classification
y = irisdata['Class']
#X, y = read_data("insert_title_here.txt") #alternative option for importing data

#option for splitting into sets - use sklearn's module (which randomly chooses subsets)
#they don't consider dev/validation sets, though, so I'll add another option here
'''
train_test_split(arrays, test_size, train_size, random_state (pick seed for randomization), 
shuffle (whether or not to shuffle data), stratify)
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=31, shuffle=True)

#creating the model
'''
svm.SVC(C=1.0, kernel=’rbf’, degree=3, gamma=’auto_deprecated’, 
coef0=0.0, shrinking=True, probability=False, tol=0.001, 
cache_size=200, class_weight=None, verbose=False, 
max_iter=-1, decision_function_shape=’ovr’, random_state=None
'''
#low C (1) means large decision boundary, some misclassifications; high C (100) means small decision boundary, may overfit
model = svm.SVC(C=1.0, kernel='linear', gamma = 'scale')  

#fit/train the model
model.fit(X_train, y_train)

#You can plot the decision function here
plot_decision_function(X_train, y_train, X_test, y_test, model)

#predictions
y_pred = model.predict(X_test)  

#Evaluate
print("Accuracy: {}%".format(model.score(X_test, y_test) * 100 ))
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))  


#Options to tune parameters
# Grid Search
# Parameter Grid
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10]}
 
# Make grid search classifier (3-fold cross-validation)
model_grid = GridSearchCV(svm.SVC(), param_grid, verbose=1)
 
# Train the classifier
model_grid.fit(X_train, y_train)
 
# model = grid.best_estimator_()
print("Best Parameters:\n", model_grid.best_params_)
print("Best Estimators:\n", model_grid.best_estimator_)
