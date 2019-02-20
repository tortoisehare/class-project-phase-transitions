from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score, f1_score
from sklearn.linear_model import SGDClassifier
import numpy as np
import pickle
from joblib import dump, load

#Linear SVM in scikit-learn using stochastic gradient descent

np.random.seed(seed=1)

#LOAD OUR DATA
training_data_path = PATH_TO_TRAINING_DATA  # Replace with path to training data file in .npy format
dev_data_path = PATH_TO_DEV_DATA # Replace with path to dev data file in .npy format

traindata = np.load(PATH_TO_TRAINING_DATA)
X_train = traindata[:,0:-2]
y_train = traindata[:,-1]

devdata = np.load(PATH_TO_DEV_DATA)
X_dev = devdata[:,0:-2]
y_dev = devdata[:,-1]

#DEFINE PARAMETERS
numepoch = 100
class_weight = {1:83.3, 0:1.0}  # can modifty class weights as desired (this is for balanced weights)


#PREPROCESSING
model = SGDClassifier(loss='hinge', penalty='l2', shuffle=True, max_iter=numepoch, verbose =1, class_weight=class_weight)
model.fit(X_train, y_train)

y_pred_dev = model.predict(X_dev)
y_pred_train = model.predict(X_train)

print('Train accuracy: {}%'.format(model.score(X_train, y_train)*100))
print(confusion_matrix(y_train, y_pred_train))
print(classification_report(y_train,y_pred_train))

print('Dev accuracy: {}%'.format(model.score(X_dev, y_dev)*100))
print(confusion_matrix(y_dev, y_pred_dev))
print(classification_report(y_dev,y_pred_dev))

dump(model,'model_SGD.pickle')


