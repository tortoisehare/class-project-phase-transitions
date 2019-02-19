from sklearn import svm
from sklearn.model_selection import train_test_split #optional
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score, f1_score
from sklearn.linear_model import SGDClassifier
import numpy as np
import pickle
from joblib import dump, load

# Linear SVM Classifier in scikit-learn using mini-batch gradient descent, no additional weight on interface (y = 1) examples

# Set the seed for reproducibility
np.random.seed(seed=1)

# Define paths to data
training_data_path = PATH_TO_TRAINING_DATA  # Replace with path to training data file in .npy format
dev_data_path = PATH_TO_DEV_DATA # Replace with path to dev data file in .npy format

# Load our data
traindata = np.load(training_data_path)
X_train = traindata[:,0:-2]  # all but the final 2 columns are features, assigned to X
y_train = traindata[:,-1]  # final column is output, assigned to y

X_train_full = traindata[:,0:-2]
y_train_full = traindata[:,-1]

devdata = np.load(dev_data_path)
X_dev = devdata[:,0:-2]
y_dev = devdata[:,-1]

# Determine the classes to be predicted
classes = np.unique(y_dev)
print 'Number of classes: {}'.format(classes)

# Define the number of epochs and mini-batch size and number
minibatchsize = 2**20

numminibatch = int(np.shape(X_train)[0]/minibatchsize)
numepoch=100

# Define model as an SGDClassifier with 'hinge' loss (linear SVM) and don't shuffle data since we are running mini-batches
model = SGDClassifier(loss='hinge', penalty='l2', shuffle=False)


itercount = 0

for ep in range(numepoch):    # iterate through epochs
	np.random.shuffle(traindata)  # shuffle training data for each epoch
	X_train = traindata[:,0:-2]  
	y_train = traindata[:,-1]

	for i in range(numminibatch):   # iterate through minibatches
		itercount += 1
                # Define X and y for minibatches
		minibatch_X = X_train[minibatchsize*i: minibatchsize*(i+1)] 
		minibatch_y = y_train[minibatchsize*i: minibatchsize*(i+1)]

                # Fit the model to the mini-batch
		model.partial_fit(minibatch_X, minibatch_y, classes = classes)

                # Predict dev error
		y_pred_dev = model.predict(X_dev)

                # Predict training error
		fulltrain_ypred = model.predict(X_train_full)

                # Determine accuracy, f1, confusion matrix for training and dev sets
		fulltrainaccuracy = accuracy_score(y_train_full, fulltrain_ypred)
		fulltrainf1 = f1_score(y_train_full, fulltrain_ypred)

		print ('Iteration # {:04d}'.format(itercount))
		trainaccuracy = accuracy_score(y_train_full, fulltrain_ypred)
		print("Train Accuracy: {}%".format(trainaccuracy * 100))
		print(confusion_matrix(y_train_full, fulltrain_ypred))  
		print(classification_report(y_train_full, fulltrain_ypred))  

		devaccuracy = accuracy_score(y_dev, y_pred_dev)
		print("Dev Accuracy: {}%".format(devaccuracy * 100))
		print(confusion_matrix(y_dev,y_pred_dev))  
		print(classification_report(y_dev,y_pred_dev))  

		with open('accuracy_dev.txt','a') as accfile:
			accfile.write('{}\n'.format(devaccuracy))

		with open('f1_dev.txt','a') as accfile:
			accfile.write('{}\n'.format(f1_score(y_dev, y_pred_dev)))

		with open('accuracy_fulltrain.txt','a') as accfile:
			accfile.write('{}\n'.format(fulltrainaccuracy))

		with open('f1_fulltrain.txt','a') as accfile:
			accfile.write('{}\n'.format(fulltrainf1))

                # Save the current model for reference
		pickle.dump(model,open('SVM_iter_{:04d}.pickle'.format(itercount),'w'))

        # Repeat above for final minibatch (will condense code)
	minibatch_X = X_train[minibatchsize*(i+1):]
	minibatch_y = y_train[minibatchsize*(i+1):]

	model.partial_fit(minibatch_X, minibatch_y, classes = classes)
	y_pred_dev = model.predict(X_dev)

	fulltrain_ypred = model.predict(X_train_full)
	fulltrainaccuracy = accuracy_score(y_train_full, fulltrain_ypred)
	fulltrainf1 = f1_score(y_train_full, fulltrain_ypred)

	itercount += 1
	print ('Iteration # {:04d}'.format(itercount))
	print("Train Accuracy: {}%".format(fulltrainaccuracy * 100))
	print(confusion_matrix(y_train_full, fulltrain_ypred))  
	print(classification_report(y_train_full, fulltrain_ypred))  

	devaccuracy = accuracy_score(y_dev, y_pred_dev)
	print("Dev Accuracy: {}%".format(devaccuracy * 100))
	print(confusion_matrix(y_dev,y_pred_dev))  
	print(classification_report(y_dev,y_pred_dev))  

	with open('accuracy_dev.txt','a') as accfile:
		accfile.write('{}\n'.format(devaccuracy))

	with open('f1_dev.txt','a') as accfile:
		accfile.write('{}\n'.format(f1_score(y_dev, y_pred_dev)))

	with open('accuracy_fulltrain.txt','a') as accfile:
		accfile.write('{}\n'.format(fulltrainaccuracy))

	with open('f1_fulltrain.txt','a') as accfile:
		accfile.write('{}\n'.format(fulltrainf1))

	dump(model,'SVM_iter_{:04d}.pickle'.format(itercount))

	

