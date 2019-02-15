#Option 2: Binary Classification using SKLearn's SVM and TensorFlow
#Author: Stephanie Tietz


from sklearn import svm
from sklearn.model_selection import train_test_split #optional
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

#need to fix this function - found it online but it doesn't implement properly
def plot_decision_function(est):
    xx, yy = np.meshgrid(np.linspace(-3, 3, 500),
                         np.linspace(-3, 3, 500))
    # We evaluate the decision function on the grid.
    Z = est.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cmap = plt.cm.Blues
    # We display the decision function on the grid.
    plt.figure(figsize=(5,5));
    plt.imshow(Z,
                extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                aspect='auto', origin='lower', cmap=cmap);
    # We display the boundary where Z = 0
    plt.contour(xx, yy, Z, levels=[0], linewidths=2,
                colors='k');
    # All point colors c fall in the interval .5<=c<=1.0 on the blue colormap. We color the true points darker blue.
    plt.scatter(X[:, 0], X[:, 1], s=30, c=.5+.5*y, lw=1,
                cmap=cmap, vmin=0, vmax=1);
    plt.axhline(0, color='k', ls='--');
    plt.axvline(0, color='k', ls='--');
    plt.xticks(());
    plt.yticks(());
    plt.axis([-3, 3, -3, 3]);

#for test purposes with available data
from sklearn import datasets
irisdata = datasets.load_iris()

#LOAD OUR DATA
#...
#X, y = read_data("insert_title_here.txt") #alternative option for importing data
#data = pd.read_csv("insert_title_here.csv", index_col=0) #if the first row is an index
data = pd.DataFrame(data= np.c_[irisdata['data'], irisdata['target']],
                     columns= irisdata['feature_names'] + ['target'])
print(data.head())



#PREPROCESSING
#should the normalization and randomization of the data go here?
#we also need to get even numbers of y=0 and y=1 (many more y=0 in the data than y=1)
X = data.drop('target', axis=1) #we can use this to cut off the "y" column that shows classification
y = data['target']

#cols = data.columns
#features = cols[0:22]
#labels = cols[22]
#data_norm = pd.DataFrame(data)

#for feature in features:
    #data[feature] = (data[feature] - data[feature].mean())/data[feature]/std()
#print("Averages are:")
#print(data.mean())
#print("\n Equal Variance:")
#print(pow(data.std(),2))

#one way to shuffle data is through changing indices
#indices = data_norm.index.tolist()
#indices = np.array(indices)
#np.random.shuffle(indices)
#X = data_norm.reindex(indices)[features]
#y = data_norm.reindex(indices)[labels]



#option for splitting into sets - use sklearn's module (which randomly chooses subsets), test_size set to 20%
#at the moment (80/10/10)
'''
train_test_split(arrays, test_size, train_size, random_state (pick seed for randomization), 
shuffle (whether or not to shuffle data), stratify)
'''
X_train, X_testdev, y_train, y_testdev = train_test_split(X, y, test_size = 0.20, random_state=31, shuffle=True)

#create dev set from test set distribution - I think this would work?
X_dev, X_test, y_dev, y_test = train_test_split(X_testdev, y_testdev, test_size = 0.50, random_state=31, shuffle=True)

'''
#to use in TF, need to be np arrays
X_train = np.array(X_train).astype(np.float32)
X_test = np.array(X_test).astype(np.float32)
X_dev = np.array(X_dev).astype(np.float32)
y_train = np.array(y_train).astype(np.float32)
y_test = np.array(y_test).astype(np.float32)
y_dev = np.array(y_dev).astype(np.float32)
'''



#creating the model
'''
svm.SVC(C=1.0, kernel=’rbf’, degree=3, gamma=’auto_deprecated’, 
coef0=0.0, shrinking=True, probability=False, tol=0.001, 
cache_size=200, class_weight=None, verbose=False, 
max_iter=-1, decision_function_shape=’ovr’, random_state=None
'''
#low C (1) means large decision boundary, some misclassifications; high C (100) means small decision boundary, may overfit
model = svm.SVC(C=1.0, kernel='linear')  

#fit/train the model
model.fit(X_train, y_train)

#You can plot the decision function here
#plot_decision_function(model)

#predictions
y_pred = model.predict(X_test)  

#Evaluate
print("Accuracy: {}%".format(model.score(X_test, y_test) * 100 ))
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))  


#Options to tune parameters
# Grid Search
# Parameter Grid
#param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10]}
 
# Make grid search classifier (3-fold cross-validation)
#model_grid = GridSearchCV(svm.SVC(), param_grid, verbose=1)
 
# Train the classifier
#model_grid.fit(X_train, y_train)
 
# model = grid.best_estimator_()
#print("Best Parameters:\n", model_grid.best_params_)
#print("Best Estimators:\n", model_grid.best_estimator_)
