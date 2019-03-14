#AUPRC - area under precision curve / precision recall curve
import matplotlib.pylab as plt
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

#change threshold

trainmetrics = dict()
trainmetrics['precision'] = list()
trainmetrics['recall'] = list()
trainmetrics['f1'] = list()

testmetrics = dict()
testmetrics['precision'] = list()
testmetrics['recall'] = list()
testmetrics['f1'] = list()

for i in range(1,10):
    y_hat = np.where(pred<0.1*i, 0, 1)
    
    trainmetrics['precision'].append(precision_score(ytrain, yhat))
    trainmetrics['recall'].append(recall_score(ytrain, yhat))
    trainmetrics['f1'].append(f1_score(ytrain, yhat))
    
    testmetrics['precision'].append(precision_score(ytest, yhat))
    testmetrics['recall'].append(recall_score(ytest, yhat))
    testmetrics['f1'].append(f1_score(ytest, yhat))
    
xrange = np.arange(0.1, 1.0, 0.1)
plt.figure()
plt.plot(xrange, trainmetrics['f1'], 'ro', label = 'train')
plt.plot(xrange, testmetrics['f1'], 'bo', label = 'test')
plt.title('F1 vs. Threshold')

plt.figure()
plt.plot(trainmetrics['recall'], trainmetrics['precision'], 'ro', label = 'train')
plt.plot(testmetrics['recall'], testmetrics['precision'], 'bo', label = 'test')
plt.title('Precision vs. Recall')

plt.show()
