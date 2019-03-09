import glob
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pylab as plt
import numpy as np

# Paths to true values
ydev = np.load('/Users/anjli/Box/Current_Classes/CS_230/00-project/05-NN-fullyconnected/01-aws-1layer/01-toaws/y_dev.npy')
#ytrain = np.load('/Users/anjli/Box/Current_Classes/CS_230/00-project/05-NN-fullyconnected/01-aws-1layer/01-toaws/y_train.npy')
ytrain = np.load('Xalphay_augtrain.npy')[:,-1]
#ytrain = np.load('../y_smallset.npy')

trainpredlist = glob.glob('*_ytrainpred.npy')
testpredlist = glob.glob('*_ytestpred.npy')

f, (ax1, ax2, ax3) = plt.subplots(3,1,sharex=True)

for i in range(len(trainpredlist)):
    print(trainpredlist[i])
    trainpred = np.round(np.squeeze(np.load(trainpredlist[i])))
    testpred = np.round(np.squeeze(np.load(testpredlist[i])))

    epochcount = float(trainpredlist[i][:trainpredlist[i].find('_')])

    test_precision = precision_score(ydev, testpred)
    test_recall = recall_score(ydev, testpred)
    test_f1 = f1_score(ydev, testpred)

    train_precision = precision_score(ytrain, trainpred)
    train_recall = recall_score(ytrain, trainpred)
    train_f1 = f1_score(ytrain, trainpred)

    ax1.plot(epochcount, train_precision, 'ro')
    ax2.plot(epochcount, train_recall, 'ro')
    ax3.plot(epochcount, train_f1, 'ro')


    ax1.plot(epochcount, test_precision, 'bo')
    ax2.plot(epochcount, test_recall, 'bo')
    ax3.plot(epochcount, test_f1, 'bo')
    
    print('Epoch {} done'.format(epochcount))

ax1.set_title('Precision vs. Epoch')
ax2.set_title('Recall vs. Epoch')
ax3.set_title('F1 vs. Epoch')

plt.savefig('performance.pdf')

plt.show()
