from sklearn.metrics import precision_score, recall_score, f1_score
import sys
import numpy as np

try:
    pred = np.round(np.squeeze(np.load(sys.argv[1])))
    true = np.squeeze(np.load(sys.argv[2]))
except:
    print('<predicted .npy file> <true .npy file>')
    exit()

print 'Precision = {}'.format(precision_score(true, pred))
print 'Recall = {}'.format(recall_score(true, pred))
print 'F1 = {}'.format(f1_score(true, pred))

