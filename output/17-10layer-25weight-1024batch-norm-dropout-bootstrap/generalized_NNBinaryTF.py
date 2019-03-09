#Binary Classification using a Neural Network in TF

#right now, one hidden layer with 100 nodes

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from copy import deepcopy
#import pandas as pd
#%matplotlib inline

# set seed
np.random.seed(seed=1)

#import data
traindata = np.load('/home/ubuntu/project/Xalphay_train.npy')
onesdata = np.load('/home/ubuntu/project/Xalphay_train.npy')

# Bootstrapping to add y = 1 points
num_toadd_1 = 75209*5
idx = np.random.randint(low = 0, high = len(onesdata), size=num_toadd_1)
traindata = np.append(traindata, traindata[idx,:], axis = 0)
np.random.shuffle(traindata)
np.save('Xalphay_augtrain.npy', traindata)

consttrain = deepcopy(traindata)

#traindata = traindata[:1000000]
#consttrain = consttrain[:1000000]

testdata = np.load('/home/ubuntu/project/Xalphay_dev.npy')
consttest = np.load('/home/ubuntu/project/Xalphay_dev.npy')

X_train = traindata[:,:-2]
Y_train = traindata[:,-1]
#Y_train = Y_train.reshape(len(Y_train),1)
X_test = testdata[:,:-2]
Y_test = testdata[:,-1]
#Y_test = Y_test.reshape(len(Y_test),1)



#define graph
n_inputs = X_train.shape[0] #number of samples
n_inputs_dim = X_train.shape[1] #number of features

# define model architecture: dict[layer #] = [n_hidden_units, activation]
modeldict = dict()
for i in range(10):
    modeldict[i] = [100, tf.nn.relu]  # defining 10 layers of 100 units each
n_output = 1 #output node, should this be number of output classes instead?

#placeholders
X_input = tf.placeholder(tf.float32, [None, n_inputs_dim], name='input')
y = tf.placeholder(tf.float32, [None, n_output], name='y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
#reshape labels to match placeholder
Y_train = Y_train.reshape(-1,1)


#Initialize weights
initializer = tf.contrib.layers.xavier_initializer()

'''
Option 1 - use TF's (fully_connected) to run the layers
'''

layerlist = list(modeldict.keys())
layerlist.sort()

A = X_input

for layer in layerlist:
    A = fully_connected(A, modeldict[layer][0], activation_fn=modeldict[layer][1], weights_initializer=initializer, scope = 'fc{}'.format(layer))
    A = tf.contrib.layers.batch_norm(A, scope = 'bn{}'.format(layer))
    A = tf.nn.dropout(A, keep_prob = keep_prob)
    #A = fully_connected(A, modeldict[layer][0], activation_fn=modeldict[layer][1], weights_initializer=initializer, scope = 'fc{}'.format(layer), normalizer_fn=tf.contrib.layers.batch_norm)

#output layer with no activation
logits = fully_connected(A, n_output, activation_fn=None,
                        weights_initializer=initializer, scope = 'outlayer')

#using TF's loss function
loss = tf.nn.weighted_cross_entropy_with_logits(logits = logits, targets=y, pos_weight = 25)
cost = tf.reduce_mean(loss)
cost_so_far = []

#learning rate, epochs, batch size
learning_rate = 0.0001
training_epochs = 100
batch_size = 1024
dropoutkeepprob = 0.9

#use Adam Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#set up saver
saver = tf.train.Saver(max_to_keep=5)

#initialize global variables
init = tf.global_variables_initializer()

#run sessions
with tf.Session() as sess:
    sess.run(init) #initiate global variables
    
    #run through epochs
    for epoch in range(training_epochs):
        avgcost = 0.0
        np.random.shuffle(traindata)  #AP-req
        X_train = traindata[:,:-2]  #AP-req
        Y_train = traindata[:,-1]  #AP-req
        Y_train = Y_train.reshape(-1,1) #AP-req

        tot_batch = int(len(X_train)/batch_size)
        X_batches = np.array_split(X_train, tot_batch)
        Y_batches = np.array_split(Y_train, tot_batch)

        #run through batches in epoch
        for i in range(tot_batch):
            batch_x, batch_y = X_batches[i], Y_batches[i]

            #run Adam and calculate cost
            _, cost_ = sess.run([optimizer, cost], feed_dict = {X_input: batch_x, y: batch_y, keep_prob: dropoutkeepprob})
            avgcost += cost_/tot_batch
            
        cost_so_far.append(avgcost)
        with open('costlist.txt','a') as costfile:
            costfile.write('{0}, {1}\n'.format(epoch, cost_so_far[-1]))

        if epoch % 5 == 0:
            print('Cost after Epoch {} = {}'.format(epoch, avgcost))
            saver.save(sess, './model.ckpt', global_step = epoch)

            # get training error
            pred = sess.run([logits], feed_dict = {X_input: consttrain[:,:-2], keep_prob: 1.0})[0]
            pred = 1.0/(1.0+np.exp(-pred)) # AP - computing sigmoid since we deleted sigmoid in output layer
            np.save('{:02d}_ytrainpred.npy'.format(epoch), pred)
            # get test error
            pred = sess.run([logits], feed_dict = {X_input: consttest[:,:-2], keep_prob: 1.0})[0]
            pred = 1.0/(1.0+np.exp(-pred)) # AP - computing sigmoid since we deleted sigmoid in output layer
            np.save('{:02d}_ytestpred.npy'.format(epoch), pred)

    '''
    #training loss, plot per epoch
    plt.figure
    plt.plot(cost_so_far)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.show()
    '''
    
    # find train accuracy - AP
    pred = sess.run([logits], feed_dict = {X_input: consttrain[:,:-2], keep_prob: 1.0})[0]
    pred = 1.0/(1.0+np.exp(-pred)) # AP - computing sigmoid since we deleted sigmoid in output layer
    np.save('final_ytrainpred.npy', pred)

    #make a prediction using logits (fully_connected)
    pred = sess.run([logits], feed_dict = {X_input: consttest[:,:-2], keep_prob: 1.0})[0]
    pred = 1.0/(1.0+np.exp(-pred)) # AP - computing sigmoid since we deleted sigmoid in output layer
    np.save('final_ytestpred.npy', pred)
    
