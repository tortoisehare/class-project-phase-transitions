#Binary Classification using a Neural Network in TF

#right now, one hidden layer with 100 nodes

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
#import pandas as pd
%matplotlib inline

#import data
traindata = np.load('Xalphay_smallset.npy')
testdata = np.load('Xalphay_dev.npy')
X_train = traindata[:,:-2]
Y_train = traindata[:,-1]
#Y_train = Y_train.reshape(len(Y_train),1)
X_test = testdata[:,:-2]
Y_test = testdata[:,-1]
#Y_test = Y_test.reshape(len(Y_test),1)

#option for splitting data - 98/1/1 split with random shuffle turned on
'''
train_test_split(arrays, test_size, train_size, random_state (pick seed for randomization), 
shuffle (whether or not to shuffle data), stratify)
'''
#X_train, X_testdev, y_train, y_testdev = train_test_split(X, y, test_size = 0.02, random_state=31, shuffle=True)

#create dev set from test set distribution - I think this would work?
#X_dev, X_test, y_dev, y_test = train_test_split(X_testdev, y_testdev, test_size = 0.5, random_state=31, shuffle=True)

#convert to np arrays
#X_train = np.array(X_train).astype(np.float32)
#X_test = np.array(X_test).astype(np.float32)
#X_dev = np.array(X_dev).astype(np.float32)
#y_train = np.array(y_train).astype(np.float32)
#y_test = np.array(y_test).astype(np.float32)
#y_dev = np.array(y_dev).astype(np.float32)

#define graph
n_inputs = X_train.shape[0] #number of samples
n_inputs_dim = X_train.shape[1] #number of features

#using one hidden layer with 100 nodes
n_hidden_1 = 100 #hidden nodes in layer 1
#later we can add n_hidden_2, etc.
n_output = 1 #output node, should this be number of output classes instead?

#placeholders
X_input = tf.placeholder(tf.float32, [None, n_inputs_dim], name='input')
y = tf.placeholder(tf.float32, [None, n_output], name='y')
#reshape labels to match placeholder
Y_train = Y_train.reshape(-1,1)


#Initialize weights
initializer = tf.contrib.layers.xavier_initializer()


'''
Option 1 - use TF's (fully_connected) to run the layers
'''
#hidden layer using ReLU for now
hidden1 = fully_connected(X_input, n_hidden_1, activation_fn=tf.nn.relu,
                         weights_initializer=initializer)
#output layer using sigmoid for now
logits = fully_connected(hidden1, n_output, activation_fn=tf.nn.linear,
                        weights_initializer=initializer)


'''
Option 2 - write out the layers manually
'''
##ReLU activation on hidden layer 1
#layer_1 = tf.add(tf.matmul(x, weights['W1']), biases['b1'])
#layer_1 = tf.nn.relu(layer_1)
#output layer - sigmoid?
#out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
#out_layer = tf.nn.sigmoid(out_layer)

#Layer weights and biases
#weights = {
    #'W1': tf.Variable(tf.random_normal([n_inputs_dim, n_hidden_1])),
    #'out': tf.Variable(tf.random_normal([n_hidden_1, n_output]))
#}

#biases = {
    #'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    #'out': tf.Variable(tf.random_normal([n_output]))
#}

#using TF's loss function
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
cost = tf.reduce_mean(loss)
cost_so_far = []


#learning rate, epochs, batch size
learning_rate = 0.01
training_epochs = 100
batch_size = 2

#use Adam Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


#initialize global variables
init = tf.global_variables_initializer()

#run sessions
with tf.Session() as sess:
    sess.run(init) #initiate global variables
    
    #run through epochs
    for epoch in range(training_epochs):
        avgcost = 0.0
        tot_batch = int(len(X_train)/batch_size)
        X_batches = np.array_split(X_train, tot_batch)
        Y_batches = np.array_split(Y_train, tot_batch)
        
        #run through batches in epoch
        for i in range(tot_batch):
            batch_x, batch_y = X_batches[i], Y_batches[i]

            #run Adam and calculate cost
            _, cost_ = sess.run([optimizer, cost], feed_dict = {X_input: batch_x, y: batch_y})
            avgcost += cost_/tot_batch
            
        cost_so_far.append(avgcost)
        
    #training loss, plot per epoch
    plt.figure
    plt.plot(cost_so_far)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.show()
    
    
    #make a prediction using logits (fully_connected)
    pred = sess.run([logits], feed_dict = {X_input: X_test})[0]
    y_hat = np.where(pred<0.5, 0, 1)
    
    #Calculate test accuracy
    accuracy = np.sum(Y_test.reshape(-1,1)==y_hat)/len(Y_test)
    print("Test Accuracy %.2f" %accuracy)
