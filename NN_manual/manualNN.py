import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
from copy import deepcopy

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X.shape[1]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
       mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
       mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
       mini_batch = (mini_batch_X, mini_batch_Y)
       mini_batches.append(mini_batch)
                                                                                                    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
       mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
       mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
       mini_batch = (mini_batch_X, mini_batch_Y)
       mini_batches.append(mini_batch)
    return mini_batches

''' I believe unnecessary
def predict(X, parameters):
    params = dict()

    for p in parameters:
        params[p] = tv.convert_to_tensor(parameters[p])

    x = tf.placeholder('float', [np.shape(X)[0], 1])
    z = forward_propagation(x, parameters)
'''

def create_placeholders(nx, ny):
    X = tf.placeholder(tf.float32, shape = (nx, None), name = 'input')
    y = tf.placeholder(tf.float32, shape = (ny, None), name = 'output')

    return X, y

def initialize_parameters(architecture):
    ''' architecture = {layer #: # of neurons}
    '''
    tf.set_random_seed(1)
    layerlist = list(architecture.keys())
    layerlist.sort()

    parameters = dict()

    for l in layerlist[:-1]:
        Wparam = tf.get_variable('W{}'.format(l), [architecture[l+1], architecture[l]], initializer = tf.contrib.layers.xavier_initializer(seed=1))
        bparam = tf.get_variable('b{}'.format(l), [architecture[l+1], 1], initializer = tf.zeros_initializer())

        parameters['W{}'.format(l)] = Wparam
        parameters['b{}'.format(l)] = bparam

    Wparam = tf.get_variable('W{}'.format(layerlist[-1]), [1, architecture[layerlist[-1]]], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    bparam = tf.get_variable('b{}'.format(layerlist[-1]), [1, 1], initializer = tf.zeros_initializer())
    parameters['W{}'.format(layerlist[-1])] = Wparam
    parameters['b{}'.format(layerlist[-1])] = bparam

    return parameters

def forward_propagation(X, parameters):
    numlayers = int(0.5*len(parameters)) #include check for odd number of parameters

    # A = deepcopy(X)
    A = X

    for l in range(numlayers):
        l += 1
        W = parameters['W{}'.format(l)]
        b = parameters['b{}'.format(l)]
        Z = tf.add(tf.matmul(W, A), b)

        A = tf.nn.relu(Z) # don't need this for last layer

    return Z

def compute_cost(Z, Y):
    logits = tf.transpose(Z) # check dimensions
    labels = tf.transpose(Y) # check dimensions

    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels))  # check if _with_logits is necessary

    return cost

def model(X_train, Y_train, X_test, Y_test, architecture, learning_rate = 0.0001, num_epochs = 1500, minibatch_size = 32, print_cost = True):
    ''' architecture = {layer #: # of neurons}
    '''

    ops.reset_default_graph()
    tf.set_random_seed(1)
    (nx, m) = X_train.shape
    ny = 1 # AP TODO: revisit!
    costs = []

    X, y = create_placeholders(nx, ny)
    parameters = initialize_parameters(architecture)
    Z = forward_propagation(X, parameters)
    cost = compute_cost(Z, y)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        seed = 1  # AP TODO: revisit

        for epoch in range(num_epochs):
            epoch_cost = 0.0

            num_minibatches = int(m / minibatch_size)
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, y: minibatch_Y})
                epoch_cost += minibatch_cost / num_minibatches
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        parameters = sess.run(parameters)
        print ("Training Complete")

        Z = tf.sigmoid(Z)
        Z = tf.round(Z)



        correct_prediction = tf.equal(Z, y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) # add other performance metrics

        '''
        # saving outputs:
        np.save('y_pred_test.npy', Z.eval({X:X_test}))
        np.save('y_pred_train.npy', Z.eval({X:X_train}))
        np.save('y_train.npy', Y_train)
        np.save('y_test.npy', Y_test)
        '''

        print ("Train Accuracy:", accuracy.eval({X: X_train, y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, y: Y_test}))
        return parameters

# TODO: save plots/loss, compute performance metrics
traindata = np.load('Xalphay_smallset.npy')
testdata = np.load('Xalphay_dev.npy')

X_train = np.transpose(traindata[:,:-2])
Y_train = traindata[:,-1]
Y_train = Y_train.reshape(1,len(Y_train))

X_test = np.transpose(testdata[:,:-2])
Y_test = testdata[:,-1]
Y_test = Y_test.reshape(1, len(Y_test))

parameters = model(X_train, Y_train,  X_test, Y_test, {1:X_train.shape[0], 2:100}, minibatch_size = 5, num_epochs=100)
