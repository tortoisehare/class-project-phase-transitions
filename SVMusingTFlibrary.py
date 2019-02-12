#Binary Classification using Linear SVM in TensorFlow
#Author: Stephanie Tietz

#Notes:
#Rodrigo used C = 0.3, got 62% accuracy on validation set and training set

#TF documentation for SVM, "fit", and "evaluate" say that some
#arguments are deprecated and will be removed in 2016 and to migrate
#to tf.estimator, but there's no SVM model in there?
#TF recommends using a custom model_fn for SVMs
#link: https://www.tensorflow.org/api_docs/python/tf/contrib/learn/SVM
#working on conversion

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#set random seed and start session
np.random.seed(53)
tf.set_random_seed(53)
sess = tf.Session()


#load data as tensors


#SVM from TF
'''
SVM Constructor in TF
__init__(
    example_id_column,
    feature_columns,
    weight_column_name=None,
    model_dir=None,
    l1_regularization=0.0,
    l2_regularization=0.0,
    num_loss_partitions=1,
    kernels=None, #reserved for use in non-linear SVMs
    config=None,
    feature_engineering_fn=None
)
'''
svm = tf.contrib.learn.SVM(example_id_column='example_id', feature_columns=[real_feature_column, sparse_feature_column], l2_regularization=10.0)

#input builders
def input_fn_train:
#...
    return x,y
def input_fn_eval
#...
    return x,y

#fit the data (train)
'''
"fit" in TF
fit(
    x=None,
    y=None,
    input_fn=None,
    steps=None,
    batch_size=None,
    monitors=None,
    max_steps=None
)
'''
svm.fit(input_fn=input_fn_train)


#evaluate the data
'''
"evaluate" in TF
evaluate(
    x=None,
    y=None,
    input_fn=None,
    feed_fn=None,
    batch_size=None,
    steps=None,
    metrics=None,
    name=None,
    checkpoint_path=None,
    hooks=None,
    log_progress=True
)
'''
svm.evaluate(input_fn=input_fn_eval)

#predict
svm.predict(x=x)
