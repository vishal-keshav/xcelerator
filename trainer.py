"""

Trainer module controls:
    * Data pre-processing
    * Iterations and quality of training
    * Tracks and provide training status

Data pre-processing: This is an important step of
any model training procedure, the data is searched,retrived
and saved in a format which is efficient to use later.
This procedure controls the complexity of data and evaluation

Quality of training: No one knows when to stop a training
procedure unless experianced enough to determine the tradeoff
between time and last ounce of accuracy. Hence, its important
to keep a check on the requirements when such training is
done in an automated way. This procedure provide options
like iterations, and time to train on a pre-processed data.

Tracking training status: This provides full stats once the
training is complete or interrupted in an automatic manner.

This module is GPLv3 licensed.
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
# No threading as of now
#import threading
#import time

import mobilenet_v1 as mobile

"""
A model trainer encapsulate data, model and
training parameters.
It stores the statistics of model training.
"""
class model_trainer:
    def __init__(self, options = None, name = 'default'):
        self.name = name
        if options != None:
            self.nr_param = len(options['parameters'])
            self.param = options['parameters']
            self.data = options['data']
            self.model = options['model']
            self.train_stats = None
        else:
            self.nr_param = None
            self.param = None
            self.data = None
            self.model = None
            self.train_stats = None

    def set_model(self, model):
        self.model = model

    def set_train_param(self, param):
        self.param = param

    def set_train_data(self, data):
        self.data = data

    def train(self, collect_stats = True):
        sess = tf.InteractiveSession()
        x = self.model['input']
        y_ = tf.placeholder(tf.float32, shape=[None, 10])
        out = self.model['logits']# We just need this

        lr = self.param['lr']
        batch_size = self.param['batch_size']
        nr_iteration = self.param['iter']
        # Create backprop algorithm, based on params
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=out))
        train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(out,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        sess.run(tf.global_variables_initializer())
        mnist_data = self.data
        for i in range(nr_iteration):
            data = mnist_data.train.next_batch(batch_size)
            batch_img = tf.reshape(data[0], [-1, 28, 28, 1])
            resized_batch_img = sess.run(tf.image.resize_images(batch_img, [224, 224]))
            if i%2 == 0:
                train_accuracy = accuracy.eval(
                        feed_dict={x:resized_batch_img, y_: data[1]})
                print("step %d, training accuracy %g"%(i, train_accuracy))
            train_step.run(feed_dict={x: resized_batch_img, y_: data[1]})
        # Evaluating the training accuracy
        test_data_img =  tf.reshape(mnist_data.test.images[0:32], [-1, 28, 28, 1])
        resized_test_img = sess.run(tf.image.resize_images(test_data_img, [224, 224]))
        self.train_stats = accuracy.eval(
            feed_dict={ x: resized_test_img,
                y_: mnist_data.test.labels[0:32]})

    def get_stats(self):
        return self.train_stats

    def train_and_stats(self):
        self.train()
        return self.train_stats

def main():
    # This a testing script which does following:
    # 1. Import a model(mobile_net) with one set model settings
    mobilenet_creator = mobile.Model("mobilenet_v1")
    param = {'resolution_multiplier': 1,
                  'width_multiplier': 1,
                  'depth_multiplier': 1}
    model = mobilenet_creator.model_creator(param)
    print(model['input'].get_shape())
    print(model['output'].get_shape())
    # 2. Describe a data set on which model is needed to be trained
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    # 3. Create a trainer, inputing model and data, with one setting
    train_param = {'lr': 0.001, 'batch_size': 32, 'iter': 10}
    options = {'parameters': train_param, 'data': mnist, 'model': model}
    mt = model_trainer(options,name = 'mobilenet_mnist_trainer')
    # 4. Start training, print out the train stats
    print(mt.train_and_stats())


if __name__== "__main__":
    main()
