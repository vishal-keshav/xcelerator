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
import threading
import time

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
        if self.is_training == True:
            return

        sess = tf.Session()
        y = self.model['output_tensor']
        y_ = tf.placeholder(tf.float32, shape=?)
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
        train_op = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
        # For stats
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # Train script
        # Train model on data based on params, in one thread
        def training_func():
            while True:
                batch = mnist.train.next_batch(100)
                global_step_val,_ = sess.run([global_step, train_step], feed_dict={x: batch[0], y_: batch[1]})
                print("global step: %d" % global_step_val)
        threading.Thread(target=training_func).start()
        self.is_training = True
        # Format and save train stats based on accuracy data
        self.train_stats = None
        self.is_training = False

    def get_stats(self):
        if self.is_training == False:
            return self.train_stats
        else:
            return None

    def train_and_stats(self):
        #Train model in this thread
        self.train()
        return self.train_stats
