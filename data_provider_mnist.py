"""
Example for a data_provider_name, replace name by "data_set_name".
Data provider provides data and followes APIs:
    * get_train_batch
    * get_test_batch
    * get_validation_batch (optional)
    * data_transformation

get_train_batch: Takes batch_size and returns batch_size data
with zero dimention representing input data and first dim
label. get_test_batch and get_validation_batch are similar

data_transformation: this transforms the data (both input and label)
before feeding into the network for training. If transformation
requires running tf ops, the module should handle it by running
the transformation in a session.

This module is GPLv3 licensed.
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class data_provider:
    def __init__(self, name):
        self.name = name
        self.data = input_data.read_data_sets('MNIST_data', one_hot=True)
        self.batch_size = None

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def data_transform(self, data):
        batch_img = tf.reshape(data[0], [-1, 28, 28, 1])
        with tf.Session() as sess1:
            resize_bilinear = tf.image.resize_images(batch_img, [224, 224])
            resized_batch_img = sess1.run(resize_bilinear)
        return (resized_batch_img, data[1])

    def _get_train_batch(self, batch_size):
        mnist_data = self.data
        data = mnist_data.train.next_batch(batch_size)
        data = self.data_transform(data)
        return data

    def _get_test_batch(self, batch_size):
        mnist_data = self.data
        data_img = mnist_data.test.images[0:batch_size]
        data_label = mnist_data.test.labels[0:batch_size]
        data = (data_img, data_label)
        data = self.data_transform(data)
        return data

    def get_train_batch(self):
        if self.batch_size != None:
            return self._get_train_batch(self.batch_size)
        else:
            print("First set batch size!")
            return None

    def get_test_batch(self):
        if self.batch_size != None:
            return self._get_test_batch(self.batch_size)
        else:
            print("First set batch size!")
            return None

def main():
    dt = data_provider('mnist')
    data = dt._get_train_batch(32)
    print(data[0].shape)
    print(data[1].shape)

if __name__ == "__main__":
    main()
