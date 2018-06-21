"""

A profiler module to profile:
    * Memory footprint
    * Parameters
    * Floating point operations
    * Multiply-accumulate operations
    * Single threaded execution

Memory footprint: Architected model is converted to protocol
buffer binary, freezed for embedded exections, read from the disk,
and analyze disk space used by the file. A more advance version of
memory footprint analyzer would be to access architecutre layer
wise and report maximum cpu highest level cache requirement.

Parameters: Number of parameters (int or floats) used the inference
architecure.

Multiply-accumulate operations: MACs are mainly multiplications
followed by additions for the operations such as convolutions and
deconvolution. Reporting this require to iterate through operations
(with a known dimention for input, possibly batch size of one) and
calculate multiplications-additions with a static formulae.
Reference: Netscope analyzer for caffe

FLOPS calculation: Takes in model architecuture, use tf profiler
for reporting FLOPS. This is a better parameter than MACs as it
includes computations by all operations (un-parametrized)

Single threaded execution: The model is run on desktop or on mobile
with single thread configuration, and report the average runtime of
100 runs along with variance for variability. This uses the tf lite
profiler for embedded version profiling

This module is GPLv3 licensed.

"""

import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_a_simple_model():
    input = tf.placeholder(tf.float32, [1, 32, 32, 3])
    conv1 = tf.layers.conv2d(inputs=input, filters=32,
                kernel_size=[3, 3], activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    return input, pool1

# Parameters
def profile_param():
    """
    Profile with metadata
    """
    tf.reset_default_graph()
    run_meta = tf.RunMetadata()
    input, output = get_a_simple_model()

    profile_op = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
    params = tf.profiler.profile(tf.get_default_graph(), run_meta=run_meta, cmd='op',
                                    options=profile_op)
    print('PARAMS: ', params.total_parameters)


def print_nodes():
    """
    Print ops in the graph, with feedable information
    """
    tf.reset_default_graph()
    input, output = get_a_simple_model()
    graph = tf.get_default_graph()
    ops_list = graph.get_operations()
    tensor_list = np.array([ops.values() for ops in ops_list])
    print('PRINTING OPS LIST WITH FEED INFORMATION')
    for t in tensor_list:
        print(t)

    """
    Iterate over trainable variables, and compute all dimentions
    """
    total_dims = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape() # of type tf.Dimension
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_dims += variable_parameters
    print('TOTAL DIMS OF TRAINABLE VARIABLES', total_dims)

# Floating point operations
def profile_flops():
    """
    Profiler with metadata
    """
    tf.reset_default_graph()
    run_meta = tf.RunMetadata()
    input, output = get_a_simple_model()

    profile_op = tf.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.profiler.profile(tf.get_default_graph(), run_meta=run_meta, cmd='op',
                                    options=profile_op)
    print('FLOPS:', flops.total_float_ops)


def main():
    print("............Testing the profiler module................")
    profile_flops()
    profile_param()
    print_nodes()
    return

if __name__ == "__main__":
    main()
