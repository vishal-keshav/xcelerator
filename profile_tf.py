"""

A profiler module to profile:
    * Memory footprint
    * Floating point operations
    * Multiply-accumulate operations
    * Single threaded execution

Memory footprint: Architected model is converted to protocol
buffer binary, freezed for embedded exections, read from the disk,
and analyze disk space used by the file. A more advance version of
memory footprint analyzer would be to access architecutre layer
wise and report maximum cpu highest level cache requirement.

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

def get_a_simple_model():
    input = tf.placeholder(tf.float32, [1, 32, 32, 3])
    conv1 = tf.layers.conv2d(inputs=input, filters=32,
                kernel_size=[3, 3], activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    return input, pool1


    # Return the inputs and outputs

def profile_flops():
    """
    Profiler uses metadata
    """
    run_meta = tf.RunMetadata()
    input, output = get_a_simple_model()

    float_ops = tf.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.profiler.profile(tf.get_default_graph(), run_meta=run_meta, cmd='op', options=float_ops)
    print('FLOPS:', flops.total_float_ops)


def main():
    print("............Testing the profiler module................")
    profile_flops()
    return

if __name__ == "__main__":
    main()
