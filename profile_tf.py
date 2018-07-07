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
from tensorflow.python.framework import graph_util
import numpy as np
import os

import os.path as op
import adb
from adb import adb_commands
from adb import sign_m2crypto

import report as report

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_a_simple_model():
    input = tf.placeholder(tf.float32, [1, 32, 32, 3], name = 'input_tensor')
    conv1 = tf.layers.conv2d(inputs=input, filters=32,
                kernel_size=[3, 3], activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    output = tf.identity(pool1, name="output_tensor")
    return {'input': input, 'output': output}

# Parameters
def profile_param(graph, verbos = True):
    """
    Profile with metadata
    Takes in a graph, and calculate
    total parameters present in that graph
    """
    run_meta = tf.RunMetadata()
    profile_op = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
    params = tf.profiler.profile(graph, run_meta=run_meta, cmd='op',
                                    options=profile_op)
    return params.total_parameters

# Floating point operations
def profile_flops(graph, verbose = True):
    """
    Profiler with metadata
    Takes in a graph, and calculate
    total number of floating point ops in that
    """
    run_meta = tf.RunMetadata()
    profile_op = tf.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.profiler.profile(graph, run_meta=run_meta, cmd='op',
                                    options=profile_op)
    return flops.total_float_ops

def print_nodes(graph, verbose = True):
    """
    Print ops in the graph, with feedable information
    """
    ops_list = graph.get_operations()
    tensor_list = np.array([ops.values() for ops in ops_list])
    if verbose == True:
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
    if verbose == True:
        print('TOTAL DIMS OF TRAINABLE VARIABLES', total_dims)
    return total_dims

"""
For profilling the model, we choose to create an untrained version of model
that we save as optimized tflite file format.
Once saved on disk, we measure the disk space taken by it.
Further, the automatic script pushes a the model with tf lite benchmark
into the connected device (Android: /data/) and retrives the execution
time (mean and variance of 100 runs), by generating a dummy input to be feeded.

Option to execute on desktop is even simpler, for which options will be passed.
"""

def create_tflite(name, model, graph, verbose = True):
    input = model['input']
    output = model['output']
    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        # Here, we could have restored the checkpoint and trained weights,
        # but that is not the intention
        graph_def = graph.as_graph_def()
        # Freeze the graph
        output_graph = graph_util.convert_variables_to_constants(sess, graph_def, ["output_tensor"])
        # Convert the model to tflite file directly.
        tflite_model = tf.contrib.lite.toco_convert(
                output_graph, input_tensors=[input], output_tensors=[output])
        with open(name + ".tflite", "wb") as f:
            f.write(tflite_model)
        if verbose == True:
            print(name + ".tflite created")

def profile_file_size(name, verbose = True):
    file_size = os.path.getsize("name" + ".tflite")
    if verbose == True:
        print('FILE_SIZE_IN_BYTES: ' + file_size)
    return file_size

def connect_to_device(verbose = True):
    if verbose == True:
        print("CONNECTING TO ADB....")
        print("If unable to connect, execute adb kill-server")
    try:
        signer = sign_m2crypto.M2CryptoSigner(op.expanduser('~/.android/adbkey'))
        device = adb_commands.AdbCommands()
        device.ConnectDevice(rsa_keys=[signer])
        return device
    except:
        print("execute adb kill-server first")
    return None

def push_tflite(name, device, verbose = True):
    # Check if tflite file is present on disk, then push it into the device
    destination_dir = '/data/local/tmp/' + name + '.tflite'
    file_name = name + '.tflite'
    if op.exists(file_name):
        console_msg = device.Push(file_name, destination_dir)
        if verbose == True:
            print(console_msg)
            print("FILE PUSHED")
    else:
        if verbose == True:
            print("FILE NOT PRESENT")
        return

def execute_tflite(name, device, nr_threads = 1, verbose = True):
    # More checks are required, but for now, its okay!
    # default_name = mobilenet_v1_1.0_224
    benchmark_file = "/data/local/tmp/label_image"
    image_file = "/data/local/tmp/grace_hopper.bmp"
    label_file = "/data/local/tmp/labels.txt"
    model_file = "/data/local/tmp/" + name +".tflite"
    exec_command = "." + benchmark_file + " -c 100 -v 1 -i " +  \
                    image_file + " -l " + label_file + " -m " + \
                    model_file + " -t " + str(nr_threads)
    console_msg = device.Shell(exec_command, timeout_ms=100000)
    if verbose == True:
        print(exec_command)
        print(console_msg)
    return console_msg

def profile_mobile_exec(name, model, graph, verbose = True):
    adb_device = connect_to_device()
    create_tflite(name, model, graph)
    if adb_device != None:
        push_tflite(name, adb_device)
        console_out = execute_tflite(name, adb_device, nr_threads = 1)
    else:
        if verbose == True:
            print("Unable to connect to device.")
        console_out = None
    #formated_out = report.format_adb_msg(console_out)
    formated_out = {'exec_time': -1}
    if verbose == True:
        print(console_out)
        print(formated_out['exec_time'])
    return formated_out

def adb_test():
    print("Connecting to ADB.")
    # KitKat+ devices require authentication
    signer = sign_m2crypto.M2CryptoSigner(op.expanduser('~/.android/adbkey'))
    # Connect to the device
    device = adb_commands.AdbCommands()
    device.ConnectDevice(rsa_keys=[signer])
    # Now we can use Shell, Pull, Push, etc!
    for i in range(10):
        print(device.Shell('echo %d' % i))


def main():
    print("............Testing the profiler module................")
    model = get_a_simple_model()
    graph = tf.get_default_graph()
    #print('FLOPS: ', profile_flops(graph))
    #print('PARAM: ', profile_param(graph))
    #nr_dims = print_nodes(graph)
    out = profile_mobile_exec('test_model', model, graph)
    #sz = profile_file_size('test_model')
    return

if __name__ == "__main__":
    main()
