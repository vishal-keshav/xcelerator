"""
General guidelines to define an architecture:
    * Try seperating out micro from macro arch
    * Try defining micro arch in terms of arch
        hyper-parameters.
    * Try integrating macro-arch from micro-arch
    * Define the macro-arch in terms of tf ops only
        since eager execution may not be supported
    * Describe the param in detail in comments.
    * Function signature should be model_creator(param)
        which atleast return {"input": in_tensor, "output": out_tensor}
    * Input tensor should be names as "input_tensor"
    * Output tensor should be names as "output_tensor".
        It would be a good idea to wrap the output in
        an identity and name it as "output_tensor".
    * Additionally, stat_updater can be defined, function
        can have any signature but the name should be stat_updater
    * It would be a good practice to make a class of the two
        functions, with required internal variables.

Defining squeezenet, with model hyper-parameters like
base_expand: Number of expansion filter in first fire module
expansion_increment: Increase in channel through expansion
expansion_filter_ratio: Ratio of 1X1 and 3X3 expansion filters in fire module
filter_expansion_freq: Expansion in filters every freq fire modules
squeeze_ratio: Squeeze ratio of squeeze and expand filters

This module is GPLv3 licensed.

"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
import profile_tf as profiler

class Model:
    def __init__(self, name):
        print("Initializing " + name)
        if name == 'squeezenet':
            self.name = name
            print("Model present, call model_creator(param) to create")
        else:
            self.name = None
            print('no such model present in this module')
        self.model = None

    def model_creator(self, param):
        if self.name == 'squeezenet':
            self.model = self.init_squeezenet(param)
            return self.model

    # Stat updator returns the model exec time and other visible param
    # important for deployment
    def stat_updater(self):
        num_param = profiler.profile_param(tf.get_default_graph())
        num_flops = profiler.profile_flops(tf.get_default_graph())
        single_thread = profiler.profile_mobile_exec(self.name, self.model,
                        tf.get_default_graph(), nr_threads = 1, verbose = False)
        multi_thread = profiler.profile_mobile_exec(self.name, self.model,
                        tf.get_default_graph(), nr_threads = 8, verbose = False)
        """single_thread = profiler.profile_mobile_exec_var(self.name, self.model,
                        tf.get_default_graph(), nr_threads = 1, verbose = False)
        multi_thread = profiler.profile_mobile_exec_var(self.name, self.model,
                        tf.get_default_graph(), nr_threads = 8, verbose = False)"""
        file_size = profiler.profile_file_size(self.name, verbose = False)
        return {"param": num_param, "flops": num_flops,
                "single_thread_mean": single_thread['exec_time'],
                "single_thread_var": single_thread['exec_var'],
                "multi_thread_mean": multi_thread['exec_time'],
                "multi_thread_var": multi_thread['exec_var'],
                "file_size": file_size}

    # Defenition of micro-architecture (fire-module)
    def fire_module(self, input, nr_squeeze_1, nr_expand_1, nr_expand_3):
        # by default, activation is relu and init is xavier
        squeeze_out = slim.convolution2d(input, nr_squeeze_1, kernel_size=[1, 1],
                            stride=1, padding='SAME')
        expand_1_out = slim.convolution2d(squeeze_out, nr_expand_1, kernel_size=[1, 1],
                            stride=1, padding='SAME')
        expand_3_out = slim.convolution2d(squeeze_out, nr_expand_3, kernel_size=[3, 3],
                            stride=1, padding='SAME')
        output = tf.concat(axis = 3, values= [expand_1_out, expand_3_out])
        return output

    # Defenition of macro-architecture
    # Defined in terms of all hyper-parameters
    def init_squeezenet(self, param):
        base_expand_kernels = param['base_expand']
        expansion_increment = param['expansion_increment']
        pct = param['expansion_filter_ratio']
        freq = param['filter_expansion_freq']
        SR = param['squeeze_ratio']

        # Some initialization for structuring
        max_pooling_index = [2, 6]
        H_W = 224
        nr_filter_first_layer = 96
        input = tf.placeholder(tf.float32, [1, H_W, H_W, 3],name='input_tensor')

        layer_1_conv = slim.convolution2d(input, nr_filter_first_layer,
                        [3, 3], stride=2, padding='SAME', scope='conv_1')
        layer_1_pool = tf.layers.max_pooling2d(layer_1_conv, pool_size = 3,
                        strides = 2, padding='valid',name='pool_1')
        temp_layer = layer_1_pool
        for i in range(8):
            expand_kernels = base_expand_kernels + (expansion_increment*(i/freq))
            nr_squeeze_1 = int(SR*expand_kernels)
            nr_expand_1 = int(expand_kernels*(1.0-pct))
            nr_expand_3 = expand_kernels - nr_expand_1
            temp_layer = self.fire_module(temp_layer, nr_squeeze_1, nr_expand_1, nr_expand_3)
            if i in max_pooling_index:
                temp_layer = tf.layers.max_pooling2d(temp_layer, pool_size = 3,
                                strides = 2, padding='valid',name='pool_' + str(i))
        layer_9_conv = slim.convolution2d(temp_layer, 1000, [1,1],
                        stride=2, padding='VALID', scope='conv_9')
        global_pool = slim.avg_pool2d(layer_9_conv, 3, 1, 'VALID')
        output = slim.softmax(global_pool, scope='Predictions')
        output = tf.identity(output, name="output_tensor")
        return {'input': input, 'output': output, 'logits': global_pool}
