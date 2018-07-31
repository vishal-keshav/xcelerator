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

Defining shufflenet, with model hyper-parameters like
filter_group: Number of groupings for shuffling.
complexity_scale_factor: Determines width of the network

This module is GPLv3 licensed.

"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
import profile_tf as profiler

class Model:
    def __init__(self, name):
        print("Initializing " + name)
        if name == 'shufflenet':
            self.name = name
            print("Model present, call model_creator(param) to create")
        else:
            self.name = None
            print('no such model present in this module')
        self.model = None

    def model_creator(self, param):
        if self.name == 'shufflenet':
            self.model = self.init_squeezenet(param)
            return self.model

    # Stat updator returns the model exec time and other visible param
    # important for deployment
    def stat_updater(self, variance = False):
        num_param = profiler.profile_param(tf.get_default_graph())
        num_flops = profiler.profile_flops(tf.get_default_graph())
        if variance == False:
            single_thread = profiler.profile_mobile_exec(self.name, self.model,
                tf.get_default_graph(), nr_threads = 1, verbose = False)
            multi_thread = profiler.profile_mobile_exec(self.name, self.model,
                tf.get_default_graph(), nr_threads = 8, verbose = False)
        else:
            single_thread = profiler.profile_mobile_exec_var(self.name, self.model,
                tf.get_default_graph(), nr_threads = 1, verbose = False)
            multi_thread = profiler.profile_mobile_exec_var(self.name, self.model,
                tf.get_default_graph(), nr_threads = 8, verbose = False)
        file_size = profiler.profile_file_size(self.name, verbose = False)
        return {"param": num_param, "flops": num_flops,
                "single_thread_mean": single_thread['exec_time'],
                "single_thread_var": single_thread['exec_var'],
                "multi_thread_mean": multi_thread['exec_time'],
                "multi_thread_var": multi_thread['exec_var'],
                "file_size": file_size}

    # Defenition of micro-architecture (shuffeling)
    def shuffle(self, input, nr_groups):
        h,w,ch = input.shape.as_list()[1:]
        ch_per_group = int(ch/nr_groups)
        shape = tf.stack([-1, h, w, nr_groups, ch_per_group])
        out = tf.reshape(input, shape)
        out = tf.transpose(out, [0,1,2,4,3])
        shape = tf.stack([-1, h, w, ch])
        out = tf.reshape(out, shape)
        return out
    # Group convolution
    def group_conv(self, input, nr_filters, nr_groups, kernel=1, stride=1):
        in_ch = input.shape.as_list()[3]
        in_ch_per_group = int(in_ch/nr_groups)
        filter_per_group = int(nr_filters/ nr_groups)
        # Create filter weights and split for group conv
        W_shape = [kernel,kernel, in_ch_per_group, nr_filters]
        W = tf.get_variable(name = 'kernel', shape = W_shape,
                dtype = tf.float32, initializer = tf.random_normal())
        X_splits = tf.split(input, [in_ch_per_group]*nr_groups, axis = 3)
        W_spilts = tf.split(W, [filter_per_group]*nr_groups, axis = 3)
        # Apply convolutions on splits and store
        res = []
        for i in range(nr_groups):
            X_temp = X_splits[i]
            W_temp = W_splits[i]
            conv_op = tf.nn.conv2d(X_temp, W_temp, [1, stride, stride, 1], 'SAME')
            res.append(conv_op)
        return tf.concat(res, 3)

    # Define shufflenet unit
    def shuffle_unit(self, input, nr_groups=3, stride=1):
        in_channels = input.shape.as_list()[3]
        layer = in_channels
        # Group conv 1
        layer = self.group_conv(input, in_channels, nr_groups)
        # No batch norm as of now
        layer = tf.nn.relu(layer)
        layer = self.shuffle(layer, nr_groups)
        # Depthwise conv
        layer = slim.separable_convolution2d(layer, num_outputs=None,
                            stride=stride, depth_multiplier=1, kernel_size=[3, 3])
        # Group conv 2
        layer = self.group_conv(input, in_channels, nr_groups)
        if stride >= 2:
            input = tf.nn.avg_pool(input, [1,3,3,1], [1,2,2,1], 'SAME')
            layer = tf.concat([layer, input], 3)
        else:
            layer = tf.add(layer, input)
        layer = tf.nn.relu(layer)
        return layer


    # Defenition of macro-architecture
    # Defined in terms of all hyper-parameters
    def init_shufflenet(self, param):
        nr_groups = param['filter_group']
        out_channel = param['out_channel']
        complexity = param['complexity_scale_factor']

        if 'input_dim' in param:
            input_dim = param['input_dim']
        else:
            H_W = 224
            input_dim = [1, 224, 224, 3]

        if 'output_dim' in param:
            out_dim = param['output_dim']
        else:
            out_dim = 1000

        out_channel = int(out_channel*complexity)
        input = tf.placeholder(tf.float32, input_dim, name='input_tensor')
        layer = slim.convolution2d(input, 24, kernel_size=[3, 3])
        layer = tf.layers.max_pooling2d(layer, pool_size = 3,
                        strides = 2, padding='valid')
        # Stage 2
        layer = self.shuffle_unit(layer, nr_groups, first=True)
        for i in range():
            layer = self.shuffle_unit(layer, nr_groups)
        # Stage 3
        layer = self.shuffle_unit(layer, nr_groups, stride = 2)
        for i in range():
            layer = self.shuffle_unit(layer, nr_groups)
        # Stage 4
        layer = self.shuffle_unit(layer, nr_groups, stride = 2)
        for i in range():
            layer = self.shuffle_unit(layer, nr_groups)
        # Outputs
        global_pool = tf.reduce_mean(layer, axis = [1,2])
        spatial_reduction = tf.squeeze(global_pool, [1, 2], name='SpatialSqueeze')
        logits = slim.fully_connected(spatial_reduction, out_dim,
                                        activation_fn=None, scope='fc')
        output = slim.softmax(logits, scope='Predictions')
        output = tf.identity(output, name="output_tensor")
        return {'input': input, 'output': output, 'logits': global_pool}
