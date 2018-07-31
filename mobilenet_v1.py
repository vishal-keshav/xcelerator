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

As an example, we show how to follow the above guidelines
for defining mobileNet_v1, with model hyper-parameters like
depth_multiplier, width_multiplier and resolution_multiplier.

SqueezeNet and shuffleNet examples will be provided accordingly.
This module is GPLv3 licensed.

"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
import profile_tf as profiler

class Model:
    def __init__(self, name):
        print("Initializing " + name)
        if name == 'mobilenet_v1':
            self.name = name
            print("Model present, model_creator(param) to create")
        else:
            self.name = None
            print('no such model present in this module')
        self.model = None

    def model_creator(self, param):
        if self.name == 'mobilenet_v1':
            self.model = self.init_mobilenet_v1(param)
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

    # Defenition of micro-architecture (depth-wise seperable conv)
    # Defined in terms of params width and depth multiplier
    def dw_separable(self, input, nr_filters, width_multiplier,
                        depth_multiplier, sc, downsample=False):
        nr_filters = round(nr_filters * width_multiplier)
        if downsample:
            stride = 2
        else:
            stride = 1
        depthwise_conv = slim.separable_convolution2d(input, num_outputs=None,
                            stride=stride, depth_multiplier=depth_multiplier,
                            kernel_size=[3, 3], scope= sc+'/depthwise_conv')
        #bn = slim.batch_norm(depthwise_conv, scope=sc+'/dw_batch_norm')
        pointwise_conv = slim.convolution2d(depthwise_conv, nr_filters, kernel_size=[1, 1],
                            scope=sc+'/pointwise_conv')
        #bn = slim.batch_norm(pointwise_conv, scope=sc+'/pw_batch_norm')
        return pointwise_conv

    # Defenition of macro-architecture
    # Defined in terms of all hyper-parameters namely
    # resolution, width and depth multiplier
    def init_mobilenet_v1(self, param):
        resolution_multiplier = param['resolution_multiplier']
        width_multiplier = param['width_multiplier']
        depth_multiplier = param['depth_multiplier']

        if 'input_dim' in param:
            input_dim = param['input_dim']
        else:
            H_W = int(224*resolution_multiplier)
            input_dim = [1, H_W, H_W, 3]

        if 'output_dim' in param:
            out_dim = param['output_dim']
        else:
            out_dim = 1000

        # Define the resolution based on resolution multiplier
        # [1, 0.858, 0.715, 0.572 ] = [224, 192, 160, 128]

        input = tf.placeholder(tf.float32, input_dim, name='input_tensor')
        layer_1_conv = slim.convolution2d(input, round(32 * width_multiplier),
                        [3, 3], stride=2, padding='SAME', scope='conv_1')
        #layer_1_bn = slim.batch_norm(layer_1_conv, scope='conv_1/batch_norm')
        layer_2_dw = self.dw_separable(layer_1_conv, 64, width_multiplier,
                            depth_multiplier, sc='conv_ds_2')
        layer_3_dw = self.dw_separable(layer_2_dw, 128, width_multiplier,
                            depth_multiplier, downsample=True, sc='conv_ds_3')
        layer_4_dw = self.dw_separable(layer_3_dw, 128, width_multiplier,
                            depth_multiplier, sc='conv_ds_4')
        layer_5_dw = self.dw_separable(layer_4_dw, 256, width_multiplier,
                            depth_multiplier, downsample=True, sc='conv_ds_5')
        layer_6_dw = self.dw_separable(layer_5_dw, 256, width_multiplier,
                            depth_multiplier, sc='conv_ds_6')
        layer_7_dw = self.dw_separable(layer_6_dw, 512, width_multiplier,
                            depth_multiplier, downsample=True, sc='conv_ds_7')
        # repeatable layers can be put inside a loop
        layer_8_12_dw = layer_7_dw
        for i in range(8, 13):
            layer_8_12_dw = self.dw_separable(layer_8_12_dw, 512, width_multiplier,
                                        depth_multiplier, sc='conv_ds_'+str(i))
        layer_13_dw = self.dw_separable(layer_8_12_dw, 1024, width_multiplier,
                            depth_multiplier, downsample=True, sc='conv_ds_13')
        layer_14_dw = self.dw_separable(layer_13_dw, 1024, width_multiplier,
                            depth_multiplier, sc='conv_ds_14')
        # Pool and reduce to output dimension
        global_pool = tf.reduce_mean(layer_14_dw, [1, 2], keep_dims=True,
                                        name='global_pool')
        spatial_reduction = tf.squeeze(global_pool, [1, 2], name='SpatialSqueeze')
        logits = slim.fully_connected(spatial_reduction, out_dim,
                                        activation_fn=None, scope='fc_16')
        output = slim.softmax(logits, scope='Predictions')
        output = tf.identity(output, name="output_tensor")
        return {'input': input, 'output': output, 'logits': logits}
