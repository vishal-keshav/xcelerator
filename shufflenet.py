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

    # Defenition of macro-architecture
    # Defined in terms of all hyper-parameters
    def init_shufflenet(self, param):
        nr_groups = param['nr_groups']
        ###
        ##Something something
        ###
        return {'input': input, 'output': output, 'logits': global_pool}
