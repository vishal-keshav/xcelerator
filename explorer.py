"""

Parameter exploration module to explore:
    * Hyperparameters
    * Model parameters
    * Structure and complexity
    * Memory relationships
    * Cache relationships
    * Processing relationships

Hyper-parameters: An architecure requires several tunable
scalars hyperparameter such as learning rate, momentum,
decay while non-convex optimization iterative search.
While some hyper-parameter setting can take estimator
to best possible configuration, others may never lead
to solution due to overshooting.

Model parameters: A good modelling is characterized by
a good micro-architecure or macro-architecure parameter
setting, giving possibility of evaluating the concept
on a different statement. Given a set of parameter
associated to architecure design itself, best can lead
both to accuracy and efficiency.

Structure and complexity: Anything other than micro
architecure design space exploration can be put in
this unstructured complexity of network.

Memory relationships: RAM in von-neuman model is
an essential block, and will remain untill some
breakthrough such os "optane" happens. Figuring out
the hidden relationship between RAM access and model architecure
(if any) is thus an essential exersise.

Cache relationships: Caches palys an important role
when it comes to relatively larger models (that cannot
fit into it at once). The access patterns and model
complexity as to do many thing with each other (with
compute kernels and testing conditions more or less
same).

Processing relationships: The frequency and temperature
(as controlled by operating system kernel) has to do many
thing with efficiency of an architected model. Fixing this
solves the problem, but if it becomes a variable, things
start becoming more interesting.

This module is GPLv3 licensed.
"""
import tensorflow as tf
import os
import profile_tf as profiler
import mobilenet_v1 as mobile
import report as report

"""
We dont save graph, and assume default graph is
all we have
"""

class model_generator:
    """
    options is of dictionary type, param has to be
    one of the key.
    """
    def __init__(self, options = None, name = 'default'):
        self.name = name
        if options != None:
            self.nr_param = len(options['parameters'])
            self.init_param = options['parameters']
            self.model_creator = options['model_creator']
            self.update_stats = options['stat_updater']
            self.param = options[param]
            self.others = options
            # do something with other options here
            self.model = self.model_creator(self.init_param)
            self.model_stats = self.update_stats()
        else:
            self.nr_param = -1
            self.init_param = None
            self.model_creator = None
            self.param = None
            self.model = None
            self.model_stats = None

    def set_creator(self, fn):
        self.model_creator = fn

    def set_stats_updater(self, fn):
        self.update_stats = fn

    def set_param(self, param):
        # do param range checks
        # do checks with respect to current setting
        self.param = param

    def get_param(self):
        return self.param

    def generate_model(self):
        # logic to generate model basef on self.param
        # checks if it is in correct range
        tf.reset_default_graph()
        self.model = self.model_creator(self.param)

    def get_model(self):
        return self.model

    def set_and_get_model(self, param):
        # do param range checks
        # do checks with respect to current setting
        self.param = param
        self.model = generate_model()
        return self.model

    def get_model_stats(self):
        return self.update_stats()

    def set_and_stats(self, param):
        self.param = param
        self.model = self.generate_model()
        return self.get_model_stats()

def main():

    mobilenet_creator = mobile.Model("mobilenet_v1")
    mg = model_generator(name = 'mobilenet_profile')

    mg.set_creator(mobilenet_creator.model_creator)
    mg.set_stats_updater(mobilenet_creator.stat_updater)

    param_list = []
    stat_list = []
    for res in [1, 0.858, 0.715, 0.572]:
        for width in [1, 0.75, 0.5]:
            for depth in [1, 2]:
                param = {'resolution_multiplier': res,
                              'width_multiplier': width,
                              'depth_multiplier': depth}
                param_list.append(param)
                stat_list.append(mg.set_and_stats(param))
    param_fields = ['resolution_multiplier', 'width_multiplier',
                    'depth_multiplier']
    stat_fields = ['param','flops','single_thread','multi_thread','file_size']
    report.write_data_to_csv(param_list, param_fields, 'model_parameters')
    report.write_data_to_csv(stat_list, stat_fields, 'model_behaviour')


if __name__ == "__main__":
    main()
