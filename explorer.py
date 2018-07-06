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
start becoming more interesting
"""
import tensorflow as tf
import os
import profile_tf as profiler

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
        self.model = generate_model()
        return get_model_stats()

def main():
    def sample_model_creator(param):
        input = tf.placeholder(tf.float32, [1, param['H'], param['W'], param['D']],
                    name = 'input_tensor')
        conv1 = tf.layers.conv2d(inputs=input, filters=param['F'],
                    kernel_size=[3, 3], activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        output = tf.identity(pool1, name="output_tensor")
        return {'input': input, 'output': output}

    def sample_stat_updater():
        num_param = profiler.profile_param(tf.get_default_graph())
        num_flops = profiler.profile_flops(tf.get_default_graph())
        return {"param": num_param, "flops": num_flops}

    mg = model_generator(name = 'sample')
    mg.set_creator(sample_model_creator)
    mg.set_stats_updater(sample_stat_updater)
    sample_param = {'H': 32, 'W': 32, 'D': 3, 'F': 32}
    mg.set_param(sample_param)
    mg.generate_model()
    print(mg.get_model_stats())


if __name__ == "__main__":
    main()
