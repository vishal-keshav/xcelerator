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

import os
import profile_tf as profiler

def update_model_stats(model):
    input = model['input']
    output = model['output']


"""
We dont save graph, and assume default graph is
all we have
"""

class model_generator:
    """
    options is of dictionary type, param has to be
    one of the key.
    """
    def __init__(self, options, name = 'default'):
        self.name = name
        self.nr_param = len(options[param])
        self.init_param = options[param]
        self.param = options[param]
        self.others = options
        # do something with other options here
        self.model = generate_model(self.init_param)
        self.model_stats = update_model_state()

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
        self.model = generate_model(self.param)
        self.model_stats = update_model_stats(self.model)

    def update_model_state():
        input = self.model['input']
        output = self.model['output']
        num_param = profiler.profile_param(tf.get_default_graph())
        num_flops = profiler.profile_flops(tf.get_default_graph())
        #file_size = profiler.profile_size(tf.get_default_graph())
        self.model_stats = {"param": num_param, "flops": num_flops}

    def get_model(self):
        return self.model

    def set_and_get_model(self, param):
        # do param range checks
        # do checks with respect to current setting
        self.param = param
        self.model = generate_model()
        return self.model

    def get_model_stats(self):
        return self.model_stats

def main():
    pass


if __name__ == "__main__":
    main()
