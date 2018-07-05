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
def main():
    pass


if __name__ == "__main__":
    main()
