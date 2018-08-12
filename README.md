> This project is under testing and the full source code will be updated soon (by mid January 2019).

# Xcelerator

Xcelerator (Accelerating deep-neural networks) is a python and C++ based framework that tends to fine-tune a deep-neural network for computational efficiency (with negligible or no loss in prediction accuracy). The most basic premise on which this framework lies on is the idea of using a meta-learning algorithm (a policy based on reinforcement learning) to fine-tune(prune weights and reduce layers) a neural network architecture such that accuracy is not affected (or can be re-gained by training the network again). The framework takes performance statistics of the platform (hardware specifications of the SOCs, inference engine capability, cache efficiency etc.) to fine-tune the network, and yet does not assume any platform dependencies to construct the network pruning policy. This is possible because the domain knowledge of the constructed policy can be transferred to a platform-specific policy (with minimal changes).

## Motivation to create **Xcelerator**
A traditional ML application development (targetted for smartphones) workflow includes these three steps:
1. Develop architecture and test it with several different sets of hyper-parameters. Once the accuracy criteria are passed, go to step 2.
2. Test the application performance on multiple platforms (smartphones with different hardware configuration). If performance criteria are passed, go to step 3. If the application fails to perform as intended on several platforms, then lower the accuracy criteria and go to step 1.
3. Finalize the application and publish it to the market.

It can be noted that from Step 2, when the model does not meet the performance criteria, the development that follows it may not be guided. The architecture is transformed (by removing learnable weights) randomly and re-trained. With our proposed framework, we tend to model the architecture search-space and take an informed decision about transforming the architecture that is aided by deliberate platform characteristics.  We have shown that an automated policy has a higher accuracy/computation ratio as compared to random architectural choices.

In summary, this framework will let you refine a model architecture for a targeted Android device by an automated policy that does the following:
1. Choosing a model architecture hyper-parameters (ranges are provided by the developer)
2. Building and executing the model on a sample on the connected target android device
3. Collecting execution statistics such as mean runtime, variance in execution, layer-wise data
4. Training the model with the same hyper-parameter
5. If accuracy is recovered or improved, then go back to step 6, else go to step 7
6. Capture the hyper-parameter setting and store it, go to step 1
7. Discard the hyper-parameter and go to step 1

While model architecture hyper-parameters (as defined to be in the design space) are captured and discarded, these become a basis for reinforcing a policy that learns to take correct action further on. Reinforcement learning paradigm is used to create such a policy that learn how to remove and what layer to remove from the model, and iteratively improves itself until convergence.

Once the convergence point is reached, the user can stop the procedure manually.

![arch][system]

## How to use this framework for your own projects?
> Right now, this is not a finished work, things are needed to be stitched together and APIs are yet to be designed. However, if interested, read the docs in the module file, and go through the examples, and probably you will be able to use it in your project.

In short, connect your target device, define the design space of the architecture around which you probably want to experiment on, feed the original training data, and let your system do the work.

The documents will be updated soon.

## Dependencies (Tested on Ubuntu 16.04 LTS)

Install the following dependencies first in the given sequence (recommended)
* python-adb
    * libssl-dev (apt-get)
    * rsa (pip)
    * PycryptoDome (pip)
    * python-m2crypto (apt-get)
    * adb (pip)
* plotly (pip)
* tensorflow (pip)
* csv

### Step-by-step procedure
```
First, check the required dependencies for this project to run on
your system, then check if your target android device is properly detected.

Connect your Android device to work-station and try running test.py
This project uses py-adb, so probably you might need to execute
adb kill-server first. Once correctly ran, it will produce pie-charts
of the performance of three widely used model with different parameters settings.
```

```
Next, read through the mobilenet_v1.py doc in order to understand the format
in which your architecture should be declared. Create a similar file your_model.py
```

```
Read through the data_provider_mnist.py doc to understand the format in which you
need to present your data to the framework. Create a similar file your_data.py
```

```
You can choose to go ahead either with DQN RL algorithms for learning policy
to fine-tune your model or change the algorithms and setting. In either case,
 policy.py is the file you should look into.
```

```
Follow the main.py file doc and create a similar my_main.py replicating all of the
 code and replacing mobilenet_v1 and mnist imports with your model and data imports.
```

[system]: res/sys.png
