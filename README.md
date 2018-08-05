# xcelerator

Xcelerator is a python based framework that learns to fine tune your best performing hand tuned architecutre, and make it even more faster by taking advantage of implicit performace characteristics of your targeted android device.

## Why this framework works?
You have probably developed an architecure and tested it multiple times on desktop environemt. Once you pass your accuracy criteria, you have probably tested the performance of your model on an android device. Having not meeting the performance criteria, you would have probably lowered your expectations and decreased the accuracy by stripping out some layers, more or less randomly.
You have missed the point that your targeted device is probably good at executing one of the other kernel (may be of different size or with different layers). And instead of randomly removing anything from anywhere in the model, you could have fine-tuned it more for the targeted device while remaining in your desing space of the modelling of architecture. Manually doing so with no intuition is a very bad idea. So, why not automate it and learn the policy that can fine tune model without any loss in accuray?

## How does this framework works?
This let you to refine you model for a targeted andgroid device by automating the procedure of:
1. Choosing a model architecture hyper-parameters (ranges are provided by model designer)
2. Building and executing the model on a sample on the connected target android device
3. Collecting execution statistics such as mean run time, variance in execution, layer-wise data
4. Training the model with the same hyper-parameter
5. If accuracy is recovered or improved, then go back to step 6, else go to step 7
6. Capture the hyper-parameter setting and store it, go to step 1
7. Discard the hyper-parameter and go to step 1

While model architecture hyper-parameters (as defined to be in the desing space) are captured and discarded, these becomes a basis for reinforcing a policy that learn to take correct action further on. Reinforcement learning paradigm is used to create such a policy that learn how to remove and what layer to remove from the model, and iteratively improves itself untill convergence.

Once both convergence point is reached, user can stop the procedure manually.
![arch][system]

## How to use this for your project?
> Right now, this is not a finished work, things are neede to be stiched together and APIs are yet to be designed. However, if interested, read the docs in module file, and go through the examples, and probably you will be able to use it in your project.

In short, connect your target device, define the design space of the architecure around which you probably want to experiemnt on, feed the original training data, and let your system do the work.

More updates on this part later.

## Dependencies

Install the following dependencies first in the given sequence (recommended)
* python-adb [To connect with your target android device]
    * libssl-dev (apt-get)
    * rsa (pip)
    * PycryptoDome (pip)
    * python-m2crypto (apt-get)
    * adb (pip)
* plotly (pip) [For graphing the tuning statistics]
* tensorflow (pip) [Based on tensorflow and tflite]
* csv [record the data, used for learning policy]

### Step-by-step procedure
```
First check the required dependencies for this project to run on
your system, then check if your target android device is properly detected.

Connect your android device to work-station and try running test.py
This project uses py-adb, so probably you might need to execute
adb kill-server first. Once correctly ran, it will produce pie-charts
of performance of three widely used model with different parameters settings.
```

```
Next, read through the mobilenet_v1.py doc in order to understand the format
in which your architecure should be declared. Create a similar file your_model.py
```

```
Read through the data_provider_mnist.py doc to understant the format in which you
need to present your data to the framework. Create a similar file your_data.py
```

```
You can choose to go ahead either with DQN RL algorithms for learninga policy
to fine tune your model, or change the algorithms and setting. In either case,
 policy.py is the file you should look into.
```

```
Follow the main.py file doc and create a similar my_main.py replicating all of the
 code and replacing mobilenet_v1 and mnist imports with your model and data imports.
```

### Contributions
As of now, this is not open for any contribution.

[system]: res/sys.png
