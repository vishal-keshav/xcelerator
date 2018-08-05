"""
Policy learning module is supposed to do the following:
    * Encapsulate RL environment
    * Define RL learning policy
    * Make policy

Encapsulate RL environment: This provides a generic environment
consisting of observation and action. Observations and action
are inputs that generate a new action, on which system reacts
to give more observation. These constructs are kept very general
in order to for extensibility of use case.

Define RL learning policy: A learning policy is an algorithm
for observation environment, taking action and updating policy
for taking action upon observing new environment. Algorithms such
as DQN, DDPG can be easily implemented.

Make Policy: Policy making is basically updating the RL algorithm
architecture parameter. Once updated and converged, policy are written
back to the disc.

This module is GPLv3 licensed.
"""
import tensorflow as tf
import os
import numpy as np
import trainer as tr
import explorer as ex
# Environment class
class environment:
    """
    Both observation and action are coded as vector
    Obdervation is the network state and action is change in state
    Rewards are scalar (accuracy+ 1/floating point operations)
    """
    def __init__(self, name, init_param):
        self.env_name = name
        self.observation = init_param['init_observation']
        self.action = init_param['init_action']

    def set_simulation(self, fn):
        self.simulate = fn

    def reset(self):
        """
		Reset the environment and returns array the observations
        """
        pass

    def get_observation_shape(self):
        return len(self.observation)

    def take_action(self, action):
        """
		Given an action, step up the environment
        """
        self.action = action
        observation, reward = self.simulate(self.action)
        return observation, reward

    def get_action_space(self):
        return len(self.action)

"""
Policy graph architecture represents what architecture
is used to learn the policy itself.
"""
class policy_graph_arch:
    def __init__(self):
        pass

    def build_graph(self, observation):
        layer = tf.layers.dense(inputs=observation, units = 1024, activation = tf.nn.relu)
        layer = tf.layers.dense(inputs=layer, units = 128, activation = tf.nn.relu)
        layer = tf.layers.dense(inputs=layer, units = 6)
        return layer

"""
Learning policy implements RL algorithm,
below class implements DQN.
"""
class learning_policy:
    def __init__(self, name, env, param):
        self.policy_name = name
        self.learning_rate = param['learning_rate']
        self.gamma = param['gamma']
        self.env = env
        self.mem_index = 0

    def set_param(self, param):
        self.learning_rate = param['learning_rate']
        self.gamma = param['gamma']

    def initialize(self, input_dim, output_dim):
        print("Initializing Q networks...")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.observation = tf.placeholder(tf.float32, self.input_dim)
        self.reward = tf.placeholder(tf.float32, [None, ])
        self.action = tf.placeholder(tf.int32, [None, ])
        self.next_observation = tf.placeholder(tf.float32, self.input_dim)
        #Episodic memory initialization, take 8 moves at a time
        self.observation_memory = np.zeros([8, self.input_dim[0])
        self.next_observation_mem = np.zeros([8, self.input_dim[0])
        self.action_mem = np.zeros([8, ])
        self.reward_mem = np.zeros([8, ])
        policy_graph = policy_graph_arch()
        with tf.variable_scope('local_net'):
            self.local_net = policy_graph.build_graph(self.observation)
        with tf.variable_scope('global_net'):
            self.global_net = policy_graph.build_graph(self.next_observation)

        #Define loss for local network
        self.q_label = tf.stop_gradient(self.reward + self.gamma*tf.reduce_max(self.global_net, axis=1))
        action_index = tf.stack([tf.range(tf.shape(self.action)[0], dtype=tf.int32), self.action], axis=1)
        self.q_prediction =tf.gather_nd(params=self.local_net, indices=action_index)
        self.loss = tf.reduce_mean(tf.squared_difference(self.q_label, self.q_prediction))
        self.train_ops = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        #Define copy constructor for network parameters as tf op that can be executed
        local_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='local_net')
        global_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='global_net')
        self.param_copy_op = [tf.assign(dest, src) for dest, src in zip(global_params, local_params)]
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        print("Q networks initialized.")

    def predict_action(self,s):
        #return self.env.action_space.sample()
        local_q_value = self.sess.run(self.local_net, feed_dict={self.observation: s})
        pred = local_q_value
        print("Action predicted from local network ", pred)
        return pred

    def store_transition(self, s, a, r, _s):
        print("Storing the transition")
        self.observation_memory[self.mem_index] = s
        self.action_mem[self.mem_index] = a
        self.reward_mem[self.mem_index] = r
        self.next_observation_mem[self.mem_index] = _s
        if self.mem_index == 7:
            self.update_local_net(self.observation_memory, self.action_mem,
                            self.reward_mem, self.next_observation_mem)
        self.mem_index = (self.mem_index + 1)%8

    def update_local_net(self, s, a, r, _s):
        print("Updating local net")
        self.sess.run(self.train_ops, feed_dict = {
            self.observation: s,
            self.action: a,
            self.reward: r,
            self.next_observation: _s,
        })
        return

    def update_global_net(self):
        print("Updating global net")
        self.sess.run(self.param_copy_op)
        return

def main():
    init_param = [[],[]] # For now, leave it blanck
    env = environment('test_env', init_param)
    """
    Now define a simulation function that does the following:
        * Takes in action
        * Converts it in new model parameters
        * Constructs the model with new parameters
        * Execute the model on device and note the exec time
        * Train the model to fixed number of iterations
        * Note the accuracy
        * calculate the reward
        * Returns current model parameters and reward
    [TO-DO]
    env.set_simulation(fn)
    """
    #def simulate(action_vector):
    #    reward = 0
    #    observation = current_observation
    #    new_state = convert_action_to_state()
    #    ...

    param = {'learning_rate': 0.001, 'gamma': 0.5}
    learning_step = 32
    input_shape = [env.get_observation_shape()]
    output_shape = [6]
    policy_algo = learning_policy('DQN', env, param)
    policy_algo.initialize()

    for e in range(100):
        #observation = env.reset()
        current_step = 0
        learning_step = 8
        #print(observation.shape)

        while True:
            action = policy_algo.predict_action(observation)

            new_observation, reward = env.take_action(action)

            policy_algo.store_transition(observation, action, reward, new_observation)
            if current_step%learning_step == 0:
                policy_algo.update_global_net()
            if current_step == 1000:
                break
            current_step = current_step + 1
            observation = new_observation

if __name__=='__main__':
    main()
