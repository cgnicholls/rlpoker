import tensorflow as tf
from collections import deque
import random
import numpy as np

class Agent:
    def __init__(self, name, input_dim, max_replay=100000, max_supervised=100000, best_response_lr=1e-2, supervised_lr=1e-3):
        self.replay_memory = deque(maxlen=max_replay)
        self.supervised_memory = deque(maxlen=max_supervised)

        self.name = name

        self.q_network = self.create_q_network('current_q' + name, input_dim)
        self.target_q_network = self.create_q_network('target_q' + name, input_dim)
        self.policy_network = self.create_policy_network('policy' + name, input_dim)

        # Create ops for copying current network to target network. We create a list
        # of the variables in both networks and then create an assign operation that
        # copies the value in the current variable to the corresponding target variable.
        current_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='current_q' + name)
        target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_q' + name)
        self.update_ops = []
        for i in range(len(current_vars)):
            self.update_ops.append(target_vars[i].assign(current_vars[i]))

        # Set up Q-learning loss functions
        self.predicted_next_q = tf.placeholder('float32', shape=[None, 3])
        self.reward = tf.placeholder('float32', shape=[None])
        self.action = tf.placeholder('int32', shape=[None])
        one_hot_action = tf.one_hot(self.action, 3)

        with tf.variable_scope('current_q' + name):
            q_value = tf.reduce_sum(one_hot_action * self.q_network['output'], axis=1)
            self.not_terminals = tf.placeholder('float32', shape=[None])
            next_q = self.reward + self.not_terminals * tf.reduce_max(self.predicted_next_q, axis=1)
            self.q_loss = tf.reduce_mean((next_q - q_value)**2)
            self.q_trainer = tf.train.AdamOptimizer(best_response_lr).minimize(self.q_loss)

        with tf.variable_scope('policy' + name):
            policy_for_actions = tf.reduce_sum(self.policy_network['output'] * one_hot_action, axis=1)
            self.policy_loss = tf.reduce_mean(-tf.log(policy_for_actions))
            self.policy_trainer = tf.train.AdamOptimizer(supervised_lr).minimize(self.policy_loss)

    def append_replay_memory(self, transitions):
        self.replay_memory.extend(transitions)

    def append_supervised_memory(self, state_action_pairs):
        self.supervised_memory.extend(state_action_pairs)

    # Get the output of the q network for the given state
    def predict_q(self, sess, state):
        return sess.run(self.q_network['output'], feed_dict={
            self.q_network['input']: state
        })

    # Get the output of the q network for the given state
    def predict_policy(self, sess, state):
        return sess.run(self.policy_network['output'], feed_dict={
            self.policy_network['input']: state
        })

    def update_target_network(self, sess):
        # Copy current q_network parameters to target_q_network
        sess.run(self.update_ops)

    def train_q_network(self, sess, batch_size):
        # Sample a minibatch from the replay memory
        minibatch = random.sample(self.replay_memory, batch_size)

        states = [d['state'] for d in minibatch]
        actions = [d['action'] for d in minibatch]
        next_states = [d['next_state'] for d in minibatch]
        rewards = [d['reward'] for d in minibatch]
        terminals = [d['terminal'] for d in minibatch]

        not_terminals = [not t for t in terminals]

        predicted_next_q = sess.run(self.target_q_network['output'], feed_dict={
            self.target_q_network['input']: next_states
        })

        q_loss, _ = sess.run([self.q_loss, self.q_trainer], feed_dict={
            self.reward: rewards,
            self.action: actions,
            self.predicted_next_q: predicted_next_q,
            self.not_terminals: np.array(not_terminals).astype('float32').ravel(),
            self.q_network['input']: states
        })
        return q_loss

    def train_policy_network(self, sess, batch_size):
        # Sample a minibatch from the supervised memory
        minibatch = random.sample(self.supervised_memory, batch_size)

        states = [d['state'] for d in minibatch]
        actions = [d['action'] for d in minibatch]

        policy_loss, _ = sess.run([self.policy_loss, self.policy_trainer], feed_dict={
            self.policy_network['input']: states,
            self.action: actions
        })
        return policy_loss

    # Create a 2 layer neural network with relu activations on the hidden layer. The output is the predicted q-value of an action.
    def create_q_network(self, scope, input_dim, num_hidden=2, hidden_dim=20, l2_reg=1e-3):
        with tf.variable_scope(scope):
            input_layer = tf.placeholder('float32', shape=[None, input_dim])

            hidden_layer = input_layer

            for i in range(num_hidden):
                hidden_layer = tf.contrib.layers.fully_connected(hidden_layer, num_outputs=hidden_dim,
                activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(),
                biases_initializer=tf.zeros_initializer(),
                weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))

            output_layer = tf.contrib.layers.fully_connected(hidden_layer, num_outputs=3,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=tf.zeros_initializer(),
            weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
        return {'input': input_layer, 'output': output_layer}

    def create_policy_network(self, scope, input_dim, num_hidden=2, hidden_dim=10, l2_reg=1e-3):
        with tf.variable_scope(scope):
            input_layer = tf.placeholder('float32', shape=[None, input_dim])

            hidden_layer = input_layer
            for i in range(num_hidden):
                hidden_layer = tf.contrib.layers.fully_connected(hidden_layer, num_outputs=num_hidden,
                activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(),
                biases_initializer=tf.zeros_initializer(),
                weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))

            output_layer = tf.contrib.layers.fully_connected(hidden_layer, num_outputs=3,
            activation_fn=tf.nn.softmax, weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=tf.zeros_initializer(),
            weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
        return {'input': input_layer, 'output': output_layer}
