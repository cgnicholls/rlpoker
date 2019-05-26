import tensorflow as tf
from collections import deque, namedtuple
import random
import numpy as np

from rlpoker.buffer import Reservoir, CircularBuffer


NetSizes = namedtuple('NetSizes', ['num_q_hidden', 'q_dim', 'num_policy_hidden', 'policy_dim'])


class Agent:
    def __init__(self, name, input_dim, action_dim, max_replay=200000,
                 max_supervised=1000000, best_response_lr=0.1,
                 supervised_lr=0.005, net_sizes=NetSizes(1, 64, 1, 64)):
        # Replay memory is a circular buffer, and supervised learning memory is a reservoir.
        self.replay_memory = CircularBuffer(max_replay)
        self.supervised_memory = Reservoir(max_supervised)

        self.input_dim = input_dim
        self.action_dim = action_dim

        self.name = name
        with tf.variable_scope('agent_{}'.format(name)):
            self.q_network = self.create_q_network('current_q', input_dim,
                    action_dim, num_hidden=net_sizes.num_q_hidden,
                    hidden_dim=net_sizes.q_dim)
            self.target_q_network = self.create_q_network('target_q',
                    input_dim, action_dim, num_hidden=net_sizes.num_q_hidden,
                    hidden_dim=net_sizes.q_dim)
            self.policy_network = self.create_policy_network('policy',
                    input_dim, action_dim,
                    num_hidden=net_sizes.num_policy_hidden,
                    hidden_dim=net_sizes.policy_dim)

            # Create ops for copying current network to target network. We create a list
            # of the variables in both networks and then create an assign operation that
            # copies the value in the current variable to the corresponding target variable.
            current_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                scope='agent_{}/current_q'.format(self.name))
            target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                scope='agent_{}/target_q'.format(self.name))
            self.update_ops = [t.assign(c) for t, c in zip(target_vars, current_vars)]

            # Set up Q-learning loss functions
            self.reward = tf.placeholder(tf.float32, shape=[None])
            self.action = tf.placeholder(tf.int32, shape=[None])
            one_hot_action = tf.one_hot(self.action, action_dim)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='agent_{}/current_q'.format(self.name))

            q_value = tf.reduce_sum(one_hot_action * self.q_network['output'], axis=1)
            self.not_terminals = tf.placeholder(tf.float32, shape=[None])
            next_q = self.reward + self.not_terminals * tf.reduce_max(tf.stop_gradient(
                self.target_q_network['output']), axis=1)
            self.q_loss = tf.reduce_mean(tf.square(next_q - q_value))
            with tf.control_dependencies(update_ops):
                self.q_trainer = tf.train.GradientDescentOptimizer(best_response_lr).minimize(self.q_loss)

            policy_for_actions = tf.reduce_sum(self.policy_network['output'] * one_hot_action, axis=1)
            self.policy_loss = tf.reduce_mean(-tf.log(policy_for_actions))
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='agent_{}/policy'.format(self.name))
            with tf.control_dependencies(update_ops):
                self.policy_trainer = tf.train.GradientDescentOptimizer(supervised_lr).minimize(self.policy_loss)

    def append_replay_memory(self, transitions):
        for transition in transitions:
            self.replay_memory.append(transition)

    def append_supervised_memory(self, state_action_pairs):
        for state_action_pair in state_action_pairs:
            self.supervised_memory.append(state_action_pair)

    # Get the output of the q network for the given state
    def predict_q(self, sess, state):
        assert len(state.shape) == 2
        assert state.shape[1] == self.input_dim
        return sess.run(self.q_network['output'], feed_dict={
            self.q_network['input']: state
        })

    # Get the output of the q network for the given state
    def predict_policy(self, sess, state):
        assert len(state.shape) == 2
        assert state.shape[1] == self.input_dim
        return sess.run(self.policy_network['output'], feed_dict={
            self.policy_network['input']: state
        })

    def update_target_network(self, sess):
        # Copy current q_network parameters to target_q_network
        sess.run(self.update_ops)

    def train_q_network(self, sess, batch_size):
        # Sample a minibatch from the replay memory
        minibatch = self.replay_memory.sample(batch_size)

        states = np.array([d['state'] for d in minibatch])
        actions = np.array([d['action'] for d in minibatch])
        next_states = np.array([d['next_state'] for d in minibatch])
        rewards = np.array([d['reward'] for d in minibatch])
        terminals = np.array([d['terminal'] for d in minibatch])

        not_terminals = np.array([not t for t in terminals]).astype('float32')

        q_loss, _ = sess.run([self.q_loss, self.q_trainer], feed_dict={
            self.reward: rewards,
            self.action: actions,
            self.not_terminals: not_terminals,
            self.q_network['input']: states,
            self.target_q_network['input']: next_states
        })
        return q_loss

    def train_policy_network(self, sess, batch_size):
        # Sample a minibatch from the supervised memory
        minibatch = self.supervised_memory.sample(batch_size)

        states = np.array([d['state'] for d in minibatch])
        actions = np.array([d['action'] for d in minibatch])

        policy_loss, _ = sess.run([self.policy_loss, self.policy_trainer], feed_dict={
            self.policy_network['input']: states,
            self.action: actions
        })
        return policy_loss

    # Create a 2 layer neural network with relu activations on the hidden
    # layer. The output is the predicted q-value of an action.
    def create_q_network(self, scope, input_dim, action_dim, num_hidden=1, hidden_dim=64):
        with tf.variable_scope(scope):
            input_layer = tf.placeholder(tf.float32, shape=[None, input_dim], name='input')
            training = tf.placeholder(tf.bool, name='training')

            hidden_layer = input_layer

            for i in range(num_hidden):
                hidden_layer = tf.layers.dense(hidden_layer, hidden_dim, activation=tf.nn.relu)
                hidden_layer = tf.layers.dropout(hidden_layer, dropout_rate, training=training)
                hidden_layer = tf.layers.batch_normalization(hidden_layer, axis=-1, training=training)

            output_layer = tf.layers.dense(hidden_layer, action_dim)
        return {'input': input_layer, 'output': output_layer, 'training': training}

    def create_policy_network(self, scope, input_dim, action_dim, num_hidden=1, hidden_dim=64):
        with tf.variable_scope(scope):
            input_layer = tf.placeholder(tf.float32, shape=[None, input_dim])
            training = tf.placeholder(tf.bool, name='training')

            hidden_layer = input_layer
            for i in range(num_hidden):
                hidden_layer = tf.layers.dense(hidden_layer, hidden_dim, activation=tf.nn.relu)
                hidden_layer = tf.layers.dropout(hidden_layer, dropout_rate, training=training)
                hidden_layer = tf.layers.batch_normalization(hidden_layer, axis=-1, training=training)

            output_layer = tf.layers.dense(hidden_layer, action_dim, activation=tf.nn.softmax)
        return {'input': input_layer, 'output': output_layer, 'training': training}

    def get_strategy(self, sess, states):
        """Returns a strategy for an agent. This is a mapping from
        information sets in the game to probability distributions over
        actions.

        Args:
            sess: tensorflow session.
            states: dict. This is a dictionary with keys the information set
                ids and values the vectors to input to the network.
        """
        strategy = dict()
        for info_set_id, state in states.items():
            policy = self.predict_policy(sess, np.array([state])).ravel()
            strategy[info_set_id] = {i: policy[i] for i in range(len(policy))}

        return strategy
