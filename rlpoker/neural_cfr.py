"""The purpose of this program is to train a neural network to approximate the solution that CFR converges to.

1. First run CFR to converge to a solution.
2. Then train a neural network to approximate this solution.
3. Then measure the exploitability of this network's strategy.
"""
import argparse
import numpy as np
import random
import tensorflow as tf

from rlpoker.best_response import compute_exploitability
from rlpoker.cfr import cfr
from rlpoker.games.leduc import Leduc, LeducNFSP
from rlpoker.games.card import get_deck

def build_network(input_dim, action_dim, layer_dims, dropout_rate=0.3):
    with tf.variable_scope('network'):
        input_layer = tf.placeholder(tf.float32, shape=[None, input_dim])
        training = tf.placeholder(tf.bool, name='training')

        hidden_layer = input_layer
        for i, layer_dim in enumerate(layer_dims):
            hidden_layer = tf.layers.dense(hidden_layer, layer_dim, activation=tf.nn.relu)
            if dropout_rate:
                hidden_layer = tf.layers.dropout(hidden_layer, dropout_rate, training=training)
            hidden_layer = tf.layers.batch_normalization(hidden_layer, axis=-1, training=training)

        logits = tf.layers.dense(hidden_layer, action_dim, activation=None)
        probs = tf.nn.softmax(logits)

    return {'input_layer': input_layer,
            'logits': logits,
            'probs': probs,
            'training': training}

def compute_network_strategy(sess, network, state_vectors):
    network_strategy = {}
    for state, vector in state_vectors.items():
         probs = sess.run(network['probs'],
            feed_dict={
                network['input_layer']: np.array([vector]),
                network['training']: False
            })
         probs = probs.ravel()
         network_strategy[state] = {action: probs[action] for action in range(action_dim)}
    
    return network_strategy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfr_iters', default=100, type=int,
                        help='The number of iterations to run CFR for.')
    parser.add_argument('--num_values', default=3, type=int,
                        help='The number of cards to use in Leduc.')
    parser.add_argument('--num_suits', default=2, type=int,
                        help='The number of suits to use in Leduc.')
    parser.add_argument('--use_chance_sampling', action='store_true',
                        help='Whether or not to use chance sampling. Defaults to False.')
    parser.add_argument('--num_epochs', default=10000,
                        help='The number of epochs to train the neural network for.')
    parser.add_argument('--learning_rate', default=1e-4,
                        help='The number of epochs to train the neural network for.')
    parser.add_argument('--dropout_rate', default=None,
                        help='The dropout rate to use.')
    args = parser.parse_args()

    dropout_rate = None
    if args.dropout_rate:
        dropout_rate = float(args.dropout_rate)

    cards = get_deck(num_values=args.num_values, num_suits=args.num_suits)
    game = Leduc(cards)

    strategy, exploitabilities = cfr(game, num_iters=args.cfr_iters,
        use_chance_sampling=args.use_chance_sampling)

    exploitability = compute_exploitability(game, strategy)
    print("Exploitability of final strategy: {}".format(exploitability))

    leduc_nfsp = LeducNFSP(cards)
    state_vectors = leduc_nfsp._state_vectors
    state_dim = leduc_nfsp.state_dim
    action_dim = leduc_nfsp.action_dim

    # Now build a network.
    layer_dims = [64, 64, 64]
    network = build_network(state_dim, action_dim, layer_dims, dropout_rate=dropout_rate)

    states = list(strategy.keys())
    xs = np.array([state_vectors[state] for state in states])
    action_probs = np.array([[strategy[state].get(action, 0.0) for action in range(action_dim)]
                           for state in states])

    assert xs.shape == (len(states), state_dim)
    assert action_probs.shape == (len(states), action_dim)
    
    labels = tf.placeholder(tf.float32, shape=[None, action_dim], name='labels')
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=network['logits'])

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer(args.learning_rate).minimize(loss)

    batch_size = 32

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i_epoch in range(args.num_epochs):
        # Now get the state vectors and train to predict the strategy.
        indices = list(range(len(states)))
        random.shuffle(indices)
        index = 0

        losses = []

        while index + batch_size < len(states):
            batch_indices = indices[index:index+batch_size]
            batch_xs = xs[batch_indices, :]
            batch_ys = action_probs[batch_indices, :]

            # Train on the batch
            _, loss_i = sess.run([train_op, loss],
                feed_dict={
                    network['input_layer']: batch_xs,
                    labels: batch_ys,
                    network['training']: True
                })
    
            losses.append(loss_i)

            index += batch_size

        if i_epoch % 100 == 0:
            print("End of epoch: {}, mean loss: {}".format(i_epoch, np.mean(losses)))
            network_strategy = compute_network_strategy(sess, network, state_vectors)
            network_exploitability = compute_exploitability(game, network_strategy)
            print("Exploitability of network's strategy: {}".format(network_exploitability))
