"""This file implements Deep CFR, as described in Brown et al. - Deep Counterfactual Regret Minimization
(2019).
"""

import typing

import tensorflow as tf

from rlpoker import cfr
from rlpoker.cfr_game import (
    get_available_actions, sample_chance_action, is_terminal, payoffs, which_player)
from rlpoker.util import sample_action
from rlpoker import buffer
from rlpoker import extensive_game
from rlpoker import neural_game


class RegretPredictor:

    """
    A RegretPredictor can be used inside cfr_traverse.
    """

    def predict_advantages(self, info_set_vector, action_indexer: neural_game.ActionIndexer):
        """
        Predicts advantages for each action available in the information set.

        Args:
            info_set_vector: ndarray.
            action_indexer: ActionIndexer. The mapping from actions to indices.

        Returns:
            ActionFloat. The predicted regret for each action in the information set.
        """
        raise NotImplementedError("Not implemented in the base class.")

    def compute_action_probs(self, info_set_vector, action_indexer: neural_game.ActionIndexer):
        """
        Compute action probabilities in this information set.

        Args:
            info_set_vector: ndarray.

        Returns:
            ActionFloat. The action probabilities in this information set.
        """
        action_advantages = self.predict_advantages(info_set_vector, action_indexer)
        return cfr.compute_regret_matching(action_advantages)

    def train(self, batch: typing.List[extensive_game.InformationSetAdvantages]):
        """
        Train on the given batch of InformationSetAdvantages.

        Args:
            batch: List[InformationSetAdvantages].

        Returns:
            - float. The loss for training on the batch.
        """
        pass


class DeepRegretNetwork(RegretPredictor):
    
    def __init__(self, state_dim: int, action_indexer: neural_game.ActionIndexer, player: int):
        """
        A DeepRegretNetwork uses a neural network to predict advantages for actions in information sets.

        Args:
            state_dim: int. The dimension of the state vector.
            action_indexer: ActionIndexer.
            player: int. The player number it represents. Used for scoping.
        """
        self.state_dim = state_dim
        self.action_indexer = action_indexer
        self.player = player

        self.scope = 'regret_network_{}'.format(self.player)
        self.tensors, self.init_op = self.build(self.state_dim, self.action_indexer.get_action_dim(), self.scope)

    def initialise(self, sess):
        """
        Initialise the weights of the network, using the tensorflow session.

        Args:
            sess: tensorflow session.
        """
        sess.run(self.init_op)

    @staticmethod
    def build(state_dim, action_dim, scope):
        with tf.variable_scope(scope):
            input_layer = tf.placeholder(tf.float32, shape=[None, state_dim], name='state')

            hidden = tf.layers.dense(input_layer, 10, activation=tf.nn.relu)

            advantages = tf.layers.dense(hidden, action_dim)

        tensors = {
            'input_layer': input_layer,
            'advantages': advantages
        }

        init_op = tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))

        return tensors, init_op

    def predict_advantages(self, sess, info_set_vector):
        
        return sess.run(self.tensors['advantages'], feed_dict={
            self.tensors['input_layer']: [info_set_vector]
        })


def deep_cfr(game: neural_game.NeuralGame, num_iters: int=100, num_traversals: int=50,
             advantage_maxlen: int=1000000, strategy_maxlen: int=1000000):
    """
    Args:
        game:
        num_iters: int. The number of iterations to run deep CFR for.
        num_traversals: int. The number of traversals per CFR iteration.
        advantage_maxlen: int. The maximum length of the advantage memories.
        strategy_maxlen: int. The maximum length of the strategy memory.

    Returns:
        strategy, exploitability.
    """

    advantage_memories = {
        1: buffer.Reservoir(maxlen=advantage_maxlen),
        2: buffer.Reservoir(maxlen=advantage_maxlen)
    }
    strategy_memory = buffer.Reservoir(maxlen=strategy_maxlen)

    action_indexer = game.action_indexer

    with tf.Session() as sess:
        network1 = DeepRegretNetwork(game.state_dim, action_indexer, 1)
        network2 = DeepRegretNetwork(game.state_dim, game.action_dim, 2)

        network1.initialise(sess)
        network2.initialise(sess)

        # Iterate over players and do cfr traversals.
        for t in range(num_iters):
            for player in [1, 2]:
                for i in range(num_traversals):
                    cfr_traverse(game, game.root, player, network1, network2, advantage_memories,
                                 strategy_memory, t, action_indexer)



def cfr_traverse(game, node, player: int, network1: RegretPredictor, network2: RegretPredictor,
                 advantage_memories: typing.Dict[int, buffer.Reservoir], strategy_memory: buffer.Reservoir,
                 t: int, action_indexer: neural_game.ActionIndexer):
    if is_terminal(node):
        return payoffs(node)[player]
    elif which_player(node) == 0:
        # Chance player
        a = sample_chance_action(node)
        return cfr_traverse(game, node.children[a], player, network1, network2,
                            advantage_memories, strategy_memory, t, action_indexer)
    elif which_player(node) == player:
        # It's the traversing player's turn.
        state_vector = game.get_state_vector_for_node(node)
        values = dict()
        for action in get_available_actions(node):
            child = node.children[action]
            values[action] = cfr_traverse(game, child, player, network1, network2,
                                          advantage_memories, strategy_memory, t, action_indexer)
        regrets = dict()

        # Compute the player's strategy
        network = network1 if player == 1 else network2
        strategy = network.compute_action_probs(state_vector, action_indexer)
        average_regret = sum([strategy[action] * values[action] for action in get_available_actions(node)])
        for action in get_available_actions(node):
            regrets[action] = values[action] - average_regret

        info_set_id = game.info_set_ids[node]
        advantage_memory.append((info_set_id, regrets))
    else:
        # It's the other player's turn.
        state_vector = game.get_state_vector_for_node(node)

        # Compute the other player's strategy
        other_player = 1 if player == 2 else 2
        network = network1 if other_player == 1 else network2
        strategy = network.compute_action_probs(state_vector, action_indexer)

        info_set_id = game.info_set_ids[node]
        strategy_memory.append((info_set_id, strategy))

        action = sample_action(strategy)
        return cfr_traverse(game, node.children[action], player, network1, network2,
                            advantage_memory, strategy_memory, t, action_indexer)
