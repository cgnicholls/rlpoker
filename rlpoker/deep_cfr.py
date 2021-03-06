"""This file implements Deep CFR, as described in Brown et al. - Deep Counterfactual Regret Minimization
(2019).
"""

import collections
import os
import time
import typing

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import rlpoker.cfr_util
from rlpoker import cfr
from rlpoker.cfr_game import (
    get_available_actions, sample_chance_action, is_terminal, payoffs, which_player)
from rlpoker.util import sample_action
from rlpoker import buffer
from rlpoker import extensive_game
from rlpoker import best_response
from rlpoker import neural_game
from rlpoker import util


StrategyMemoryElement = collections.namedtuple('StrategyMemoryElement', [
    'info_set_id', 't', 'info_set_strategy'
])

AdvantageMemoryElement = collections.namedtuple('AdvantageMemoryElement', [
    'info_set_id', 't', 'info_set_advantages'
])


class RegretPredictor:

    """
    A RegretPredictor can be used inside cfr_traverse.
    """

    def predict_advantages(self, info_set_vector, action_indexer: neural_game.ActionIndexer) -> \
            extensive_game.ActionFloat:
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
            action_indexer: ActionIndexer.

        Returns:
            ActionFloat. The action probabilities in this information set.
        """
        action_advantages = self.predict_advantages(info_set_vector, action_indexer)
        return rlpoker.cfr_util.compute_regret_matching(action_advantages, highest_regret=True)

    def train(self, batch: typing.List[AdvantageMemoryElement],
              action_indexer: neural_game.ActionIndexer,
              info_set_vectoriser: neural_game.InfoSetVectoriser,
              current_time: int):
        """Train on one batch of AdvantageMemoryElements.

        Args:
            batch: list of AdvantageMemoryElement objects.
            action_indexer: ActionIndexer. Turns actions into indices.
            info_set_vectoriser: InfoSetVectoriser. Turns info set ids into vectors.
            current_time: int. The current iteration we are training on.

        Returns:
            loss: float.
        """
        pass


class DeepRegretNetwork(RegretPredictor):
    
    def __init__(self, state_shape: typing.Tuple[int], action_indexer: neural_game.ActionIndexer, player: int):
        """
        A DeepRegretNetwork uses a neural network to predict advantages for actions in information sets.

        Args:
            state_dim: int. The dimension of the state vector.
            action_indexer: ActionIndexer.
            player: int. The player number it represents. Used for scoping.
        """
        self.state_shape = state_shape
        self.action_indexer = action_indexer
        self.player = player

        self.sess = None

        self.scope = 'regret_network_{}'.format(self.player)
        self.tensors, self.init_op = self.build(self.state_shape, self.action_indexer.action_dim, self.scope)

        self.global_step = 0

    def initialise(self):
        """
        Initialise the weights of the network, using the tensorflow session self.sess.
        """
        self.sess.run(self.init_op)

    @staticmethod
    def build(state_shape, action_dim, scope, hidden_dim=64):
        with tf.variable_scope(scope):
            input_layer = tf.placeholder(tf.float32, shape=(None,) + state_shape, name='state')

            # For now, we flatten so that we can accept any state shape.
            hidden = tf.layers.flatten(input_layer, name='flatten')
            hidden = tf.layers.dense(hidden, hidden_dim, activation=tf.nn.relu)
            hidden = tf.layers.dense(hidden, hidden_dim, activation=tf.nn.relu)

            advantages = tf.layers.dense(hidden, action_dim)

            info_set_advantages = tf.placeholder(tf.float32, shape=(None, action_dim), name='info_set_advantages')
            times = tf.placeholder(tf.float32, shape=(None, 1), name='times')
            current_time = tf.placeholder(tf.float32, shape=(), name='current_time')

            regrets = tf.reduce_sum((info_set_advantages - advantages)**2, axis=1, name='regrets')

            loss = tf.reduce_mean(times * regrets) / current_time

            summary = tf.summary.scalar('loss', loss)

            train_op = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

        tensors = {
            'input_layer': input_layer,
            'advantages': advantages,
            'train_op': train_op,
            'loss': loss,
            'times': times,
            'current_time': current_time,
            'info_set_advantages': info_set_advantages,
            'summary': summary
        }

        init_op = tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))

        return tensors, init_op

    def set_sess(self, sess: tf.Session):
        self.sess = sess

    def predict_advantages(self, info_set_vector, action_indexer: neural_game.ActionIndexer) -> \
            extensive_game.ActionFloat:
        advantages = self.sess.run(self.tensors['advantages'], feed_dict={
            self.tensors['input_layer']: [info_set_vector]
        })

        return extensive_game.ActionFloat({
            action: advantages[0, self.action_indexer.get_index(action)] for action in self.action_indexer.actions
        })

    def train(self, batch: typing.List[AdvantageMemoryElement],
              action_indexer: neural_game.ActionIndexer,
              info_set_vectoriser: neural_game.InfoSetVectoriser,
              current_time: int):
        """Train on one batch of AdvantageMemoryElements.

        Args:
            batch: list of AdvantageMemoryElement objects.
            action_indexer: ActionIndexer. Turns actions into indices.
            info_set_vectoriser: InfoSetVectoriser. Turns info set ids into vectors.
            current_time: int. The current iteration we are training on.

        Returns:
            loss: float.
            summary: tensorflow summary.
        """
        # Each batch is an AdvantageMemoryElement.
        info_set_vectors = [info_set_vectoriser.get_vector(element.info_set_id) for element in batch]
        times = [element.t for element in batch]
        info_set_advantages = [info_set_advantages_to_vector(action_indexer, element.info_set_advantages)
                               for element in batch]

        _, computed_loss, summary = self.sess.run(
            [self.tensors['train_op'], self.tensors['loss'], self.tensors['summary']],
            feed_dict={
                self.tensors['input_layer']: np.array(info_set_vectors),
                self.tensors['times']: np.array(times).reshape(-1, 1),
                self.tensors['current_time']: current_time,
                self.tensors['info_set_advantages']: np.array(info_set_advantages).reshape(-1, action_indexer.action_dim)
        })

        self.global_step += 1

        return computed_loss, summary


def info_set_advantages_to_vector(action_indexer: neural_game.ActionIndexer,
                                  info_set_advantages: typing.Dict[typing.Any, float]):
    """

    Args:
        action_indexer:
        info_set_advantages: dict mapping actions to advantages.

    Returns:
        vector with the advantage for each action in the correct index.
    """
    advantages = np.zeros(action_indexer.action_dim)
    for action, advantage in info_set_advantages.items():
        advantages[action_indexer.get_index(action)] = advantage

    return advantages


def early_stopping(losses: typing.List[float], consecutive_increases: int=2):
    """Returns True if and only if losses[-consecutive_increases-1:] is monotonically increasing.

    Args:
        losses: list of floats. The losses.
        consecutive_increases: int. The number of consecutive increases to see before early stopping.

    Returns:
        early_stop: bool. True if and only if we should early stop.
    """
    # Can't early stop before we see enough losses.
    if len(losses) <= consecutive_increases:
        return False

    relevant_losses = losses[-consecutive_increases-1:]
    return sorted(relevant_losses) == relevant_losses


def early_stopping_water_mark(losses: typing.List[float], num_attempts: int=5):
    """Returns True if and only if the loss has failed to beat the low water mark in num_attempts 
    attempts.

    Args:
        losses: list of floats. The losses.
        num_attempts: int. The number of attempts to beat the low water mark.

    Returns:
        early_stop: bool. True if and only if we should early stop.
    """
    # Can't early stop before we see enough losses.
    if len(losses) <= num_attempts:
        return False

    return min(losses[-num_attempts:]) > min(losses)


def train_network(network: DeepRegretNetwork, advantage_memory: buffer.Reservoir,
                  action_indexer: neural_game.ActionIndexer,
                  info_set_vectoriser: neural_game.InfoSetVectoriser,
                  current_time: int,
                  writer: tf.summary.FileWriter,
                  batch_size=1024, num_sgd_updates=4000):
    """Trains the given network from scratch

    Args:
        network: DeepRegretNetwork. The network to train.
        advantage_memory: Reservoir. Each entry should be an AdvantageMemoryElement.
        action_indexer: ActionIndexer. Turns actions into indices.
        info_set_vectoriser: InfoSetVectoriser. Turns information set ids into vectors.
        current_time: int. The current time.
        writer: tf.summary.FileWriter.
        batch_size: int. The size to use for each batch.
        num_sgd_updates: int. The number of sgd updates to do.

    Returns:
        mean_loss: float. The mean loss over the period.
    """
    # First reset the network.
    network.initialise()

    losses = []

    print("Training.")
    indices = list(range(len(advantage_memory)))
    for i in tqdm(range(num_sgd_updates)):
        # Shuffle the advantage memory.
        batch_indices = np.random.choice(indices, batch_size, replace=True)

        batch = advantage_memory.get_elements(batch_indices)

        loss, summary = network.train(batch, action_indexer, info_set_vectoriser, current_time=current_time)
        writer.add_summary(summary, network.global_step)
        losses.append(loss)

        # Early stopping.
        if early_stopping_water_mark(losses, num_attempts=20):
            print("Losses: {}".format(losses))
            print("Early stopping.")
            break

    print("Losses % through the data: {}".format(
        [losses[int(frac / 100 * len(losses))] for frac in [0.0, 25.0, 50.0, 75.0, 99.99]]
    ))

    return np.min(losses)


def compute_mean_strategy(strategy_memory: buffer.Reservoir):
    """Returns the mean strategy for each information set, weighted by time.

    Args:
        strategy_memory: Reservoir consisting of StrategyMemoryElement objects.

    Returns:
        Strategy.
    """
    strategies = collections.defaultdict(list)
    for info_set_id, t, info_set_strategy in strategy_memory.buffer:
        strategies[info_set_id].append((t, info_set_strategy))

    return extensive_game.compute_weighted_strategy(strategies)


def deep_cfr(n_game: neural_game.NeuralGame,
             num_iters: int=100, num_traversals: int=10000,
             advantage_maxlen: int=1000000, strategy_maxlen: int=1000000,
             batch_size: int=1024, num_sgd_updates: int=100):
    """
    Args:
        n_game: NeuralGame.
        num_iters: int. The number of iterations to run deep CFR for.
        num_traversals: int. The number of traversals per CFR iteration.
        advantage_maxlen: int. The maximum length of the advantage memories.
        strategy_maxlen: int. The maximum length of the strategy memory.
        batch_size: int. The batch size to use in training.
        num_sgd_updates: int. The number of sgd updates per training.

    Returns:
        strategy, exploitability.
    """
    game, action_indexer, info_set_vectoriser = n_game

    advantage_memory1 = buffer.Reservoir(maxlen=advantage_maxlen)
    advantage_memory2 = buffer.Reservoir(maxlen=advantage_maxlen)
    strategy_memory = buffer.Reservoir(maxlen=strategy_maxlen)

    # Create summary tensors
    valid_summariser = util.TBSummariser(['exploitability'])

    time_str = time.strftime("%Y-%m-%d-%H:%M:%S", time.gmtime())
    save_path = os.path.join('experiments', time_str)

    if not os.path.exists(save_path):
        print("Path doesn't exist, so creating: {}".format(save_path))
        os.makedirs(save_path)

    log_file = os.path.join(save_path, 'nfsp.log')
    print("Log file {}".format(log_file))

    print("To run tensorboard: tensorboard --logdir {}".format(os.path.join(os.getcwd(), save_path)))

    with tf.Session() as sess:
        network1 = DeepRegretNetwork(info_set_vectoriser.state_shape, action_indexer, 1)
        network1.set_sess(sess)
        network2 = DeepRegretNetwork(info_set_vectoriser.state_shape, action_indexer, 2)
        network2.set_sess(sess)

        network1.initialise()
        network2.initialise()

        tf_train_writer = tf.summary.FileWriter(os.path.join(save_path, 'train'), graph=sess.graph)

        # Iterate over players and do cfr traversals.
        for t in range(1, num_iters + 1):
            print("Iteration t = {}".format(t))
            for player in [1, 2]:
                print("Player: {}".format(player))
                print("Traversing")
                for i in tqdm(range(num_traversals)):
                    cfr_traverse(game, action_indexer, info_set_vectoriser,
                                 game.root, player, network1, network2,
                                 advantage_memory1, advantage_memory2,
                                 strategy_memory, t)

                # Train the traversing player's network on the cfr traversals.
                network = network1 if player == 1 else network2
                network.initialise()
                advantage_memory = advantage_memory1 if player == 1 else advantage_memory2
                mean_loss = train_network(
                    network, advantage_memory, action_indexer, info_set_vectoriser, t,
                    tf_train_writer, batch_size, num_sgd_updates)

                print("Mean loss: {}".format(mean_loss))
                tf_train_writer.flush()

            # print("################")
            #
            # print("----------------")
            # print("Advantage memory 1:")
            # print(advantage_memory1.buffer)
            # print("----------------")
            # print("Advantage memory 2:")
            # print(advantage_memory2.buffer)
            # print("----------------")
            #
            # print("################")
            #

            # print("----------------")
            # print("Predicted advantages:")
            # for info_set_id in set(game.info_set_ids.values()):
            #     print("{}: {}".format(
            #         info_set_id,
            #         network.predict_advantages(info_set_vectoriser.get_vector(info_set_id), action_indexer))
            #     )
            # print("----------------")
            #

            print("Advantage memory 1 length: {}".format(len(advantage_memory1)))
            print("Advantage memory 2 length: {}".format(len(advantage_memory2)))
            print("Strategy memory length: {}".format(len(strategy_memory)))

            mean_strategy = compute_mean_strategy(strategy_memory)
            # print("Strategy summary")
            # print(mean_strategy)
            if game.is_strategy_complete(mean_strategy):
                exploitability = best_response.compute_exploitability(game, mean_strategy)
            else:
                print("Strategy not complete, filling uniformly.")
                exploitability = best_response.compute_exploitability(
                    game,
                    mean_strategy,
                )
            print("Exploitability: {} mbb/h".format(exploitability * 1000))

            valid_summary = valid_summariser.summarise(sess, {'exploitability': exploitability})
            tf_train_writer.add_summary(valid_summary, global_step=t)

    # TODO(chrisn). Train the network on the strategy memory.
    return mean_strategy, exploitability


def cfr_traverse(game: extensive_game.ExtensiveGame, action_indexer: neural_game.ActionIndexer,
                 info_set_vectoriser: neural_game.InfoSetVectoriser,
                 node: extensive_game.ExtensiveGameNode, player: int,
                 network1: RegretPredictor, network2: RegretPredictor,
                 advantage_memory1: buffer.Reservoir, advantage_memory2: buffer.Reservoir,
                 strategy_memory: buffer.Reservoir, t: int):
    """

    Args:
        game: ExtensiveGame.
        action_indexer: ActionIndexer. This maps actions to indices, so that we can use neural networks.
        info_set_vectoriser: InfoSetVectoriser. This maps information sets to vectors, so we can use neural networks.
        node: ExtensiveGameNode. The current node.
        player: int. The traversing player. Either 1 or 2.
        network1: RegretPredictor. The network for player 1.
        network2: RegretPredictor. The network for player 2.
        advantage_memory1: Reservoir. The advantage memory for player 1.
        advantage_memory2: Reservoir. The advantage memory for player 2.
        strategy_memory: Reservoir. The strategy memory (for both players).
        t: int. The current iteration of deep cfr.

    Returns:

    """
    if is_terminal(node):
        return payoffs(node)[player]
    elif which_player(node) == 0:
        # Chance player
        a = sample_chance_action(node)
        return cfr_traverse(game, action_indexer, info_set_vectoriser, node.children[a], player,
                            network1, network2,
                            advantage_memory1, advantage_memory2, strategy_memory, t)
    elif which_player(node) == player:
        # It's the traversing player's turn.
        state_vector = info_set_vectoriser.get_vector(game.get_info_set_id(node))
        values = dict()
        for action in get_available_actions(node):
            child = node.children[action]
            values[action] = cfr_traverse(game, action_indexer, info_set_vectoriser, child, player,
                                          network1, network2,
                                          advantage_memory1, advantage_memory2, strategy_memory, t)
            assert values[action] is not None, print("Shouldn't be None! node was: {}".format(node))
        info_set_regrets = dict()

        # Compute the player's strategy
        network = network1 if player == 1 else network2
        if t == 1:
            # This is the equivalent of initialising the network so it starts with all zeroes.
            info_set_strategy = extensive_game.ActionFloat.initialise_uniform(action_indexer.actions)
        else:
            info_set_strategy = network.compute_action_probs(state_vector, action_indexer)

        sampled_counterfactual_value = sum([info_set_strategy[action] * values[action] for action in
                                            get_available_actions(
            node)])
        for action in get_available_actions(node):
            info_set_regrets[action] = values[action] - sampled_counterfactual_value

        info_set_id = game.info_set_ids[node]
        advantage_memory = advantage_memory1 if player == 1 else advantage_memory2
        advantage_memory.append(AdvantageMemoryElement(info_set_id, t, info_set_regrets))

        # In traverser infosets, the value passed back up is the weighted average of all action values,
        # where action a’s weight is info_set_strategy[a]
        return sampled_counterfactual_value
    else:
        # It's the other player's turn.
        state_vector = info_set_vectoriser.get_vector(game.get_info_set_id(node))

        # Compute the other player's strategy
        other_player = 1 if player == 2 else 2
        network = network1 if other_player == 1 else network2
        if t == 1:
            # This is the equivalent of initialising the network so it starts with all zeroes.
            info_set_strategy = extensive_game.ActionFloat.initialise_uniform(action_indexer.actions)
        else:
            info_set_strategy = network.compute_action_probs(state_vector, action_indexer)

        info_set_id = game.info_set_ids[node]
        strategy_memory.append(StrategyMemoryElement(info_set_id, t, info_set_strategy))

        action = sample_action(info_set_strategy, available_actions=get_available_actions(node))
        return cfr_traverse(game, action_indexer, info_set_vectoriser, node.children[action], player,
                            network1, network2, advantage_memory1, advantage_memory2, strategy_memory, t)
