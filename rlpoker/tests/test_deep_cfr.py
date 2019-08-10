
import unittest
from unittest.mock import Mock

import numpy as np
import tensorflow as tf

from rlpoker import deep_cfr, neural_game, extensive_game, buffer
from rlpoker.tests import util

class TestDeepRegretNetwork(unittest.TestCase):

    def setUp(self):
        tf.reset_default_graph()

    def test_initialise(self):

        action_indexer = neural_game.ActionIndexer(['a', 'b'])
        network = deep_cfr.DeepRegretNetwork(state_shape=(5,), action_indexer=action_indexer, player=1)

        with tf.Session() as sess:
            network.set_sess(sess)
            network.initialise()

            # Check we can predict
            network.predict_advantages(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), action_indexer)

    def test_predict_advantages(self):
        info_set_vector = np.array([1, 2, 3, 4, 5]).astype(np.float32)
        action_indexer = neural_game.ActionIndexer(['a', 'b'])

        network = deep_cfr.DeepRegretNetwork(state_shape=(5,), action_indexer=action_indexer, player=1)

        with tf.Session() as sess:
            network.set_sess(sess)
            network.initialise()

            computed = network.predict_advantages(info_set_vector, action_indexer)

            self.assertEqual(set(computed.keys()), {'a', 'b'})
            self.assertTrue(type(computed['a']) == np.float32)
            self.assertTrue(type(computed['b']) == np.float32)


class TestDeepCFR(unittest.TestCase):

    def test_cfr_traverse_advantage_memory(self):
        game, action_indexer, info_set_vectoriser = util.rock_paper_scissors()

        node = game.root
        player = 1

        network1 = Mock()
        network1.compute_action_probs = Mock(return_value=extensive_game.ActionFloat({'R': 0.2, 'P': 0.7, 'S': 0.1}))

        network2 = Mock()
        network2.compute_action_probs = Mock(return_value=extensive_game.ActionFloat({'R': 0.3, 'P': 0.6, 'S': 0.1}))

        advantage_memory1 = buffer.Reservoir(maxlen=100)
        advantage_memory2 = buffer.Reservoir(maxlen=100)

        strategy_memory = buffer.Reservoir(maxlen=100)

        deep_cfr.cfr_traverse(
            game,
            action_indexer,
            info_set_vectoriser,
            node,
            player,
            network1,
            network2,
            advantage_memory1,
            advantage_memory2,
            strategy_memory,
            t=2
        )

        # We add to the traverser's advantage memory in each of their nodes, of which there is 1.
        self.assertEqual(len(advantage_memory1), 1)

        # We don't update 2's advantage memory
        self.assertEqual(len(advantage_memory2), 0)

        # We add to the strategy memory for each node of the non-traversing player, of which there are 3.
        self.assertEqual(len(strategy_memory), 3)

    def test_cfr_traverse_advantage_memory_player2(self):
        game, action_indexer, info_set_vectoriser = util.rock_paper_scissors()
        game.print_tree()

        node = game.root.children['R']
        assert node.player == 2
        player = 2
        t = 3

        network1 = Mock()
        network1.compute_action_probs = Mock(return_value=extensive_game.ActionFloat({'R': 0.2, 'P': 0.7, 'S': 0.1}))

        network2 = Mock()
        network2.compute_action_probs = Mock(return_value=extensive_game.ActionFloat({'R': 0.3, 'P': 0.6, 'S': 0.1}))

        advantage_memory1 = buffer.Reservoir(maxlen=100)
        advantage_memory2 = buffer.Reservoir(maxlen=100)

        strategy_memory = buffer.Reservoir(maxlen=100)

        deep_cfr.cfr_traverse(
            game,
            action_indexer,
            info_set_vectoriser,
            node,
            player,
            network1,
            network2,
            advantage_memory1,
            advantage_memory2,
            strategy_memory,
            t
        )

        # Shouldn't have used network 1
        network1.compute_action_probs.assert_not_called()

        # Should have used network 2 once.
        network2.compute_action_probs.assert_called_once()

        # We add to the traverser's advantage memory in each of their nodes, of which there is 1.
        self.assertEqual(len(advantage_memory1), 0)

        # We don't update 2's advantage memory
        self.assertEqual(len(advantage_memory2), 1)

        # We add to the strategy memory for each node of the non-traversing player, of which there are 3.
        self.assertEqual(len(strategy_memory), 0)

        advantage = advantage_memory2.buffer[0]
        print("Buffer: {}".format(advantage_memory2.buffer))
        print("Advantage: {}".format(advantage))
        expected = deep_cfr.AdvantageMemoryElement(game.get_info_set_id(node), t, {
            'R': -0.5,
            'P': 0.5,
            'S': -1.5
        })
        print("Expected: {}".format(expected))
        self.assertEqual(advantage, expected)
