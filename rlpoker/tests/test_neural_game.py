
import unittest

import numpy as np

from rlpoker import neural_game


class TestActionIndexer(unittest.TestCase):

    def test_action_indexing(self):
        action_indexer = neural_game.ActionIndexer(['a', 'b', 'c', 'd'])

        self.assertEqual(action_indexer.get_index('a'), 0)
        self.assertEqual(action_indexer.get_index('d'), 3)

        self.assertEqual(action_indexer.get_action(2), 'c')

    def test_get_dim(self):
        action_indexer = neural_game.ActionIndexer(['a', 'b', 'c'])
        self.assertEqual(action_indexer.action_dim, 3)

        action_indexer = neural_game.ActionIndexer(['a', 'b', 'c', 'd', 'e'])
        self.assertEqual(action_indexer.action_dim, 5)


class TestInfoSetVectoriser(unittest.TestCase):

    def test_get_state_shape(self):

        vs = [np.array([1, 2, 3]), np.array([3, 4, 5])]
        computed = neural_game.InfoSetVectoriser._get_state_shape(vs)
        self.assertEqual(computed, (3,))

        vs = [np.array([[1, 2, 3]]), np.array([[3, 4, 5]])]
        computed = neural_game.InfoSetVectoriser._get_state_shape(vs)
        self.assertEqual(computed, (1, 3))

        with self.assertRaises(ValueError):
            vs = [np.array([1, 2, 3]), np.array([3, 4, 5]), 5]
            computed = neural_game.InfoSetVectoriser._get_state_shape(vs)

        with self.assertRaises(ValueError):
            vs = [np.array([1, 2, 3]), np.array([3, 4, 5]), np.array([[2, 3, 4]])]
            computed = neural_game.InfoSetVectoriser._get_state_shape(vs)
