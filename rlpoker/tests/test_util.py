
import unittest

import numpy as np

import rlpoker.util as util
from rlpoker.games import card


class TestUtil(unittest.TestCase):

    def test_sample_action(self):
        strategy = {1: 0.2, 2: 0.8}
        np.random.seed(0)
        actions = [util.sample_action(strategy) for i in range(20)]

        expected_actions = [
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2
        ]

        self.assertEqual(actions, expected_actions)

        strategy = {'R': 0.2, 'P': 0.7, 'S': 0.1}
        np.random.seed(0)
        actions = [util.sample_action(strategy) for i in range(20)]

        expected_actions = [
            'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'S', 'P', 'P', 'P', 'P', 'S', 'R', 'R', 'R', 'P', 'P', 'P'
        ]

        self.assertEqual(actions, expected_actions)

        # Test we can sample an arbitrary object
        strategy = {'a': 0.2, 'b': 0.8}
        computed = util.sample_action(strategy)
        self.assertTrue(computed in strategy)

        strategy = {card.Card(3, 4): 0.2, card.Card(2, 2): 0.8}
        computed = util.sample_action(strategy)
        self.assertTrue(computed in strategy)

        # Check that we don't sample an unavailable action
        strategy = {1: 0.2, 2: 0.8}
        np.random.seed(0)
        actions = [util.sample_action(strategy, available_actions=[1]) for i in range(1000)]
        self.assertEqual(set(actions), {1})
