
import unittest

import numpy as np

from rlpoker import cfr
from rlpoker.extensive_game import ActionFloat
from rlpoker.cfr import ActionFloat


class TestCFR(unittest.TestCase):

    def test_compute_regret_matching(self):
        regrets = ActionFloat({1: 0.5, 2: 1.0, 3: -1.0})
        strategy = cfr.compute_regret_matching(regrets)
        expected_strategy = ActionFloat({1: 0.5 / 1.5, 2: 1.0 / 1.5, 3: 0.0})
        self.assertEqual(strategy, expected_strategy)

        # If all regrets are negative, play uniformly.
        regrets = ActionFloat({1: -1.0, 2: -3.0, 3: -1.0})
        strategy = cfr.compute_regret_matching(regrets)
        expected_strategy = ActionFloat({1: 1 / 3, 2: 1 / 3, 3: 1 / 3})
        self.assertEqual(strategy, expected_strategy)

        # If all regrets are zero, play uniformly.
        regrets = ActionFloat({1: 0.0, 2: 0.0, 3: 0.0})
        strategy = cfr.compute_regret_matching(regrets)
        expected_strategy = ActionFloat({1: 1 / 3, 2: 1 / 3, 3: 1 / 3})
        self.assertEqual(strategy, expected_strategy)

    def test_action_regrets(self):
        action_regrets = ActionFloat.initialise_zero(['a', 'b'])

        assert isinstance(action_regrets, ActionFloat)

        self.assertEqual(action_regrets['a'], 0.0)
        self.assertEqual(action_regrets['b'], 0.0)

        self.assertEqual(set(action_regrets.keys()), {'a', 'b'})
