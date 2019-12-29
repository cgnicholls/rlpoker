
import unittest

import rlpoker.cfr_util
from rlpoker import cfr
from rlpoker.extensive_game import ActionFloat


class TestCFR(unittest.TestCase):

    def test_compute_regret_matching(self):
        regrets = ActionFloat({1: 0.5, 2: 1.0, 3: -1.0})
        strategy = rlpoker.cfr_util.compute_regret_matching(regrets, epsilon=0.0)
        expected_strategy = ActionFloat({1: 0.5 / 1.5, 2: 1.0 / 1.5, 3: 0.0})
        self.assertEqual(strategy, expected_strategy)

        regrets = ActionFloat({1: 0.5, 2: 1.0, 3: -1.0})
        strategy = rlpoker.cfr_util.compute_regret_matching(regrets, epsilon=1e-7)
        expected_strategy = rlpoker.cfr_util.normalise_probs(ActionFloat({1: 0.5, 2: 1.0, 3: 1e-7}))
        self.assertEqual(strategy, expected_strategy)

        # If all regrets are negative, play uniformly.
        regrets = ActionFloat({1: -1.0, 2: -3.0, 3: -1.0})
        strategy = rlpoker.cfr_util.compute_regret_matching(regrets)
        expected_strategy = ActionFloat({1: 1 / 3, 2: 1 / 3, 3: 1 / 3})
        self.assertEqual(strategy, expected_strategy)

        # If all regrets are zero, and not highest_regret, play uniformly.
        regrets = ActionFloat({1: 0.0, 2: 0.0, 3: 0.0})
        strategy = rlpoker.cfr_util.compute_regret_matching(regrets)
        expected_strategy = ActionFloat({1: 1 / 3, 2: 1 / 3, 3: 1 / 3})
        self.assertEqual(strategy, expected_strategy)

        # If all regrets are non-positive, and highest_regret is True, play highest regret action.
        regrets = ActionFloat({1: -3.0, 2: -1.0, 3: -4.0})
        strategy = rlpoker.cfr_util.compute_regret_matching(regrets, epsilon=1e-6, highest_regret=True)
        expected_strategy = rlpoker.cfr_util.normalise_probs(ActionFloat({1: 0.0, 2: 1.0, 3: 0.0}), epsilon=1e-6)
        self.assertEqual(strategy, expected_strategy)

    def test_normalise_probs(self):
        probs = ActionFloat({'a': 1.0, 'b': 2.0})
        computed = rlpoker.cfr_util.normalise_probs(probs, epsilon=1e-7)
        expected = ActionFloat({'a': 1.0 / 3.0, 'b': 2.0 / 3.0})
        self.assertEqual(computed, expected)

        epsilon = 1e-7
        probs = ActionFloat({'a': 1.0, 'b': 0.0})
        computed = rlpoker.cfr_util.normalise_probs(probs, epsilon=epsilon)
        expected = ActionFloat({'a': 1.0 / (1 + epsilon), 'b': epsilon / (1 + epsilon)})
        self.assertEqual(computed, expected)

        epsilon = 0.0
        probs = ActionFloat({'a': 1.0, 'b': 0.0})
        computed = rlpoker.cfr_util.normalise_probs(probs, epsilon=epsilon)
        expected = ActionFloat({'a': 1.0, 'b': 0.0})
        self.assertEqual(computed, expected)

    def test_action_regrets(self):
        action_regrets = ActionFloat.initialise_zero(['a', 'b'])

        assert isinstance(action_regrets, ActionFloat)

        self.assertEqual(action_regrets['a'], 0.0)
        self.assertEqual(action_regrets['b'], 0.0)

        self.assertEqual(set(action_regrets.keys()), {'a', 'b'})
