
import unittest

import numpy as np

import rlpoker.util as util


class TestUtil(unittest.TestCase):

    def test_sample_action(self):
        strategy = {1: 0.2, 2: 0.8}
        np.random.seed(0)
        actions = [util.sample_action(strategy) for i in range(20)]

        expected_actions = [
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2
        ]

        self.assertEqual(actions, expected_actions)
