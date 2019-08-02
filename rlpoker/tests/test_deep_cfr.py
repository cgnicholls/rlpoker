
import unittest

import numpy as np
import tensorflow as tf

from rlpoker import deep_cfr, extensive_game

class TestDeepRegretNetwork(unittest.TestCase):

    def test_initialise(self):

        action_indexer = extensive_game.ActionIndexer(['a', 'b'])
        network = deep_cfr.DeepRegretNetwork(state_dim=5, action_indexer=action_indexer, player=1)

        with tf.Session() as sess:
            network.initialise(sess)

            computed = network.predict_advantages(sess, np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

            self.assertEqual(computed.shape, (1, 2))
