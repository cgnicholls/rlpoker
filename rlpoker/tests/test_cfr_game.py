
import unittest

from rlpoker import extensive_game
from rlpoker import cfr_game


class TestCFRGame(unittest.TestCase):

    def test_sample_chance_action(self):

        # Check we get an exception if the node isn't player 0.
        node = extensive_game.ExtensiveGameNode(player=1, action_list=(1, 2, 3), hidden_from={3})
        with self.assertRaises(ValueError):
            cfr_game.sample_chance_action(node)

        # Check we can sample a chance action.
        node = extensive_game.ExtensiveGameNode(player=0, action_list=(1, 2, 3), hidden_from={3})