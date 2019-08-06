
import unittest

from rlpoker import neural_game


class TestActionIndexer(unittest.TestCase):

    def test_action_indexing(self):
        action_indexer = neural_game.ActionIndexer(['a', 'b', 'c', 'd'])

        self.assertEqual(action_indexer.get_index('a'), 0)
        self.assertEqual(action_indexer.get_index('d'), 3)

        self.assertEqual(action_indexer.get_action(2), 'c')

    def test_get_dim(self):
        action_indexer = neural_game.ActionIndexer(['a', 'b', 'c'])
        self.assertEqual(action_indexer.get_action_dim(), 3)

        action_indexer = neural_game.ActionIndexer(['a', 'b', 'c', 'd', 'e'])
        self.assertEqual(action_indexer.get_action_dim(), 5)
