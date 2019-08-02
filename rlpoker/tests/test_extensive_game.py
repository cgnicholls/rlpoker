import unittest

from rlpoker import extensive_game


class TestExtensiveGame(unittest.TestCase):

    def test_action_indexing(self):
        action_indexer = extensive_game.ActionIndexer(['a', 'b', 'c', 'd'])

        self.assertEqual(action_indexer.get_index('a'), 0)
        self.assertEqual(action_indexer.get_index('d'), 3)

        self.assertEqual(action_indexer.get_action(2), 'c')


class TestStrategy(unittest.TestCase):

    def test_copy_strategy(self):
        strategy1 = extensive_game.Strategy({
            'info1': extensive_game.ActionFloat({'a1': 0.4, 'a2': 0.6}),
            'info2': extensive_game.ActionFloat({'a2': 0.3, 'a4': 0.7})
        })

        strategy2 = strategy1.copy()

        self.assertEqual(strategy1['info1'], strategy2['info1'])
        self.assertEqual(strategy1['info2'], strategy2['info2'])

        strategy1['info1']['a1'] = 0.2
        strategy1['info1']['a2'] = 0.8

        self.assertEqual(strategy1['info2'], strategy2['info2'])

        print(strategy1)
        print(strategy2)
        self.assertEqual(strategy2['info1'], extensive_game.ActionFloat({'a1': 0.4, 'a2': 0.6}))


class TestActionFloat(unittest.TestCase):

    def test_copy(self):
        action_floats1 = extensive_game.ActionFloat({'a1': 0.4, 'a2': 0.6})
        action_floats2 = action_floats1.copy()

        self.assertEqual(action_floats1, action_floats2)

        action_floats1['a1'] = 2.0

        self.assertEqual(action_floats2, extensive_game.ActionFloat({'a1': 0.4, 'a2': 0.6}))


class TestActionIndexer(unittest.TestCase):

    def test_get_dim(self):
        action_indexer = extensive_game.ActionIndexer(['a', 'b', 'c'])
        self.assertEqual(action_indexer.get_action_dim(), 3)

        action_indexer = extensive_game.ActionIndexer(['a', 'b', 'c', 'd', 'e'])
        self.assertEqual(action_indexer.get_action_dim(), 5)
