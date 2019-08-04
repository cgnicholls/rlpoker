import unittest

from rlpoker import extensive_game


def rock_paper_scissors() -> extensive_game.ExtensiveGame:
    """Returns a rock paper scissors game.

    Returns:
        ExtensiveGame
    """
    # Define rock, paper scissors.
    root = extensive_game.ExtensiveGameNode(1, action_list=(), hidden_from={2})

    # Add actions for player 1.
    root.children = {
        'R': extensive_game.ExtensiveGameNode(2, action_list=('R',), hidden_from={1}),
        'P': extensive_game.ExtensiveGameNode(2, action_list=('P',), hidden_from={1}),
        'S': extensive_game.ExtensiveGameNode(2, action_list=('S',), hidden_from={1}),
    }

    # Add actions for player 2.
    root.children['R'].children = {
        'R': extensive_game.ExtensiveGameNode(-1, action_list=('R', 'R'), utility={1: 0, 2: 0}),
        'P': extensive_game.ExtensiveGameNode(-1, action_list=('R', 'P'), utility={1: -1, 2: 1}),
        'S': extensive_game.ExtensiveGameNode(-1, action_list=('R', 'S'), utility={1: 1, 2: -1}),
    }

    # Add actions for player 2.
    root.children['P'].children = {
        'R': extensive_game.ExtensiveGameNode(-1, action_list=('P', 'R'), utility={1: 1, 2: -1}),
        'P': extensive_game.ExtensiveGameNode(-1, action_list=('P', 'P'), utility={1: 0, 2: 0}),
        'S': extensive_game.ExtensiveGameNode(-1, action_list=('P', 'S'), utility={1: -1, 2: 1}),
    }

    # Add actions for player 2.
    root.children['S'].children = {
        'R': extensive_game.ExtensiveGameNode(-1, action_list=('S', 'R'), utility={1: -1, 2: 1}),
        'P': extensive_game.ExtensiveGameNode(-1, action_list=('S', 'P'), utility={1: 1, 2: -1}),
        'S': extensive_game.ExtensiveGameNode(-1, action_list=('S', 'S'), utility={1: 0, 2: 0}),
    }

    game = extensive_game.ExtensiveGame(root)

    return game


class TestExtensiveGame(unittest.TestCase):

    def test_action_indexing(self):
        action_indexer = extensive_game.ActionIndexer(['a', 'b', 'c', 'd'])

        self.assertEqual(action_indexer.get_index('a'), 0)
        self.assertEqual(action_indexer.get_index('d'), 3)

        self.assertEqual(action_indexer.get_action(2), 'c')

    def test_expected_value_exact(self):

        game = rock_paper_scissors()

        strategy1 = extensive_game.Strategy({
            game.info_set_ids[game.get_node(())]: extensive_game.ActionFloat({
                'R': 0.5,
                'P': 0.2,
                'S': 0.3
            })
        })

        strategy2 = extensive_game.Strategy({
            game.info_set_ids[game.get_node(('R',))]: extensive_game.ActionFloat({
                'R': 0.2,
                'P': 0.3,
                'S': 0.5
            })
        })

        computed1, computed2 = game.expected_value_exact(strategy1=strategy1, strategy2=strategy2)

        expected1 = 0.5 * (0.2 * 0 + 0.3 * -1 + 0.5 * 1) + \
                    0.2 * (0.2 * 1 + 0.3 * 0 + 0.5 * -1) + \
                    0.3 * (0.2 * -1 + 0.3 * 1 + 0.5 * 0)

        self.assertEqual(computed1, expected1)
        self.assertEqual(computed2, -expected1)

    def test_get_node(self):
        game = rock_paper_scissors()

        # Check we can get the root node
        node = game.get_node(actions=())
        self.assertEqual(node, game.root)

        # Check we can get the node for R
        node = game.get_node(actions=('R',))
        self.assertEqual(node, game.root.children['R'])

        # Check we can get the node for R, P
        node = game.get_node(actions=('R', 'P'))
        self.assertEqual(node, game.root.children['R'].children['P'])

        # Check the node for R, P, A doesn't exist.
        node = game.get_node(actions=('R', 'P', 'A'))
        self.assertEqual(node, None)


class TestExtensiveGameNode(unittest.TestCase):

    def test_actions(self):

        node = extensive_game.ExtensiveGameNode(
            player=1,
            action_list=(),
            children={
                'a': extensive_game.ExtensiveGameNode(2, ('a',)),
                'b': extensive_game.ExtensiveGameNode(2, ('b',))
            }
        )

        self.assertEqual(set(node.actions), {'a', 'b'})



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
