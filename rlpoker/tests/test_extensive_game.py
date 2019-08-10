import unittest

from rlpoker import extensive_game
from rlpoker.games.rock_paper_scissors import rock_paper_scissors


class TestExtensiveGame(unittest.TestCase):

    def test_expected_value_exact(self):

        game, _, _ = rock_paper_scissors()

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
        game, _, _ = rock_paper_scissors()

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

    def test_is_strategy_complete(self):
        game, _, _ = rock_paper_scissors()

        # Incomplete because missing an information set.
        strategy = extensive_game.Strategy({
            (): extensive_game.ActionFloat({'R': 0.4, 'P': 0.5, 'S': 0.1})
        })
        computed = game.is_strategy_complete(strategy)
        self.assertEqual(computed, False)

        # Incomplete because missing an action.
        strategy = extensive_game.Strategy({
            (): extensive_game.ActionFloat({'R': 0.4, 'P': 0.5, 'S': 0.1}),
            (-1,): extensive_game.ActionFloat({'R': 0.4, 'P': 0.5})
        })
        computed = game.is_strategy_complete(strategy)
        self.assertEqual(computed, False)

        # Complete.
        strategy = extensive_game.Strategy({
            (): extensive_game.ActionFloat({'R': 0.4, 'P': 0.5, 'S': 0.1}),
            (-1,): extensive_game.ActionFloat({'R': 0.4, 'P': 0.3, 'S': 0.3})
        })
        computed = game.is_strategy_complete(strategy)
        self.assertEqual(computed, True)


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

        strategy1['info1'] = extensive_game.ActionFloat({'a1': 0.2, 'a2': 0.8})

        self.assertEqual(strategy1['info2'], strategy2['info2'])

        self.assertEqual(strategy2['info1'], extensive_game.ActionFloat({'a1': 0.4, 'a2': 0.6}))

    def test_compute_weighted_strategy(self):
        strategies = {
            'info1': [
                (1.0, extensive_game.ActionFloat({'a1': 0.4, 'a2': 0.6})),
                (2.0, extensive_game.ActionFloat({'a1': 0.5, 'a2': 0.5}))
            ],
            'info2': [
                (3.0, extensive_game.ActionFloat({'a1': 0.6, 'a2': 0.4})),
                (2.0, extensive_game.ActionFloat({'a1': 0.3, 'a2': 0.7})),
                (1.0, extensive_game.ActionFloat({'a1': 0.0, 'a2': 1.0}))
            ]
        }

        expected = extensive_game.Strategy({
            'info1': extensive_game.ActionFloat({
                'a1': (1.0 * 0.4 + 2.0 * 0.5) / (1.0 + 2.0),
                'a2': (1.0 * 0.6 + 2.0 * 0.5) / (1.0 + 2.0)
            }),
            'info2': extensive_game.ActionFloat({
                'a1': (3.0 * 0.6 + 2.0 * 0.3 + 1.0 * 0.0) / (3.0 + 2.0 + 1.0),
                'a2': (3.0 * 0.4 + 2.0 * 0.7 + 1.0 * 1.0) / (3.0 + 2.0 + 1.0)
            })
        })
        computed = extensive_game.compute_weighted_strategy(strategies)

        self.assertEqual(computed, expected)

    def test_equals(self):
        strategy1 = extensive_game.Strategy({
            'info1': extensive_game.ActionFloat({'a1': 0.4, 'a2': 0.6}),
            'info2': extensive_game.ActionFloat({'a2': 0.3, 'a4': 0.7})
        })

        # Same as strategy1
        strategy2 = extensive_game.Strategy({
            'info1': extensive_game.ActionFloat({'a1': 0.4, 'a2': 0.6}),
            'info2': extensive_game.ActionFloat({'a2': 0.3, 'a4': 0.7})
        })

        # Different to strategy1
        strategy3 = extensive_game.Strategy({
            'info1': extensive_game.ActionFloat({'a1': 0.3, 'a2': 0.7}),
            'info2': extensive_game.ActionFloat({'a2': 0.3, 'a4': 0.7})
        })

        self.assertEqual(strategy1, strategy2)
        self.assertNotEqual(strategy1, strategy3)


class TestActionFloat(unittest.TestCase):

    def test_copy(self):
        action_floats1 = extensive_game.ActionFloat({'a1': 0.4, 'a2': 0.6})
        action_floats2 = action_floats1.copy()

        self.assertEqual(action_floats1, action_floats2)

        action_floats1 = extensive_game.ActionFloat({'a1': 0.3, 'a2': 0.7})

        self.assertNotEqual(action_floats1, action_floats2)

    def test_iter(self):
        action_float = extensive_game.ActionFloat({'a1': 0.4, 'a2': 0.6})

        actions = [a for a in action_float]
        self.assertEqual(set(actions), {'a1', 'a2'})

    def test_add(self):
        action_float1 = extensive_game.ActionFloat({'a': 1.0, 'b': -1.0})
        action_float2 = extensive_game.ActionFloat({'a': 1.0, 'c': 2.0})

        action_float = extensive_game.ActionFloat.sum(action_float1, action_float2)

        expected = extensive_game.ActionFloat({
            'a': 2.0, 'b': -1.0, 'c': 2.0
        })
        self.assertEqual(action_float, expected)