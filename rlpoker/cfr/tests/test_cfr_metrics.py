import collections
import unittest

from rlpoker import extensive_game
from rlpoker.games import rock_paper_scissors
from rlpoker.cfr import cfr_metrics


class TestCFRBasic(unittest.TestCase):

    def test_rock_paper_scissors_recursive(self):
        game = rock_paper_scissors.create_rock_paper_scissors()

        info_set_1 = game.info_set_ids[game.root]
        info_set_2 = game.info_set_ids[game.root.children['R']]

        # Player 1 plays (R, P, S) with probabilities (0.5, 0.3, 0.2), respectively.
        sigma_1 = extensive_game.Strategy({
            info_set_1: extensive_game.ActionFloat({
                'R': 0.5, 'P': 0.3, 'S': 0.2
            })
        })

        # Player 2 plays (R, P, S) with probabilities (0.3, 0.3, 0.4), respectively.
        sigma_2 = extensive_game.Strategy({
            info_set_2: extensive_game.ActionFloat({
                'R': 0.3, 'P': 0.3, 'S': 0.4
            }),
        })

        # Check that terminal nodes have value equal to their utility to the player.
        terminal_nodes = [
            game.get_node((a1, a2)) for a1 in ['R', 'P', 'S'] for a2 in ['R', 'P', 'S']
        ]

        for node in terminal_nodes:
            v1, v2 = cfr_metrics.compute_expected_utility_recursive(game, node, sigma_1, sigma_2,
                                                                    collections.defaultdict(float), collections.defaultdict(float),
                                                                    1.0, 1.0, 1.0)
            expected_v1 = node.utility[1]
            expected_v2 = node.utility[2]

            self.assertEqual(v1, expected_v1)
            self.assertEqual(v2, expected_v2)

        # Check the values of the player 2 nodes.
        v1, v2 = cfr_metrics.compute_expected_utility_recursive(game, game.get_node(('R',)), sigma_1, sigma_2,
                                                                collections.defaultdict(float), collections.defaultdict(float),
                                                                0.5, 1.0, 1.0)
        self.assertEqual(v1, 0 * 0.3 + -1 * 0.3 + 1 * 0.4)
        self.assertEqual(v2, 0 * 0.3 + 1 * 0.3 + -1 * 0.4)

        # Check the values of the (only) player 1 node.
        v1, v2 = cfr_metrics.compute_expected_utility_recursive(game, game.get_node(()), sigma_1, sigma_2,
                                                                collections.defaultdict(float), collections.defaultdict(float),
                                                                0.5, 1.0, 1.0)
        self.assertEqual(v1,
                         (0.5 * (0 * 0.3 + -1 * 0.3 + 1 * 0.4) +  # RR, RP, RS
                          0.3 * (1 * 0.3 + 0 * 0.3 + -1 * 0.4) +  # PR, PP, PS
                          0.2 * (-1 * 0.3 + 1 * 0.3 + 0 * 0.4)))  # SR, SP, SS
        self.assertEqual(v2,
                         (0.5 * (0 * 0.3 + 1 * 0.3 + -1 * 0.4) +  # RR, RP, RS
                          0.3 * (-1 * 0.3 + 0 * 0.3 + 1 * 0.4) +  # PR, PP, PS
                          0.2 * (1 * 0.3 + -1 * 0.3 + 0 * 0.4)))  # SR, SP, SS

    def test_rock_paper_scissors(self):
        game = rock_paper_scissors.create_rock_paper_scissors()

        info_set_1 = game.info_set_ids[game.root]
        info_set_2 = game.info_set_ids[game.root.children['R']]

        # Player 1 plays (R, P, S) with probabilities (0.5, 0.3, 0.2), respectively.
        sigma_1 = extensive_game.Strategy({
            info_set_1: extensive_game.ActionFloat({
                'R': 0.5, 'P': 0.3, 'S': 0.2
            })
        })

        # Player 2 plays (R, P, S) with probabilities (0.3, 0.3, 0.4), respectively.
        sigma_2 = extensive_game.Strategy({
            info_set_2: extensive_game.ActionFloat({
                'R': 0.3, 'P': 0.3, 'S': 0.4
            }),
        })

        # Check the values.
        expected_utility_1, expected_utility_2 = cfr_metrics.compute_expected_utility(game, sigma_1, sigma_2)

        utility_root = (0.5 * (0 * 0.3 + -1 * 0.3 + 1 * 0.4) +  # RR, RP, RS
                        0.3 * (1 * 0.3 + 0 * 0.3 + -1 * 0.4) +  # PR, PP, PS
                        0.2 * (-1 * 0.3 + 1 * 0.3 + 0 * 0.4))   # SR, SP, SS
        self.assertEqual(expected_utility_1[game.get_node(())], utility_root)

        utility_R = 0 * 0.3 + 1 * 0.3 + -1 * 0.4  # RR + RP + RS
        self.assertEqual(expected_utility_2[game.get_node(('R',))], utility_R)
        utility_P = -1 * 0.3 + 0 * 0.3 + 1 * 0.4  # PR, PP, PS
        self.assertEqual(expected_utility_2[game.get_node(('P',))], utility_P)
        utility_S = 1 * 0.3 + -1 * 0.3 + 0 * 0.4  # SR, SP, SS
        self.assertEqual(expected_utility_2[game.get_node(('S',))], utility_S)
