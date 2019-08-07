
import typing

import numpy as np

from rlpoker import extensive_game, neural_game


def rock_paper_scissors() -> typing.Tuple[extensive_game.ExtensiveGame,
                                          neural_game.ActionIndexer,
                                          neural_game.InfoSetVectoriser]:
    """Returns a rock paper scissors game, ActionIndexer and InfoStateVectoriser.

    Returns:
        ExtensiveGame, ActionIndexer, InfoStateVectoriser
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

    action_indexer = neural_game.ActionIndexer(['R', 'P', 'S'])

    # There are only two information sets in rock, paper, scissors. Each player has one, since each player knows no
    # information about the other player's move when they take their own move.
    vectors = {
        game.get_info_set_id(root): np.array([1, 0]),
        game.get_info_set_id(root.children['R']): np.array([0, 1]),
    }
    info_set_vectoriser = neural_game.InfoSetVectoriser(vectors)

    return game, action_indexer, info_set_vectoriser
