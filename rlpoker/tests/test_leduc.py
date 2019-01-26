
from collections import Counter

import numpy as np

from rlpoker.games.leduc import (Leduc, compute_state_vectors,
    compute_betting_rounds, LeducNFSP)
from rlpoker.games.card import Card

def test_leduc():
    """Test Leduc.
    """

    # Create a game of Leduc with 9 cards.
    cards = [Card(value, suit) for value in range(3) for suit in range(2)]
    game = Leduc(cards=cards)

    # Check we can play out a hand:
    node = game.root

    assert set(node.children.keys()) == set(cards)

    # 1 gets a (0, 0), 2 gets a (1, 0).
    node = node.children[Card(0, 0)]
    node = node.children[Card(1, 0)]

    # Player 1 checks
    node = node.children[1]

    # Player 2 bets
    node = node.children[2]

    # Player 1 calls
    node = node.children[1]

    assert set(node.children.keys()) == {Card(2, 0), Card(0, 1), Card(1, 1),
                                         Card(2, 1)}

    # The board is (1, 1).
    node = node.children[Card(1, 1)]

    # Player 2 to play
    assert node.player == 2

    # 4 bets
    node = node.children[2]
    node = node.children[2]
    node = node.children[2]
    node = node.children[2]

    # End of game. Player 2 wins with a pair.
    # The raise amount is 2 in the first round and 4 in the second. Both
    # players ante 1 chip initially. Thus, the pot is: 2 * 1 + 2 * 2 + 4 * 4
    #  = 22. Since 2 put in 11 to this pot, they gain 22 - 11 = 11. Player 1
    #  loses 11.
    assert node.utility == {1: -11, 2: 11}


def test_compute_showdown():

    # Check that the same hole cards results in a draw.
    computed = Leduc.compute_showdown(Card(2, 3), Card(2, 4), Card(1, 0),
                                      {1: 10, 2: 10})
    expected = {1: 0, 2: 0}
    assert computed == expected

    # Check that player 1 pair results in a win.
    computed = Leduc.compute_showdown(Card(2, 3), Card(1, 4), Card(2, 0),
                                      {1: 10, 2: 10})
    expected = {1: 10, 2: -10}
    assert computed == expected

    # Check that player 2 pair results in a win.
    computed = Leduc.compute_showdown(Card(1, 3), Card(2, 4), Card(2, 0),
                                      {1: 10, 2: 10})
    expected = {1: -10, 2: 10}
    assert computed == expected

    # Check that player 1 higher card and no pairs results in a win.
    computed = Leduc.compute_showdown(Card(1, 3), Card(0, 4), Card(2, 0),
                                      {1: 10, 2: 10})
    expected = {1: 10, 2: -10}
    assert computed == expected

    # Check that player 2 higher card and no pairs results in a win.
    computed = Leduc.compute_showdown(Card(1, 3), Card(3, 4), Card(2, 0),
                                      {1: 10, 2: 10})
    expected = {1: -10, 2: 10}
    assert computed == expected


def test_compute_betting_rounds():
    info_set_ids = [
        (-1, Card(1, 2), 1, 2, 2),
        (-1, Card(1, 2), 1, 2, 2, Card(3, 3)),
        (-1, Card(1, 2), 1, 2, 2, Card(3, 3), 1),
        (Card(1, 2), -1, 2, 1, Card(3, 3), 1, 2)
    ]

    expected = [
        ((1, 2, 2), (), 1),
        ((1, 2, 2), (), 2),
        ((1, 2, 2), (1,), 2),
        ((2, 1), (1, 2), 2)
    ]

    for info_set_id, expected in zip(info_set_ids, expected):
        assert expected == compute_betting_rounds(info_set_id)


def test_compute_state_vectors():

    card_indices = {Card(1, 2): 0,
                    Card(2, 2): 1,
                    Card(3, 3): 2}
    info_set_ids = [
        (-1, Card(1, 2), 1, 2, 2),
        (-1, Card(1, 2), 1, 2, 2, Card(3, 3)),
        (-1, Card(1, 2), 1, 2, 2, Card(3, 3), 1),
        (Card(1, 2), -1, 2, 1, Card(3, 3), 1, 2)
    ]

    expected = [
        np.concatenate([
            np.array([1]),
            np.array([1, 0, 0]),
            np.array([0, 0, 0]),
            np.array([0, 0, 1, 0, 1]),
            np.array([0, 0, 0, 0, 0])], axis=0).astype(float)
    ]
    computed = compute_state_vectors(info_set_ids, card_indices, max_raises=2)

    print(expected[0])
    print(computed[info_set_ids[0]])
    assert np.all(expected[0] == computed[info_set_ids[0]])


def test_compute_state_vectors_unique():

    cards = [Card(1, 2), Card(2, 2), Card(3, 3)]
    game = LeducNFSP(cards)

    state_vectors = game._state_vectors

    seen_vectors = {}

    for info_set_id, v in state_vectors.items():
        t = tuple(v)
        if t in seen_vectors:
            print("Clash1: {}, {}".format(seen_vectors[t], t))
            print("Clash2: {}, {}".format(info_set_id, t))
            print(info_set_id)

            assert False

        seen_vectors[t] = info_set_id

    # assert len(set(game.state_vectors.keys())) == len({tuple(v) for v in game.state_vectors.values()})
