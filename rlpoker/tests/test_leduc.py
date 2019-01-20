
from rlpoker.games.leduc import Leduc
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
