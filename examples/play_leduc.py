import argparse

from games.leduc import Leduc
from rlpoker.extensive_game import ExtensiveGame

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--raise_amount', type=int, default=2,
                        help='The raise amount for the first round. This '
                             'doubles in the second round.')
    parser.add_argument('--max_raises', type=int, default=4,
                        help='The maximum total number of raises allowed in '
                             'a betting round.')
    parser.add_argument('--cards', type=int, nargs="+",
                        help='The deck. Should be a space separated list of '
                             'card numbers, each at least 10.')
    args = parser.parse_args()

    cards = tuple(args.cards)
    assert min(cards) >= 10
    assert len(cards) >= 3

    root = Leduc.create_tree(cards,
                             max_raises=args.max_raises,
                             raise_amount=args.raise_amount)
    game = ExtensiveGame(root)

    node = root
    while node.player != -1:
        print("Node: {}".format(node))

        action = None
        while action not in node.children:
            action = int(input("Choose move: (0 = fold, 1 = call, 2 = bet) "))

        node = node.children[action]

    print("Terminal node")
    print(node)
    print("Utility: {}".format(node.utility))
