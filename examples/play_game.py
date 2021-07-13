import argparse

from rlpoker.cfr.cfr_game import is_terminal, which_player, sample_chance_action
from rlpoker.games.game_builder import buildExtensiveGame

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--game_specifier', default='Leduc:values_3:suits_2')
    args = parser.parse_args()

    game = buildExtensiveGame(spec=args.game_specifier)

    node = game.root
    while not is_terminal(node):
        print("Node: {}".format(node))

        # If we are the chance player, then make a move randomly.
        if which_player(node) == 0:
            action = sample_chance_action(node)
            node = node.children[action]
        else:
            print(f"Player: {which_player(node)}")
            action = None
            while action not in node.children:
                action = int(input("Choose move: (0 = fold, 1 = call, 2 = bet) "))
            node = node.children[action]

    print("Terminal node")
    print(node)
    print("Utility: {}".format(node.utility))
