import argparse
from typing import Dict

from rlpoker.cfr.cfr import CFRStrategySaver
from rlpoker.cfr.cfr_game import is_terminal, which_player, sample_chance_action, get_information_set
from rlpoker.experiment import Experiment
from rlpoker.extensive_game import Strategy, ExtensiveGame
from rlpoker.games.game_builder import buildExtensiveGame
from rlpoker.util import sample_action


def load_strategies(spec: str):
    strategies = dict()
    if spec:
        player_specs = spec.split(',')
        for spec in player_specs:
            player, exp_name = spec.split(':')

            experiment = Experiment(exp_name)
            saver = CFRStrategySaver(experiment)
            strategies[int(player)] = saver.load_best_strategy()

    return strategies


def play_game(game: ExtensiveGame, strategies: Dict[int, Strategy], verbose: bool = False):
    node = game.root
    while not is_terminal(node):
        if verbose:
            print("Node: {}".format(node))
        player = which_player(node)

        # If we are the chance player, then make a move randomly.
        if player == 0:
            action = sample_chance_action(node)
            node = node.children[action]
        else:
            print(f"Player: {which_player(node)}")
            info_set = get_information_set(game, node)

            if player in strategies:
                action_probs = strategies[player].get_action_probs(info_set)
                action = sample_action(action_probs)
            else:
                print(f"Information set: {info_set}")
                print(f"Valid actions: {sorted(node.children.keys())}")
                action = None
                while action not in node.children:
                    user_input = input("Choose move: (0 = fold, 1 = call, 2 = bet) ")
                    try:
                        action = int(user_input)
                    except ValueError:
                        print(f"Invalid input: {user_input}")

            print(f"Action: {action}")
            node = node.children[action]

    print(node)
    print("Utility: {}".format(node.utility))

    return node.utility


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--game_specifier', default='Leduc:values_3:suits_2')
    parser.add_argument('--strategies', type=str, default=None,
                        help="If given, then load these strategies. The format is comma-separated with each strategy "
                             "specified as 'player:experiment', where 'player' is the player number, e.g. 1 or 2")
    args = parser.parse_args()

    game = buildExtensiveGame(spec=args.game_specifier)

    strategies = load_strategies(args.strategies)

    total_utility = dict()
    n_games = 0
    while True:
        game_utility = play_game(game, strategies)
        n_games += 1

        for player in game_utility:
            if player not in total_utility:
                total_utility[player] = 0
            total_utility[player] += game_utility[player]

        print(f"Average utilities")
        for player, utility in total_utility.items():
            print(f"Player: {player}. Utility: {float(utility) / n_games:.04f}")
