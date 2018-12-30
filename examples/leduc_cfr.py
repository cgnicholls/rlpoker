import argparse

from rlpoker.cfr import cfr, save_strategy, load_strategy
from games.leduc import Leduc
from games.one_card_poker import OneCardPoker
from rlpoker.best_response import compute_exploitability


if __name__ == "__main__":
    games = ['Leduc', 'OneCardPoker']

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_iters', default=10000, type=int,
                        help='The number of iterations to run CFR for.')
    parser.add_argument('--game', default='Leduc', type=str,
                        choices=games,
                        help='The game to run CFR on.')
    parser.add_argument('--num_cards', default=3, type=int,
                        help='In OneCardPoker or Leduc, pass the number of '
                             'cards to use.')
    parser.add_argument('--use_chance_sampling', action='store_true',
                        help='Pass this option to use chance sampling. By '
                             'default, we don\'t use chance sampling.')
    args = parser.parse_args()

    if args.game == 'Leduc':
        print("Solving Leduc Hold Em with {} iterations".format(
            args.num_iters))
        game = Leduc.create_game(args.num_cards)

    elif args.game == 'OneCardPoker':
        print("Solving One Card Poker")
        game = OneCardPoker.create_game(args.num_cards)

    strategy, exploitabilities = cfr(game, num_iters=args.num_iters,
        use_chance_sampling=args.use_chance_sampling)

    strategy_name = '{}.strategy'.format(args.game)
    print("Saving strategy at {}".format(strategy_name))
    save_strategy(strategy, strategy_name)

    exploitability = compute_exploitability(game, strategy)
    print("Exploitability of saved strategy: {}".format(exploitability))
