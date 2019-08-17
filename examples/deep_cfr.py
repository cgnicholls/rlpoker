import argparse

from rlpoker import deep_cfr
from rlpoker.games import leduc, rock_paper_scissors, one_card_poker
from rlpoker.games import card
from rlpoker.games.rock_paper_scissors import create_neural_rock_paper_scissors
from rlpoker.best_response import compute_exploitability



if __name__ == "__main__":
    games = ['Leduc', 'RockPaperScissors']

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_iters', default=100, type=int,
                        help='The number of iterations to run deep CFR for.')
    parser.add_argument('--num_traversals', default=10000, type=int,
                        help='The number of traversals in each iteration.')
    parser.add_argument('--advantage_maxlen', default=1000000, type=int,
                        help='The maximum length of the advantage memory reservoirs.')
    parser.add_argument('--strategy_maxlen', default=1000000, type=int,
                        help='The maximum length of the strategy memory reservoir.')
    parser.add_argument('--batch_size', default=10000, type=int,
                        help='The batch size to use in training.')
    parser.add_argument('--num_sgd_updates', default=4000, type=int,
                        help='The number of epochs to use in training.')
    parser.add_argument('--game', default='Leduc', choices=games,
                        help='The game to play')

    parser.add_argument('--num_values', default=3, type=int,
                        help='In OneCardPoker or Leduc, pass the number of cards to use.')
    parser.add_argument('--num_suits', default=2, type=int,
                        help='In Leduc, pass the number of suits to use.')

    args = parser.parse_args()

    if args.game == 'Leduc':
        print("Solving Leduc Hold'em")
        cards = card.get_deck(num_values=args.num_values, num_suits=args.num_suits)
        n_game = leduc.create_neural_leduc(cards)
    elif args.game == 'RockPaperScissors':
        print("Solving rock paper scissors")
        n_game = rock_paper_scissors.create_neural_rock_paper_scissors()

    strategy, exploitabilities = deep_cfr.deep_cfr(n_game,
                                                   num_iters=args.num_iters, num_traversals=args.num_traversals,
                                                   advantage_maxlen=args.advantage_maxlen,
                                                   strategy_maxlen=args.strategy_maxlen,
                                                   batch_size=args.batch_size,
                                                   num_sgd_updates=args.num_sgd_updates)

    exploitability = compute_exploitability(n_game.extensive_game, strategy)
    print("Exploitability of strategy: {}".format(exploitability))
