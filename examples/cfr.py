import argparse

# import bokeh.plotting as plt

from rlpoker.cfr import cfr, save_strategy, load_strategy
from rlpoker import external_cfr
from rlpoker.games.leduc import Leduc
from rlpoker.games.rock_paper_scissors import create_neural_rock_paper_scissors
from rlpoker.games.card import get_deck
from rlpoker.games.one_card_poker import OneCardPoker
from rlpoker.best_response import compute_exploitability
from rlpoker import cfr_metrics

if __name__ == "__main__":
    games = ['Leduc', 'OneCardPoker', 'RockPaperScissors']
    cfr_algorithms = ['vanilla', 'external']

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', default=None, type=str, help='The name of the experiment.')
    parser.add_argument('--num_iters', default=10000, type=int,
                        help='The number of iterations to run CFR for.')
    parser.add_argument('--game', default='Leduc', type=str, choices=games,
                        help='The game to run CFR on.')
    parser.add_argument('--num_values', default=3, type=int,
                        help='In OneCardPoker or Leduc, pass the number of '
                             'cards to use.')
    parser.add_argument('--num_suits', default=2, type=int,
                        help='In Leduc, pass the number of suits to use.')
    parser.add_argument('--use_chance_sampling', action='store_true',
                        help='Pass this option to use chance sampling. By '
                             'default, we don\'t use chance sampling.')
    parser.add_argument('--cfr_algorithm', choices=cfr_algorithms, required=True,
                        help='Which cfr algorithm to use. Choose between vanilla and external.')
    args = parser.parse_args()

    if args.game == 'Leduc':
        print("Solving Leduc Hold Em with {} iterations".format(args.num_iters))
        cards = get_deck(num_values=args.num_values, num_suits=args.num_suits)
        game = Leduc(cards)

    elif args.game == 'OneCardPoker':
        print("Solving One Card Poker")
        game = OneCardPoker.create_game(args.num_values)

    elif args.game == 'RockPaperScissors':
        print("Solving rock paper scissors")
        game, _, _ = create_neural_rock_paper_scissors()

    exp_name = f'{args.cfr_algorithm}/{args.exp_name}'

    if args.cfr_algorithm == 'vanilla':
        strategy, exploitabilities, strategies = cfr(
            exp_name,
            game,
            num_iters=args.num_iters,
            use_chance_sampling=args.use_chance_sampling
        )
    elif args.cfr_algorithm == 'external':
        strategy, exploitabilities, strategies = external_cfr.external_sampling_cfr(
            exp_name,
            game,
            num_iters=args.num_iters
        )
    else:
        raise ValueError("args.cfr_algorithm was not in {}".format(cfr_algorithms))

    # Now compute the immediate regrets.
    immmediate_regret, _, _ = cfr_metrics.compute_immediate_regret(game, strategies)
    print("Immediate regret: {}".format(immmediate_regret))

    # Save the strategy and plot the performance.

    strategy_name = '{}_cfr.strategy'.format(args.game)
    print("Saving strategy at {}".format(strategy_name))
    save_strategy(strategy, strategy_name)

    exploitability = compute_exploitability(game, strategy)
    print("Exploitability of saved strategy: {}".format(exploitability))

    # plot_name = '{}.html'.format(args.game)
    # plt.output_file(plot_name)
    # p = plt.figure(title='Exploitability for CFR trained on {}'.format(
    #     args.game), x_axis_label='t', y_axis_label='Exploitability')
    # times = [pair[0] for pair in exploitabilities]
    # exploits = [pair[1] for pair in exploitabilities]
    # p.line(times, exploits)
    #
    # print("Saved plot of exploitability at: {}".format(plot_name))
