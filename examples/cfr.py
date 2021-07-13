import argparse

# import bokeh.plotting as plt

from rlpoker.cfr import cfr, external_cfr, cfr_metrics
from rlpoker.experiment import Experiment, WANDBExperimentWriter
from rlpoker.games.util import ExtensiveGameBuilder
from rlpoker.games.leduc import Leduc
from rlpoker.games.rock_paper_scissors import create_neural_rock_paper_scissors
from rlpoker.games.card import get_deck
from rlpoker.games.one_card_poker import OneCardPoker
from rlpoker.best_response import compute_exploitability

if __name__ == "__main__":
    games = ['Leduc', 'OneCardPoker', 'RockPaperScissors']
    cfr_algorithms = ['vanilla', 'external']

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', required=True, type=str, help='The name of the experiment.')
    parser.add_argument('--num_iters', default=10000, type=int,
                        help='The number of iterations to run CFR for.')
    parser.add_argument('--game_specifier', default='leduc:values_3:suits_2')
    parser.add_argument('--use_chance_sampling', action='store_true',
                        help='Pass this option to use chance sampling. By '
                             'default, we don\'t use chance sampling.')
    parser.add_argument('--cfr_algorithm', choices=cfr_algorithms, required=True,
                        help='Which cfr algorithm to use. Choose between vanilla and external.')
    args = parser.parse_args()

    game = ExtensiveGameBuilder.build(spec=args.game_specifier)

    experiment = Experiment(args.exp_name)
    experiment_writer = WANDBExperimentWriter(experiment)

    if args.cfr_algorithm == 'vanilla':
        strategy, exploitabilities, strategies = cfr.cfr(
            experiment,
            game,
            num_iters=args.num_iters,
            use_chance_sampling=args.use_chance_sampling,
            experiment_writer=experiment_writer,
        )
    elif args.cfr_algorithm == 'external':
        strategy, exploitabilities, strategies = external_cfr.external_sampling_cfr(
            experiment,
            game,
            num_iters=args.num_iters,
            experiment_writer=experiment_writer,
        )
    else:
        raise ValueError("args.cfr_algorithm was not in {}".format(cfr_algorithms))

    # Now compute the immediate regrets.
    immmediate_regret, _, _ = cfr_metrics.compute_immediate_regret(game, strategies)
    print("Immediate regret: {}".format(immmediate_regret))

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
