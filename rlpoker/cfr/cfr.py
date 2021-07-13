# This implements Counterfactual Regret Minimization in a general zero-sum two-player game.

import pickle
import time
import typing

import numpy as np

from rlpoker import best_response
from rlpoker.cfr import cfr_metrics, cfr_util
from rlpoker.cfr.cfr_game import get_available_actions, get_information_set, \
    sample_chance_action, is_terminal, payoffs, which_player
from rlpoker.experiment import Experiment, StrategySaver, ExperimentWriter
from rlpoker.extensive_game import ActionFloat, Strategy


class CFRStrategySaver(StrategySaver):

    def _save_strategy(self, strategy, file_name: str):
        """Saves the strategy in the given file. A strategy is just a dictionary, so we save using json."""
        with open(file_name, 'wb') as f:
            pickle.dump(strategy, f)

    def _load_strategy(self, file_name: str):
        """Loads the strategy from a file. This is just json loading."""
        with open(file_name, 'rb') as f:
            strategy = pickle.load(f)

        return strategy


def compute_average_strategy(action_counts: typing.Dict) -> Strategy:
    average_strategy = dict()
    for information_set in action_counts:
        num_actions = sum([v for k, v in action_counts[information_set].items()])
        if num_actions > 0:
            average_strategy[information_set] = {
                k: float(v) / float(num_actions) for k, v in
                action_counts[information_set].items()}

    return Strategy(average_strategy)


def compare_strategies(s1: Strategy, s2: Strategy):
    """Returns the average Euclidean distance between the probability distributions.

    Args:
        s1: Strategy
        s2: Strategy

    Returns:
        float. The average Euclidean distance between the distributions.
    """
    common_keys = [k for k in s1.keys() if k in s2.keys()]
    distances = []
    for information_set in common_keys:
        prob_dist_diff = [
            float(s1[information_set][a] - s2[information_set][a]) ** 2 for a in
            s1[information_set]]
        distances.append(np.sqrt(np.mean(prob_dist_diff)))
    return np.mean(distances)


def cfr(experiment: Experiment, game, num_iters=10000, use_chance_sampling=True, linear_weight=False,
        experiment_writer: ExperimentWriter = False):
    """

    Args:
        experiment: the experiment.
        game:
        num_iters:
        use_chance_sampling:
        experiment_writer: used to write logs.

    Returns:
        average_strategy, exploitabilities
    """
    # regrets is a dictionary where the keys are the information sets and values
    # are dictionaries from actions available in that information set to the
    # counterfactual regret for not playing that action in that information set.
    # Since information sets encode the player, we only require one dictionary.
    regrets = dict()

    # Similarly, action_counts is a dictionary with keys the information sets
    # and values dictionaries from actions to action counts.
    action_counts = dict()

    cfr_state = cfr_util.CFRState()

    # Strategy_t holds the strategy at time t; similarly strategy_t_1 holds the
    # strategy at time t + 1.
    strategy_t = Strategy.initialise()
    strategy_t_1 = Strategy.initialise()

    average_strategy = None
    exploitabilities = []
    strategies = []

    average_strategy2 = cfr_util.AverageStrategy(game)

    saver = CFRStrategySaver(experiment=experiment)

    # Each information set is uniquely identified with an action tuple.
    start_time = time.time()
    for t in range(num_iters):
        weight = t if linear_weight else 1.0
        for i in [1, 2]:
            cfr_recursive(game, game.root, i, t, 1.0, 1.0, 1.0, regrets,
                          action_counts, strategy_t, strategy_t_1,
                          cfr_state,
                          use_chance_sampling=use_chance_sampling,
                          weight=weight)

        average_strategy = compute_average_strategy(action_counts)
        cfr_util.update_average_strategy(game, average_strategy2, strategy_t, weight=weight)

        # Update strategy_t to equal strategy_t_1. We update strategy_t_1 inside
        # cfr_recursive.  We take a copy because we update it inside
        # cfr_recursive, and want to hold on to strategy_t_1 separately to
        # compare.
        strategy_t = strategy_t_1.copy()
        strategies.append(strategy_t.copy())

        # Compute the exploitability of the strategy.
        if t % 10 == 0:
            print("t: {}. Time since last evaluation: {:.4f} s".format(t, time.time() - start_time))
            start_time = time.time()
            exploitability = best_response.compute_exploitability(
                game, average_strategy)
            exploitabilities.append((t, exploitability))

            print("t: {}, nodes touched: {}, exploitability: {} mbb/h".format(t, cfr_state.nodes_touched,
                                                                              exploitability * 1000))

            # exploitability = best_response.compute_exploitability(game, average_strategy2.compute_strategy())
            # print("Exploitability (av strategy method 2): {} mbb/h".format(exploitability * 1000))

            immediate_regret, _, _ = cfr_metrics.compute_immediate_regret(game, strategies)
            print("Immediate regret: {}".format(immediate_regret))

            log_data = {
                "nodes_touched": cfr_state.nodes_touched,
                "exploitability_mbbh": exploitability * 1000,
                "immediate_regret": immediate_regret,
            }

            if experiment_writer:
                experiment_writer.log(data=log_data, global_step=t)

            saver.save_best_strategy(average_strategy, t, exploitability)

    return average_strategy, exploitabilities, strategies


# The Game object holds a game state at any point in time, and can return an information set label
# for that game state, which uniquely identifies the information set and is the same for all states
# in that information set.
def cfr_recursive(game, node, i, t, pi_c, pi_1, pi_2, regrets: typing.Dict[typing.Any, ActionFloat],
                  action_counts, strategy_t, strategy_t_1, cfr_state: cfr_util.CFRState,
                  use_chance_sampling=False, weight=1.0):
    cfr_state.node_touched()
    # If the node is terminal, just return the payoffs
    if is_terminal(node):
        return payoffs(node)[i]
    # If the next player is chance, then sample the chance action
    elif which_player(node) == 0:
        if use_chance_sampling:
            a = sample_chance_action(node)
            return cfr_recursive(
                game, node.children[a], i, t, pi_c, pi_1, pi_2,
                regrets, action_counts, strategy_t, strategy_t_1,
                cfr_state,
                use_chance_sampling=use_chance_sampling,
                weight=weight,
            )
        else:
            value = 0
            for a, cp in node.chance_probs.items():
                value += cp * cfr_recursive(
                    game, node.children[a], i, t, cp * pi_c, pi_1, pi_2,
                    regrets, action_counts, strategy_t, strategy_t_1,
                    cfr_state,
                    use_chance_sampling=use_chance_sampling,
                    weight=weight,
                )
            return value

    # Get the information set
    information_set = get_information_set(game, node)

    # Get the player to play and initialise values
    player = which_player(node)
    value = 0
    available_actions = get_available_actions(node)
    values_Itoa = {a: 0 for a in available_actions}

    # Initialise strategy_t[information_set] uniformly at random.
    if information_set not in strategy_t.get_info_sets():
        strategy_t.set_uniform_action_probs(information_set, available_actions)

    # Compute the counterfactual value of this information set by computing the counterfactual
    # value of the information sets where the player plays each available action and taking
    # the expected value (by weighting by the strategy).
    for a in available_actions:
        if player == 1:
            values_Itoa[a] = cfr_recursive(
                game, node.children[a], i, t, pi_c,
                strategy_t.get_action_probs(information_set)[a] * pi_1, pi_2,
                regrets, action_counts, strategy_t, strategy_t_1,
                cfr_state,
                use_chance_sampling=use_chance_sampling,
                weight=weight,
            )
        else:
            values_Itoa[a] = cfr_recursive(
                game, node.children[a], i, t, pi_c,
                pi_1, strategy_t[information_set][a] * pi_2,
                regrets, action_counts, strategy_t, strategy_t_1,
                cfr_state,
                use_chance_sampling=use_chance_sampling,
                weight=weight
            )
        value += strategy_t[information_set][a] * values_Itoa[a]

    # Update regrets now that we have computed the counterfactual value of the
    # information set as well as the counterfactual values of playing each
    # action in the information set. First initialise regrets with this
    # information set if necessary.
    if information_set not in regrets:
        regrets[information_set] = ActionFloat.initialise_zero(available_actions)
    if player == i:
        if information_set not in action_counts:
            action_counts[information_set] = ActionFloat.initialise_zero(available_actions)

        action_counts_to_add = {a: 0.0 for a in available_actions}
        regrets_to_add = {a: 0.0 for a in available_actions}
        for a in available_actions:
            pi_minus_i = pi_c * pi_1 if i == 2 else pi_c * pi_2
            pi_i = pi_1 if i == 1 else pi_2
            regrets_to_add[a] = weight * (values_Itoa[a] - value) * pi_minus_i
            # action_counts_to_add[a] = pi_c * pi_i * strategy_t[information_set][a]
            action_counts_to_add[a] = weight * pi_i * strategy_t[information_set][a]

        # Update the regrets and action counts.
        regrets[information_set] = ActionFloat.sum(regrets[information_set], ActionFloat(regrets_to_add))
        action_counts[information_set] = ActionFloat.sum(
            action_counts[information_set],
            ActionFloat(action_counts_to_add)
        )

        # Update strategy t plus 1
        strategy_t_1[information_set] = cfr_util.compute_regret_matching(regrets[information_set])

    # Return the value
    return value


def evaluate_strategies(game, strategy, num_iters=500):
    """ Given a strategy in the form of a dictionary from information sets to
    probability distributions over actions, sample a number of games to
    approximate the expected value of player 1.
    """
    return game.expected_value(strategy, strategy, num_iters)
