"""
In this file we implement some metrics for CFR-style algorithms. We assume we are given a sequence of strategies for
times t = 1, ..., T for an extensive game that implements CFR.
"""

import collections
from typing import Any, Dict, List

from rlpoker.cfr import cfr_game
from rlpoker import extensive_game


def compute_immediate_regret(game: extensive_game.ExtensiveGame, strategies: List[extensive_game.Strategy]) -> \
        (List[float], List[Dict[Any, float]]):
    """
    Computes the immediate counterfactual regret at each information set for each player.

    This is defined as
        1 / T * max_a \sum_{t=1}^T \sum_{h \in I} u_i^{sigma^t}(ha) - u_i^{sigma^t}(h),
    where u_i^sigma(h) is the counterfactual value to player i = player(I) of being in node h.

    Args:
        game: ExtensiveGame.
        strategies: list of strategies, one for each time step. Each strategy should contain both player 1 and
            player 2 strategies.

    Returns:
        immediate_regret: float. The immediate regret at time T (the last time).
        cumulative_immediate_regrets: list where the t^th element is the cumulative immediate regret (summed over
            all information sets) at time t.
        all_immediate_regrets: list of dictionaries where the t^th dictionary maps information sets to their
            immediate regret at time t.
    """
    all_info_set_regrets = []
    for t, strategy in enumerate(strategies):
        node_regrets = collections.defaultdict(dict)
        # This function fills in the immediate regrets for each node.
        completed_strategy, num_missing = game.complete_strategy_uniformly(strategy)
        compute_node_regret_recursive(game, game.root, completed_strategy, node_regrets, 1.0, 1.0, 1.0)

        # Initialise the immediate regrets for each info set.
        info_set_regrets = collections.defaultdict(dict)
        for node in node_regrets.keys():
            info_set = game.get_info_set_id(node)
            info_set_regrets[info_set] = collections.defaultdict(float)

        # The immediate regret for an info set is just the sum of the immediate regrets of the nodes in that info set.
        for node, node_regret in node_regrets.items():
            info_set = game.get_info_set_id(node)
            for action, regret in node_regret.items():
                info_set_regrets[info_set][action] += regret

        all_info_set_regrets.append(info_set_regrets)

    # Now compute the immediate regrets on each information set.
    all_immediate_regrets = []
    cumulative_immediate_regrets = []

    # Compute cumulative regret, so that, at time t, cumulative_regret[I][a] = sum_{s=1}^t regret^s[I][a]. Then we
    # can compute the immediate regret at I as 1/t max_{a} cumulative_regret[I][a].
    cumulative_regret = collections.defaultdict(dict)
    for t, info_set_regret in enumerate(all_info_set_regrets):
        immediate_regrets = dict()
        for info_set, regrets in info_set_regret.items():
            for action, regret in regrets.items():
                if action not in cumulative_regret[info_set]:
                    cumulative_regret[info_set][action] = 0.0

                cumulative_regret[info_set][action] += regret

            immediate_regrets[info_set] = 1 / (1 + t) * max(cumulative_regret[info_set].values())

        all_immediate_regrets.append(immediate_regrets)
        cumulative_immediate_regrets.append(sum(immediate_regrets.values()))

    immediate_regret = cumulative_immediate_regrets[-1]

    return immediate_regret, cumulative_immediate_regrets, all_immediate_regrets


def compute_node_regret_recursive(
        game: extensive_game.ExtensiveGame,
        node: extensive_game.ExtensiveGameNode,
        strategy: extensive_game.Strategy,
        node_regrets: collections.defaultdict,
        pi_1: float,
        pi_2: float,
        pi_c: float,
):
    """
    Computes the immediate counterfactual regret at each node for each player. This is defined as:
        regret_i(sigma, h, a) = u_i(sigma, ha) - u_i(sigma, h),
    where h is a player i node and u_i(sigma, h) is the counterfactual value to player i of being in node h,
    given that the strategy profile is sigma. Formally,
        u_i(sigma, h) = pi_i^sigma(h) \sum_{z in Z_h} pi^sigma(h, z) v_i(z),
    where
        v_i(z) is the utility to player i of the terminal node z.

    Args:
        game: ExtensiveGame.
        node: ExtensiveGameNode.
        strategy: strategy for both players.
        node_regrets: defaultdict mapping nodes to dictionaries mapping actions to the immediate regret of
            the player not playing the given action in the given node.
        pi_1: float.
        pi_2: float.
        pi_c: float.

    Returns:
        v1: the expected utility sum_{z in Z_h} u_1(z) pi^sigma(h, z).
        v2: the expected utility sum_{z in Z_h} u_2(z) pi^sigma(h, z).
    """
    node_player = cfr_game.which_player(node)
    if cfr_game.is_terminal(node):
        return node.utility[1], node.utility[2]
    elif node_player in [1, 2]:
        v1 = 0.0
        v2 = 0.0
        information_set = cfr_game.get_information_set(game, node)
        values_1 = dict()
        values_2 = dict()
        for action, child in node.children.items():
            pi_1_new = pi_1 * strategy[information_set][action] if node_player == 1 else pi_1
            pi_2_new = pi_2 * strategy[information_set][action] if node_player == 2 else pi_2
            values_1[action], values_2[action] = compute_node_regret_recursive(
                game, child, strategy, node_regrets,
                pi_1_new,
                pi_2_new,
                pi_c,
            )
            action_prob = strategy[information_set][action] if node_player == 1 else strategy[information_set][action]
            v1 += action_prob * values_1[action]
            v2 += action_prob * values_2[action]

        # Compute the immediate regret for the player in the node h for not playing each action.
        for action in node.children.keys():
            if node_player == 1:
                node_regrets[node][action] = pi_c * pi_2 * (values_1[action] - v1)
            elif node_player == 2:
                node_regrets[node][action] = pi_c * pi_1 * (values_2[action] - v2)

        return v1, v2
    elif node_player == 0:
        # Chance player.
        v1 = 0.0
        v2 = 0.0
        for action, child in node.children.items():
            chance_prob = node.chance_probs[action]
            v1a, v2a = compute_node_regret_recursive(
                game, child, strategy, node_regrets,
                pi_1, pi_2, pi_c * chance_prob
            )
            v1 += chance_prob * v1a
            v2 += chance_prob * v2a

        return v1, v2


def compute_expected_utility(game: extensive_game.ExtensiveGame, sigma_1: extensive_game.Strategy,
                             sigma_2: extensive_game.Strategy):
    """
    Computes the expected utility at each node for each player.

    Args:
        game: ExtensiveGame.
        sigma_1: strategy for player 1.
        sigma_2: strategy for player 2.

    Returns:
        expected_utility_1: dict mapping player 1 nodes to their utility.
        expected_utility_2: dict mapping player 2 nodes to their utility.
    """
    expected_utility_1 = collections.defaultdict(float)
    expected_utility_2 = collections.defaultdict(float)

    _, _ = compute_expected_utility_recursive(game, game.root, sigma_1, sigma_2,
                                              expected_utility_1, expected_utility_2,
                                              1.0, 1.0, 1.0)

    return expected_utility_1, expected_utility_2


def compute_expected_utility_recursive(
        game: extensive_game.ExtensiveGame,
        node: extensive_game.ExtensiveGameNode,
        sigma_1: extensive_game.Strategy,
        sigma_2: extensive_game.Strategy,
        expected_utility_1: Dict[extensive_game.ExtensiveGameNode, float],
        expected_utility_2: Dict[extensive_game.ExtensiveGameNode, float],
        pi_1: float,
        pi_2: float,
        pi_c: float,
):
    """
    Computes the expected utility of the given node for each player. This is defined as
        v_i(sigma, h) = sum_{z in Z_h} u_i(z) pi^sigma(h, z),
    where Z_h is the set of terminal nodes with h as a prefix, and pi^sigma(h, z) is the product of all
    probabilities in the strategy profile sigma on the route from h to z.

    Args:
        game: the game.
        node: the current node.
        sigma_1: player 1 strategy.
        sigma_2: player 2 strategy.
        expected_utility_1: dictionary mapping player 1 nodes to their utility. We fill this in.
        expected_utility_2: dictionary mapping player 2 nodes to their utility. We fill this in.
        pi_1: the reach probability of the node for player 1.
        pi_2: the reach probability of the node for player 2.
        pi_c: the reach probability of the node for the chance player.

    Returns:
        v_1: float. The expected utility of the given node.
        v_2: float. The expected utility of the given node.
    """
    node_player = cfr_game.which_player(node)
    if cfr_game.is_terminal(node):
        return node.utility[1], node.utility[2]
    elif node_player in [1, 2]:
        v1 = 0.0
        v2 = 0.0
        information_set = cfr_game.get_information_set(game, node)
        for action, child in node.children.items():
            pi_1_new = pi_1 * sigma_1[information_set][action] if node_player == 1 else pi_1
            pi_2_new = pi_2 * sigma_2[information_set][action] if node_player == 2 else pi_2
            v1a, v2a = compute_expected_utility_recursive(
                game, child, sigma_1, sigma_2,
                expected_utility_1, expected_utility_2,
                pi_1_new,
                pi_2_new,
                pi_c,
            )
            action_prob = sigma_1[information_set][action] if node_player == 1 else sigma_2[information_set][action]
            v1 += action_prob * v1a
            v2 += action_prob * v2a

        if node_player == 1:
            expected_utility_1[node] += v1
        elif node_player == 2:
            expected_utility_2[node] += v2

        return v1, v2
    elif node_player == 0:
        # Chance player.
        v1 = 0.0
        v2 = 0.0
        for action, child in node.children.items():
            chance_prob = node.chance_probs[action]
            v1a, v2a = compute_expected_utility_recursive(game, child, sigma_1, sigma_2,
                                                          expected_utility_1, expected_utility_2,
                                                          pi_1, pi_2, pi_c * chance_prob)
            v1 += chance_prob * v1a
            v2 += chance_prob * v2a

        return v1, v2
