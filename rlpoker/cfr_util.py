from rlpoker import extensive_game
from rlpoker import cfr_game


def compute_regret_matching(action_regrets: extensive_game.ActionFloat, epsilon=1e-7, highest_regret=False):
    """Given regrets r_i for actions a_i, we compute the regret matching strategy as follows.

    If sum_i max(0, r_i) > 0:
        Play action a_i proportionally to max(0, r_i)
    Else:
        Play all actions uniformly.

    Args:
        regrets: dict
        epsilon: the minimum probability to return for each action, for numerical stability.
        highest_regret: if True, then when all regrets are negative, return epsilon for all but the highest regret
            actions.

    Returns:
        extensive_game.ActionFloat. The probability of taking each action in this information set.
    """
    # If no regrets are positive, just return the uniform probability distribution on available actions.
    if max([v for k, v in action_regrets.items()]) <= 0.0:
        if highest_regret:
            probs = {action: epsilon for action in action_regrets}
            best_action = max(action_regrets, key=action_regrets.get)
            probs[best_action] = 1.0
            return normalise_probs(extensive_game.ActionFloat(probs), epsilon=epsilon)
        else:
            return extensive_game.ActionFloat.initialise_uniform(action_regrets.action_list)
    else:
        # Otherwise take the positive part of each regret (i.e. the maximum of the regret and zero),
        # and play actions with probability proportional to positive regret.
        return normalise_probs(extensive_game.ActionFloat({k: max(0.0, v) for k, v in action_regrets.items()}),
                               epsilon=epsilon)


def normalise_probs(probs: extensive_game.ActionFloat, epsilon=1e-7):
    """Sets the minimum prob to be epsilon, and then normalises by dividing by the sum.

    Args:
        probs: extensive_game.ActionFloat. Must all be non-negative.

    Returns:
        norm_probs: extensive_game.ActionFloat.
    """
    assert min(probs.values()) >= 0.0
    probs = {a: max(prob, epsilon) for a, prob in probs.items()}
    return extensive_game.ActionFloat({a: prob / sum(probs.values()) for a, prob in probs.items()})


class AverageStrategy:
    """
    An AverageStrategy is used to compute a time-average of a sequence of strategies in an imperfect information game.
    """

    def __init__(self, game: extensive_game.ExtensiveGame):
        self.game = game

        self.alpha = dict()  # A dictionary mapping information sets to ActionFloats.
        self.beta = dict()  # A dictionary mapping information sets to floats.

    def update(self, node, pi: float, action_probs: extensive_game.ActionFloat):
        """
        Update alpha and beta for the given node. Let node be a player i game node.

        Args:
            node: ExtensiveGameNode.
            pi: float. Reach probability for player i at node.
            action_probs: strategy probabilities for player i at node.
        """
        info_set = cfr_game.get_information_set(self.game, node)
        additional = extensive_game.ActionFloat.scalar_multiply(action_probs, pi)
        if info_set not in self.alpha:
            self.alpha[info_set] = additional
        else:
            self.alpha[info_set] = extensive_game.ActionFloat.sum(self.alpha[info_set], additional)

        self.beta[info_set] = self.beta.get(info_set, 0.0) + pi


def update_average_strategy(game: extensive_game.ExtensiveGame, average_strategy: AverageStrategy,
                            strategy: extensive_game.Strategy):
    """
    Updates the average strategy with the given strategy.

    Args:
        game: ExtensiveGame.
        average_strategy: AverageStrategy. The average strategy for both players.
        strategy: Strategy. The strategy for both players.

    Returns:
        average_strategy: the updated average strategy.
    """
    update_average_strategy_recursive(game, game.root, average_strategy, strategy, pi1=1.0, pi2=1.0)

    return average_strategy


def update_average_strategy_recursive(
        game: extensive_game.ExtensiveGame,
        node: extensive_game.ExtensiveGameNode,
        average_strategy: AverageStrategy,
        strategy: extensive_game.Strategy,
        pi1: float,
        pi2: float):
    """
    Walk over the game tree and update alpha, beta for the AverageStrategy.

    Let I be a player i information set. Then let pi_i^sigma(I) be the reach probability, which also equals:

        pi_i^sigma(I) = sum_{h in I} pi_i^sigma(h),

    where pi_i^sigma(h) is the product of all probabilities in player i nodes for player i to play the required
    action to reach h from the root of the tree.

    Define

        alpha_T[I][a] = sum_{t=1}^T pi_i^{sigma^t}[I] sigma^t[I][a].
        beta_T[I] = sum_{t=1}^T pi_i^{sigma^t}[I].

    The average strategy is average_sigma[I][a] = alpha_T[I][a] / beta_t[I].

    Args:
        game:
        node:
        average_strategy:
        strategy:
        pi1: the reach probability for this node according to just player 1's strategy.
        pi2: the reach probability for this node according to just player 2's strategy.
    """

    if node.player == -1:
        # Terminal node
        return
    elif node.player == 0:
        # Chance node
        for a, chance_prob in node.chance_probs.items():
            update_average_strategy_recursive(game, node.children[a], average_strategy, strategy, pi1, pi2)
    elif node.player == 1:
        info_set = cfr_game.get_information_set(game, node)
        action_probs = strategy.get_action_probs(info_set)

        # Update alpha and beta
        average_strategy.update(node, pi1, action_probs)
        for a, action_prob in action_probs.items():
            update_average_strategy_recursive(
                game, node.children[a], average_strategy, strategy, pi1 * action_prob, pi2)
    elif node.player == 2:
        info_set = cfr_game.get_information_set(game, node)
        action_probs = strategy.get_action_probs(info_set)

        # Update alpha and beta
        average_strategy.update(node, pi2, action_probs)
        for a, action_prob in action_probs.items():
            update_average_strategy_recursive(
                game, node.children[a], average_strategy, strategy, pi1, pi2 * action_prob)
