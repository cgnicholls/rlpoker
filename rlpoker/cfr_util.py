from rlpoker.extensive_game import ActionFloat


def compute_regret_matching(action_regrets: ActionFloat, epsilon=1e-7, highest_regret=False):
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
        ActionFloat. The probability of taking each action in this information set.
    """
    # If no regrets are positive, just return the uniform probability distribution on available actions.
    if max([v for k, v in action_regrets.items()]) <= 0.0:
        if highest_regret:
            probs = {action: epsilon for action in action_regrets}
            best_action = max(action_regrets, key=action_regrets.get)
            probs[best_action] = 1.0
            return normalise_probs(ActionFloat(probs), epsilon=epsilon)
        else:
            return ActionFloat.initialise_uniform(action_regrets.action_list)
    else:
        # Otherwise take the positive part of each regret (i.e. the maximum of the regret and zero),
        # and play actions with probability proportional to positive regret.
        return normalise_probs(ActionFloat({k: max(0.0, v) for k, v in action_regrets.items()}), epsilon=epsilon)


def normalise_probs(probs: ActionFloat, epsilon=1e-7):
    """Sets the minimum prob to be epsilon, and then normalises by dividing by the sum.

    Args:
        probs: ActionFloat. Must all be non-negative.

    Returns:
        norm_probs: ActionFloat.
    """
    assert min(probs.values()) >= 0.0
    probs = {a: max(prob, epsilon) for a, prob in probs.items()}
    return ActionFloat({a: prob / sum(probs.values()) for a, prob in probs.items()})
