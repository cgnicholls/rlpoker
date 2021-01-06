import typing

import numpy as np


def sample_action(strategy, available_actions: typing.Union[None, typing.List] = None):
    """Samples an action from the given strategy. If available actions is given, then we first restrict to those
    actions that are available.

    Args:
        strategy: dict with keys the actions and values the probability of taking the action.
        available_actions: None or a list of actions.

    Returns:
        action.
    """
    actions = [a for a in strategy]
    if available_actions is not None:
        actions = [a for a in actions if a in available_actions]
    probs = np.array([strategy[a] for a in actions])

    assert np.sum(probs) > 0.0, print("Oops: {}, {}".format(probs, actions))

    probs = probs / np.sum(probs)

    idx = np.random.choice(list(range(len(actions))), p=probs)
    return actions[idx]
