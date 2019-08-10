import typing

import tensorflow as tf

import numpy as np


class TBSummariser:

    def __init__(self, scalar_names):
        self.placeholders = {name: tf.placeholder(dtype=tf.float32, name=name) for name in scalar_names}
        self.scalars = {name: tf.summary.scalar(name, placeholder) for name, placeholder in self.placeholders.items()}

        self.merged = tf.summary.merge([v for v in self.scalars.values()])

    def summarise(self, sess, scalar_values):

        feed_dict = {
            self.placeholders[name]: scalar_values[name] for name in self.placeholders
        }
        return sess.run(self.merged, feed_dict=feed_dict)


def sample_action(strategy, available_actions: typing.Union[None, typing.List]=None):
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

    probs = probs / np.sum(probs)

    idx = np.random.choice(list(range(len(actions))), p=probs)
    return actions[idx]
