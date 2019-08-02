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


def sample_action(strategy):
    """Samples an action from the given strategy.

    Args:
        strategy: dict with keys the actions and values the probability of taking the action.

    Returns:
        action.
    """
    actions = [a for a in strategy]
    probs = [strategy[a] for a in actions]

    return np.random.choice(actions, p=probs)


