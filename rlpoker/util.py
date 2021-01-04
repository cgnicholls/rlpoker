import os
import time
import typing

import numpy as np
from tensorboardX import SummaryWriter


class ExperimentSummaryWriter(SummaryWriter):
    def __init__(self, exp_name: str = None, base_save_path: str = 'experiments', flush_secs: int = 120):
        self._base_save_path = base_save_path
        self._exp_name = exp_name

        self._exp_name = exp_name
        if self._exp_name is None:
            self._exp_name = time.strftime("%Y-%m-%d-%H:%M:%S", time.gmtime())

        if os.path.exists(self.save_path):
            raise ValueError(f"Experiment already exists at: {self.save_path}.")

        os.makedirs(self.save_path)

        print("To run tensorboard: tensorboard --logdir {}".format(os.path.join(os.getcwd(), self.save_path)))

        logdir = os.path.join(self.save_path, 'logs')
        super().__init__(logdir=logdir, flush_secs=flush_secs)

    @property
    def save_path(self):
        return os.path.join(self._base_save_path, self._exp_name)


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
