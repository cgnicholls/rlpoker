import abc
import os
import time
from typing import Dict, Any

import numpy as np
import wandb
from tensorboardX import SummaryWriter


class Experiment:
    def __init__(self, exp_name: str = None, base_save_path: str = 'experiments'):
        self._base_save_path = base_save_path
        self._exp_name = exp_name

        self._exp_name = exp_name
        if self._exp_name is None:
            self._exp_name = time.strftime("%Y-%m-%d-%H:%M:%S", time.gmtime())

        os.makedirs(self.save_path, exist_ok=True)

    @property
    def save_path(self):
        return os.path.join(self._base_save_path, self._exp_name)

    @property
    def exp_name(self):
        return self._exp_name


class ExperimentWriter(abc.ABC):

    @abc.abstractmethod
    def log(self, data: Dict[str, Any], global_step: int = None):
        pass


class WANDBExperimentWriter(ExperimentWriter):

    def __init__(self, experiment: Experiment, project: str = "rlpoker", entity: str = "cgnicholls",
                 tags: Dict[str, Any] = None):
        self._tags = tags if tags is not None else dict()
        self._tags['exp_name'] = experiment.exp_name
        wandb.init(project=project, entity=entity, tags=self._tags)

    def log(self, data: Dict[str, Any], global_step: int = None):
        wandb.log(data, step=global_step)


class ExperimentSummaryWriter(ExperimentWriter, SummaryWriter):
    def __init__(self, experiment: Experiment, flush_secs: int = 120):
        self._save_path = experiment.save_path

        print("To run tensorboard: tensorboard --logdir {}".format(os.path.join(os.getcwd(), self._save_path)))

        logdir = os.path.join(self._save_path, 'logs')
        if os.path.exists(logdir):
            raise ValueError(f"Experiment already exists at: {logdir}.")

        super().__init__(logdir=logdir, flush_secs=flush_secs)

    def log(self, data: Dict[str, Any], global_step: int = None):
        for tag, val in data.items():
            self.add_scalar(tag, val, global_step=global_step)


class StrategySaver:

    def __init__(self, experiment: Experiment):
        self._save_path = experiment.save_path
        self._best_exploitability = np.float('inf')
        self._best_exploitability_t = 0

    def file_name(self, t: int):
        return os.path.join(self._save_path, f'strategy_{t:05d}.pkl')

    def _save_strategy(self, strategy, file_name: str):
        raise NotImplementedError()

    def _load_strategy(self, file_name: str):
        raise NotImplementedError()

    def save_strategy_at_step(self, strategy, t: int):
        self._save_strategy(strategy, self.file_name(t))

    def load_strategy_from_step(self, t: int):
        return self._load_strategy(self.file_name(t))

    @property
    def file_name_best_exploitability(self):
        return os.path.join(self._save_path, f'best_strategy.pkl')

    def save_best_strategy(self, strategy, t: int, exploitability: float):
        if exploitability < self._best_exploitability:
            self._save_strategy(strategy, self.file_name_best_exploitability)
            self._best_exploitability = exploitability
            self._best_exploitability_t = t

    def load_best_strategy(self):
        return self._load_strategy(self.file_name_best_exploitability)
