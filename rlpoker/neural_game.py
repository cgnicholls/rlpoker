
import typing

import numpy as np


class ActionIndexer:
    """
    An ActionIndexer maps possible actions in a game to indices, so we can use neural networks and map between
    actions and action indices.
    """

    def __init__(self, actions: typing.List[typing.Any]):
        assert len(set(actions)) == len(actions), "Actions must be unique."

        self._actions = actions
        self.action_indices = dict(zip(actions, range(len(actions))))

    def get_index(self, action):
        return self.action_indices[action]

    def get_action(self, index):
        return self._actions[index]

    @property
    def actions(self):
        return self._actions

    @property
    def action_dim(self):
        return len(self._actions)


class InfoSetVectoriser:

    def __init__(self, vectors: typing.Dict):
        """Maps info set ids to numpy arrays.

        Args:
            vectors: Dict. A dictionary mapping info set ids to numpy arrays.
        """
        self.vectors = vectors

        self._state_shape = self._get_state_shape(list(self.vectors.values()))

    @staticmethod
    def _get_state_shape(vectors: typing.List[np.ndarray]):
        if len(vectors) == 0:
            raise ValueError("No vectors passed.")

        for v in vectors:
            if type(v) != np.ndarray:
                raise ValueError("All vectors must be ndarrays.")

        shapes = {v.shape for v in vectors}

        if len(shapes) > 1:
            raise ValueError("Shapes of all state vectors must be the same.")

        return list(shapes)[0]

    def get_vector(self, info_set_id) -> np.ndarray:
        return self.vectors[info_set_id]

    @property
    def state_shape(self):
        return self._state_shape
