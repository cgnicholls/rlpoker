
import typing

from rlpoker import extensive_game


class NeuralGame(extensive_game.ExtensiveGame):

    def __init__(self):
        pass

    def get_vector(self, info_set_id):
        pass

    @property
    def state_dim(self):
        pass

    @property
    def action_indexer(self):
        pass


class ActionIndexer:
    """
    An ActionIndexer maps possible actions in a game to indices, so we can use neural networks and map between
    actions and action indices.
    """

    def __init__(self, actions: typing.List[typing.Any]):
        assert len(set(actions)) == len(actions), "Actions must be unique."

        self.actions = actions
        self.action_indices = dict(zip(actions, range(len(actions))))

    def get_index(self, action):
        return self.action_indices[action]

    def get_action(self, index):
        return self.actions[index]

    def get_action_dim(self):
        return len(self.actions)
