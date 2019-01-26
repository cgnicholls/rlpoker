"""This class is a wrapper around an ExtensiveGame to make it compatible
with NFSP.
"""
import abc

from rlpoker.extensive_game import ExtensiveGame

class NFSPGame(ExtensiveGame):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @abc.abstractmethod
    def get_action_dim(self):
        """Return the dimension of the action space."""
        pass