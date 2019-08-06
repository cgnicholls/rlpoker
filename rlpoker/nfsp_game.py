"""This class is a wrapper around an ExtensiveGame to make it compatible
with NFSP.
"""
import abc

from rlpoker.extensive_game import ExtensiveGame

class NFSPGame(ExtensiveGame, metaclass=abc.ABCMeta):

    def __init__(self):
        pass

    @abc.abstractmethod
    def reset(self, first_player):
        """Resets the game with the given player as the first player."""
        pass

    @abc.abstractmethod
    def step(self, action):
        """Takes the action in the game at the current node."""
        pass
