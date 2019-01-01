"""This class is a wrapper around an ExtensiveGame to make it compatible
with NFSP.
"""
import abc

from rlpoker.extensive_game import ExtensiveGame
from games.leduc import Leduc

class NFSPGame(ExtensiveGame):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @abc.abstractmethod
    def get_action_dim(self):
        """Return the dimension of the action space."""




class LeducNFSP(NFSPGame):

    def __init__(self, num_cards):
        self.num_cards = num_cards
        self.game = Leduc.create_game(num_cards)

        self.current_node = None
        # The action history stores a list of tuples (player, action) for
        # the actions played by each player to reach this node.
        self.action_history = []
        self.reset()

    def reset(self):
        self.current_node = self.game.root
        self.action_history = []

    def step(self, action):
        """This function takes the given action for the player to play in
        self.current_node.

        Args:
            action: one of the keys in self.current_node.children.

        Returns:
            player in next node, state of next node, rewards, is the next
            node terminal.
        """
        # Check the action is available.
        assert action in self.current_node.children

        self.action_history.append((self.current_node.player, action))

        # If it's a chance node, then we sample

    @staticmethod
    def encoding(info_set_id):
        """Encodes the info set id into a vector for NFSP to use for the
        given player. The information set is only available to the player
        who is to play in this position.

        Args:
            info_set_id: tuple. The tuple representing the information set.

        Returns:
            ndarray. A 1d numpy array representing the information set.
        """



        action_list = [x[1] for x in action_history]
        bets = Leduc.compute_bets(action_list)

        # A new round occurs precisely when chance deals a card.
        round = len([i for i in action_history if i[0] == 0])
