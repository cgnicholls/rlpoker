# coding: utf-8

import numpy as np

from rlpoker.util import sample_action


def payoffs(node):
    """ If the action sequence is terminal, returns the payoffs for players
    1 and 2 in a dictionary with keys 1 and 2.
    """
    return node.utility


def is_terminal(node):
    """ Returns True/False if the action sequence is terminal or not.
    """
    return node.player == -1


def which_player(node):
    """ Returns the player who is to play following the action sequence.
    """
    return node.player


def get_available_actions(node):
    """ Returns the actions available to the player to play.
    """
    return [a for a in node.children.keys()]


def sample_chance_action(node):
    """ If the player for the game state corresponding to the action
    sequence is the chance player, then sample one of the available actions.
    Return the action.
    """
    assert node.player == 0
    return sample_action(node.chance_probs)


def get_information_set(game, node):
    """ Returns a unique hashable identifier for the information set
    containing the action sequence. This could be a tuple with the
    actions that are visible to the player. The information set belongs
    to the player who is to play following the action sequence.
    """
    return game.info_set_ids[node]
