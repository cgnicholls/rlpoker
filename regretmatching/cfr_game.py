# coding: utf-8
# To run CFR on a two-player game, we need the following functions:
# For each action-sequence (a list of actions):
# 1. Is the action-sequence terminal?
# 2. Which player is to play? (including chance)
# 3. Sample a chance outcome, if the player is chance
# 4. Give a unique label (a tuple of actions) for each action-sequence


class CFRGame:

	def __init__(self):
		pass

	def payoffs(self, action_sequence):
		""" If the action sequence is terminal, returns the payoffs for players
		1 and 2 in a dictionary with keys 1 and 2.
		"""
		pass

	def is_terminal(self, action_sequence):
		""" Returns True/False if the action sequence is terminal or not.
		"""
		pass

	def which_player(self, action_sequence):
		""" Returns the player who is to play following the action sequence.
		"""
		pass

	def sample_chance_action(self, action_sequence):
		""" If the player for the game state corresponding to the action
		sequence is the chance player, then sample one of the available actions.
		"""
		pass

	def information_set(self, action_sequence):
		""" Returns a unique hashable identifier for the information set
		containing the action sequence. This could be a tuple with the
		actions that are visible to the player. The information set belongs
		to the player who is to play following the action sequence.
		"""
		pass