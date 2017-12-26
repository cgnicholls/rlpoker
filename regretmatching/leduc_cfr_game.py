# coding: utf-8

import cfr_game

# LeducHoldEm is the following game.
# The deck consists of (J, J, Q, Q, K, K).
# Each player gets 1 card. There are two betting rounds and the number of raises in each round is at most 2.
# In the second round, one card is revealed on the table and is used to create a hand.
# There are two types of hands: pairs and highest card. There are three actions: fold, call and raise.
# Each of the two players antes 1. In the first round, the betting amount is 2 (including the ante for the first bet).
# In round 2 it is 4.
# Player 1 starts round 1 and player 2 starts round 2.

# Betting in Poker:
# Bet: place a wager into the pot, or raise, by matching the outstanding bet and placing an extra bet into the pot.
# Call: matching the amount of all outstanding bets (or check, by passing when there is no bet)
# Fold: pass when there is an outstanding bet and forfeiting the current game.

# So keep track of whether the previous player called or not. Note that the big blind is an initial bet.

# Actions are: 0 - fold
# 1 - call
# 2 - raise

# Rounds: 1 - pre-flop
# 2 - flop

# A dictionary that returns the other player.
other_player = {1: 2, 2: 1}

class LeducCFR(CFRGame):
	# We interpret an empty action sequence as meaning the root of the game.
	# The next step is to sample the cards for the two players.

	# The deck is J, J, Q, Q, K, K. We represent J = 11, Q = 12, K = 13.
	deck = [11,12,13]
	max_raises_per_round = 2
	num_rounds = 2


	# We just make a dictionary with the available actions (assuming a 2 raise maximum).
	# If the bet sequence isn't in this dictionary, it is invalid.
	# If we want to have a larger raise maximum, we should just make a tree to parse the
	# bet sequence, but this will do for now.
	available_actions = {
		(): [1,2],
		(1): [1,2],
		(1, 1): [],
		(1, 2): [0,1,2],
		(1, 2, 0): [],
		(1, 2, 1): [],
		(1, 2, 2): [0, 1],
		(1, 2, 2, 0): [],
		(1, 2, 2, 1): [],

		(2): [0,1,2],
		(2, 0): [],
		(2, 1): [],
		(2, 2): [0, 1],
		(2, 2, 0): [],
		(2, 2, 1): []
	}

	# We represent actions as: fold = 0, call/check = 1, bet/raise = 2.

	def __init__(self):
		pass

	@static
	def interpret_action_sequence(action_sequence):
		""" Validate the action sequence -- check it makes sense. If it's invalid, return an empty dictionary.
		Otherwise, returns a dictionary with: player to play, current pot, is it terminal, the information set.
		"""
		if len(action_sequence) == 0:
			return {'player': 0, 'pot': 0, 'terminal': False, 'information_set': (), 'round': 0}

		# An action sequence of length 1 is invalid
		if len(action_sequence) == 1:
			return dict()

		if len(action_sequence) == 2:
			if (not action_sequence[0] in LeducCFR.deck) and (not action_sequence[1] in LeducCFR.deck):
				return dict()

		# We first split into betting rounds.
		# Valid betting rounds are, as strings,
		# 'cc' <- first player called and second player checked.
		# 'r'**n + 'c', where n <= max_raises <- a sequence of n raises followed by a call.
		# 'r'**n + 'f', where n <= max_raises <- a sequence of n raises followed by a fold.
		# Note that even if n == max_raises, the last action played must be to call, so that
		# the next round is reached with both players having made equal bets.

		bet_sequences = []
		bet_sequence = []
		cards = []
		for a in action_sequence:
			if a < 10:
				bet_sequence.append(a)
			else:
				cards.append(a)
				if len(bet_sequence) > 0:
					bet_sequences.append(bet_sequence)
					bet_sequence = []

		# Now interpret the bet sequences. There are either 0, 1 or 2 in a valid sequence
		round = len(bet_sequences)
		return {''}

	@static
	def compute_bets(bet_sequences):
		""" Given 0, 1 or 2 bet sequences, compute the current bets by players 1 and 2.
		"""
		# Both players ante 1
		bets = {1: 1, 2: 1}

		assert len(bet_sequences) <= 2

		for i, bet_sequence in enumerate(bet_sequences):
			# Player 1 starts the round 1 and player 2 starts round 2.
			player = 1 if i == 0 else 2
			# The raise amount is 2 in the first round and 4 in the second round.
			raise_amount = 2 if i == 0 else 4
			for bet_action in bet_sequence:
				if bet_action == 0:
					# Fold
					return bets
				else if bet_action == 1:
					# Check/call
					# Update the bet of the current player to equal the bet of the other player
					bets[player] = bets[other_player[player]]
				else if bet_action == 2:
					# Bet/Raise
					# First call the bet of the other player, and then add the betting amount to
					# player's bet.
					bets[player] = bets[other_player[player]]
					bets[player] += raise_amount
				# Switch players
				player = other_player[player]
		return bets

	@static
	def available_actions(bet_sequence):
		""" Given a single bet sequence (for one round), returns the available actions.
		"""
		assert bet_sequence in LeducCFR.available_actions
		return LeducCFR.available_actions[bet_sequence]

	@static
	def payoffs(action_sequence):
		""" If the action sequence is terminal, returns the payoffs for players
		1 and 2 in a dictionary with keys 1 and 2.
		"""
		assert LeducCFR.is_terminal(action_sequence)

		bet_sequences = LeducCFR.interpret_action_sequence()
		bets = LeducCFR.compute_bets(bet_sequences)

		# If the action sequence

	@static
	def is_terminal(bet_sequences):
		""" Returns True/False if the bet_sequences is terminal or not.
		"""
		# We just check that there are no available actions to a player in the
		# last bet sequence, and that there are two bet sequences
		if len(bet_sequences) < 2:
			return False
		return len(LeducCFR.available_actions[bet_sequences[-1]]) == 0

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