# coding: utf-8

import cfr_game
import numpy as np
from cfr_game import CFRGame

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
	deck = [11,11,12,12,13,13]
	max_raises_per_round = 2
	num_rounds = 2

	# We just make a dictionary with the available actions (assuming a 2 raise maximum).
	# If the bet sequence isn't in this dictionary, it is invalid.
	# If we want to have a larger raise maximum, we should just make a tree to parse the
	# bet sequence, but this will do for now.
	available_actions_dict = {
		(): [1,2],
		(1,): [1,2],
		(1, 1): [],
		(1, 2): [0,1,2],
		(1, 2, 0): [],
		(1, 2, 1): [],
		(1, 2, 2): [0, 1],
		(1, 2, 2, 0): [],
		(1, 2, 2, 1): [],

		(2,): [0,1,2],
		(2, 0): [],
		(2, 1): [],
		(2, 2): [0, 1],
		(2, 2, 0): [],
		(2, 2, 1): []
	}

	# We represent actions as: fold = 0, call/check = 1, bet/raise = 2.

	def __init__(self):
		self.reset()

	@staticmethod
	def interpret_history(history):
		""" Validate the action sequence -- check it makes sense. If it's invalid, return an empty dictionary.
		Otherwise, returns a dictionary with: player to play, current pot, is it terminal, the information set.
		"""
		if len(history) == 0:
			return [], []

		# A history of length 1 or 2 must consist of the hole cards.
		if len(history) <= 2:
			cards = history[:]
			for card in cards:
				assert card in LeducCFR.deck
			return cards, []

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
		for a in history:
			if not a in LeducCFR.deck:
				bet_sequence.append(a)
			else:
				cards.append(a)
				if len(bet_sequence) > 0:
					bet_sequences.append(bet_sequence)
					bet_sequence = []
		
		# We can also have a bet sequence without having drawn a card, and still need to add
		# this to bet_sequences.
		if len(bet_sequence) > 0:
			bet_sequences.append(bet_sequence)

		# Now interpret the bet sequences. There are either 0, 1 or 2 in a valid sequence
		return cards, bet_sequences

	@staticmethod
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
				elif bet_action == 1:
					# Check/call
					# Update the bet of the current player to equal the bet of the other player
					bets[player] = bets[other_player[player]]
				elif bet_action == 2:
					# Bet/Raise
					# First call the bet of the other player, and then add the betting amount to
					# player's bet.
					bets[player] = bets[other_player[player]]
					bets[player] += raise_amount
				# Switch players
				player = other_player[player]
		return bets

	@staticmethod
	def available_actions(history):
		cards, bet_sequences = LeducCFR.interpret_history(history)
		# If no bet sequences so far, then there better be two hole cards, and
		# we can set bet_sequence as []
		if len(bet_sequences) == 0:
			assert len(cards) == 2
			bet_sequence = []
		else:
			# If 1 bet sequence so far then we are either part way through the bet
			# sequence, or we have finished it. If we have finished it, then there
			# better be three cards drawn and we can set the bet sequence as [].
			if len(bet_sequences) == 1:
				if len(cards) == 3:
					bet_sequence = []
				else:
					# Only part way through first round.
					bet_sequence = bet_sequences[-1]
			elif len(bet_sequences) == 2:
				# If 2 bet sequence then we are partway through the last round
				bet_sequence = bet_sequences[-1]
		return LeducCFR.available_actions_bet_sequence(bet_sequence)

	@staticmethod
	def available_actions_bet_sequence(bet_sequence):
		""" Given a single bet sequence (for one round), returns the available actions.
		"""
		bet_sequence_tuple = tuple(bet_sequence)
		assert bet_sequence_tuple in LeducCFR.available_actions_dict
		return LeducCFR.available_actions_dict[bet_sequence_tuple]

	@staticmethod
	def payoffs(history):
		""" If the action sequence is terminal, returns the payoffs for players
		1 and 2 in a dictionary with keys 1 and 2.
		"""
		assert LeducCFR.is_terminal(history)

		cards, bet_sequences = LeducCFR.interpret_history(history)
		hole_cards = {1: cards[0], 2: cards[1]}

		bets = LeducCFR.compute_bets(bet_sequences)

		# We have reached a terminal node, so we have to decide who won and give them the whole pot.
		pot = bets[1] + bets[2]

		# If 1 and 2 have the same hole cards, it's a draw, so split the pot. This means
		# both players gain 0, since they have the same amount in the pot.
		if hole_cards[1] == hole_cards[2]:
			return {1: 0, 2: 0}

		# If the last action in the game was a fold, then that player loses and the other wins
		# The first player in round 2 is 2. Hence the last player is 2 if there are an even number of moves
		# and otherwise 1.

		if bet_sequences[0][-1] == 0:
			# If the last action in bet_sequences[0] is a fold, the game is over and the other player wins.
			last_player = 1 if len(bet_sequences[0]) % 2 == 0 else 2
			winner = other_player[last_player]
		elif bet_sequences[1][-1] == 0:
			# If the last action in bet_sequences[1] is a fold, then the game is over and the other player wins.
			last_player = 2 if len(bet_sequences[1]) % 2 == 0 else 1
			winner = other_player[last_player]
		else:
			# Otherwise the last action was a call, so it goes to a showdown.
			flop = cards[2]
			if hole_cards[1] == flop:
				winner = 1
			elif hole_cards[2] == flop:
				winner = 2
			else:
				# There is no pair, so the winner is the one with highest card. We already checked the
				# hole cards aren't equal.
				winner = 1 if hole_cards[1] > hole_cards[2] else 2

		return {winner: float(pot)/2.0, other_player[winner]: -float(pot)/2.0}

	@staticmethod
	def is_terminal(history):
		""" Return whether the history is terminal
		"""
		_, bet_sequences = LeducCFR.interpret_history(history)
		return LeducCFR.is_terminal_bet_sequences(bet_sequences)

	@staticmethod
	def is_terminal_bet_sequences(bet_sequences):
		""" Returns True/False if the bet_sequences is terminal or not.
		"""
		# We just check that there are no available actions to a player in the
		# last bet sequence.
		# A terminal bet sequences occurs if any player folded, or if we are in the second
		# bet sequence and there are no available actions.
		assert len(bet_sequences) <= 2
		if len(bet_sequences) == 0:
			return False
		elif len(bet_sequences) == 1:
			# If there is one bet sequence, then it is terminal if and only if the last
			# action was to fold.
			if bet_sequences[0][-1] == 0:
				return True
			else:
				# If the last action wasn't to fold, then we continue to the next round.
				return False
		bet_sequence_tuple = tuple(bet_sequences[-1])
		return len(LeducCFR.available_actions_dict[bet_sequence_tuple]) == 0

	@staticmethod
	def which_player(history):
		""" Returns the player who is to play following the action sequence.
		Returns an error if played on a terminal history.
		"""
		assert not LeducCFR.is_terminal(history)

		cards, bet_sequences = LeducCFR.interpret_history(history)

		# If the hole cards haven't been drawn yet, then sample these
		if len(cards) < 2:
			return 0

		# Else the hole cards have been drawn. If no bet sequences, then it's
		# player 1's turn to play first
		if len(bet_sequences) == 0:
			return 1
		elif len(bet_sequences) == 1:
			# If there is a single bet sequence so far, it is either finished
			# (in which case it's chance's turn to draw the flop), or it isn't
			# finished, and one of the players still has to play. Or it is finished
			# and the flop card has already been drawn.
			available_actions = LeducCFR.available_actions_bet_sequence(bet_sequences[0])
			if len(available_actions) == 0:
				# Either the flop card has been drawn already or it hasn't.
				if history[-1] in LeducCFR.deck:
					# The last move in the history was the flop, so it's player 2 to 
					# start round 2.
					return 2
				else:
					# The last move in the history was a player ending the betting round,
					# so it's chance's turn to draw the flop.
					return 0
			else:
				return 1 if len(bet_sequences[0]) % 2 == 0 else 2
		else:
			# Otherwise there are two bet sequences, and we return the player to
			# play. Make sure it's not a terminal node. Since player 2 plays first
			# in round 2, it's player 2 to play if and only if an even number of
			# actions have been played in the second bet sequence.
			bet_sequence_tuple = bet_sequences[1]
			available_actions = LeducCFR.available_actions_bet_sequence(bet_sequence_tuple)
			assert len(available_actions) > 0
			return 2 if len(bet_sequences[1]) % 2 == 0 else 1

	@staticmethod
	def sample_chance_action(history):
		""" If the player for the game state corresponding to the action
		sequence is the chance player, then sample one of the available actions.
		Return the action.
		"""
		cards, bet_sequences = LeducCFR.interpret_history(history)

		# Assert that it is the chance player to play.
		assert LeducCFR.which_player(history) == 0

		# Copy the Leduc deck and remove cards that have already been drawn.
		deck = LeducCFR.deck[:]
		for card in cards:
			deck.remove(card)

		# Just sample a card from the deck.
		return np.random.choice(deck)

	@staticmethod
	def information_set(history):
		""" Returns a unique hashable identifier for the information set
		containing the action sequence. This could be a tuple with the
		actions that are visible to the player. The information set belongs
		to the player who is to play following the action sequence.
		"""
		player = LeducCFR.which_player(history)

		assert player in [1,2]
		if player == 1:
			return tuple([history[0]] + history[2:])
		else:
			return tuple([history[1]] + history[2:])

	def reset(self):
		""" Resets the game, and returns the information set and player to play.
		Chance actions are automatically taken.
		"""
		self.history = []
		while LeducCFR.which_player(self.history) == 0:
			self.history.append(LeducCFR.sample_chance_action(self.history))

		if LeducCFR.is_terminal(self.history):
			terminal = True
			payoffs = LeducCFR.payoffs(self.history)
			player = None
			information_set = None
		else:
			terminal = False
			payoffs = None
			player = LeducCFR.which_player(self.history)
			information_set = LeducCFR.information_set(self.history)

		return player, information_set, terminal, payoffs

	def play_action(self, action):
		""" Play the action in the game. Also plays any chance actions.
		Returns the player to play and the information set they are in.
		If action is None, then play uniformly at random among available actions.
		"""
		if action is None:
			available_actions = LeducCFR.available_actions(self.history)
			action = np.random.choice(available_actions)
		assert action in LeducCFR.available_actions(self.history)
		self.history.append(action)

		if LeducCFR.is_terminal(self.history):
			terminal = True
			payoffs = LeducCFR.payoffs(self.history)
			player = None
			information_set = None
		else:
			# Play any chance actions
			while LeducCFR.which_player(self.history) == 0:
				self.history.append(LeducCFR.sample_chance_action(self.history))
				
			terminal = False
			payoffs = None
			player = LeducCFR.which_player(self.history)
			information_set = LeducCFR.information_set(self.history)

		return player, information_set, terminal, payoffs