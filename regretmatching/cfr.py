# coding: utf-8
# This program plays the counterfactual regret minimisation algorithm on any two player zero-sum game.

def cfr(game):
	# regrets is a dictionary where the keys are the information sets and values are
	# dictionaries from actions available in that information set to the counterfactual regret for not
	# playing that action in that information set. Since information sets encode the player,
	# we only require one dictionary.
	regrets = dict()

	# Similarly, action_counts is a dictionary with keys the information
	# sets and values dictionaries from actions to action counts.
	action_counts = dict()

	# Strategy_t holds the strategy at time t; similarly strategy_t_1 holds the strategy at time t + 1.
	strategy_t = dict()
	strategy_t_1 = dict()

	# Each information set is uniquely identified with an action tuple.

	for t in range(100):
		for i in [1,2]:
			cfr_recursive(game, [], i, t, 1.0, 1.0)

# The Game object holds a game state at any point in time, and can return an information set label
# for that game state, which uniquely identifies the information set and is the same for all states
# in that information set.

def cfr_recursive(game, history, player, t, pi_1, pi_2):
	# If the history is terminal, just return the payoffs
	if game.is_terminal(history):
		return game.payoffs(history)
	# If the next player is chance, then sample the chance action
	else if game.which_player(history) == 0:
		a = game.sample_chance_action(history)
		
