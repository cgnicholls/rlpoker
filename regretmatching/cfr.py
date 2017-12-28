# coding: utf-8
# This program plays the counterfactual regret minimisation algorithm on any two player zero-sum game.

import numpy as np

def cfr(game, num_iters=10000):
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

	average_strategy = None
	average_strategy_snapshot = None

	# Each information set is uniquely identified with an action tuple.
	values = {1: [], 2: []}
	for t in range(num_iters):
		for i in [1,2]:
			cfr_recursive(game, [], i, t, 1.0, 1.0, regrets, action_counts, strategy_t, strategy_t_1)
		
		if (t % 100 == 0) and (not average_strategy is None):
			print("t: {}".format(t))
			if not average_strategy_snapshot is None:
				snapshot_distance = compare_strategies(average_strategy, average_strategy_snapshot)
				print("Distance between strategies (t - 100): {}".format(snapshot_distance))
			average_strategy_snapshot = average_strategy.copy()
		average_strategy = compute_average_strategy(action_counts)
		
		# Update strategy_t to equal strategy_t_1. We update strategy_t_1 inside cfr_recursive.
		# We take a copy because we update it inside cfr_recursive, and want to hold on to
		# strategy_t_1 separately to compare.
		strategy_t = strategy_t_1.copy()

		#print("Values {}".format(values))
		if t % 1000 == 0:
			all_payoffs = evaluate_strategies(game, strategy_t, num_iters=5000)
			print("t: {}".format(t))
			print("Average value 1: {}, std: {}".format(np.mean(all_payoffs[1]), np.std(all_payoffs[1]) / np.sqrt(len(all_payoffs[1]))))
			print("Average value 2: {}, std: {}".format(np.mean(all_payoffs[2]), np.std(all_payoffs[2]) / np.sqrt(len(all_payoffs[2]))))

	return average_strategy

def compute_average_strategy(action_counts):
	average_strategy = dict()
	for information_set in action_counts:
		num_actions = sum([v for k,v in action_counts[information_set].items()])
		if num_actions > 0:
			average_strategy[information_set] = {k: float(v) / float(num_actions) for k,v in action_counts[information_set].items()}

	return average_strategy

def compare_strategies(s1, s2):
	""" Takes the average Euclidean distance between the probability distributions.
	"""
	common_keys = [k for k in s1.keys() if k in s2.keys()]
	distances = []
	for information_set in common_keys:
		prob_dist_diff = [float(s1[information_set][a] - s2[information_set][a])**2 for a in s1[information_set]]
		distances.append(np.sqrt(np.mean(prob_dist_diff)))
	return np.mean(distances)

# The Game object holds a game state at any point in time, and can return an information set label
# for that game state, which uniquely identifies the information set and is the same for all states
# in that information set.
def cfr_recursive(game, history, i, t, pi_1, pi_2, regrets, action_counts, strategy_t, strategy_t_1):
	#print("cfr_recursive for history: {}".format(history))
	# If the history is terminal, just return the payoffs
	if game.is_terminal(history):
		return game.payoffs(history)[i]
	# If the next player is chance, then sample the chance action
	elif game.which_player(history) == 0:
		a = game.sample_chance_action(history)
		return cfr_recursive(game, history + [a], i, t, pi_1, pi_2, regrets, action_counts, strategy_t, strategy_t_1)

	information_set = game.information_set(history)
	#print("Information set: {}".format(information_set))
	player = game.which_player(history)
	value = 0
	#print("player: {}".format(player))
	available_actions = game.available_actions(history)
	#print("Available actions: {}".format(available_actions))
	values_Itoa = {a: 0 for a in available_actions}
	for a in available_actions:
		#print("Action: {}".format(a))
		# Have to ensure that strategy_t[information_set][a] is initialised
		if not information_set in strategy_t:
			strategy_t[information_set] = {ad: 1.0/float(len(available_actions)) for ad in available_actions}
		if player == 1:
			values_Itoa[a] = cfr_recursive(game, history + [a], i, t, strategy_t[information_set][a] * pi_1, pi_2, regrets, action_counts, strategy_t, strategy_t_1)
		else:
			values_Itoa[a] = cfr_recursive(game, history + [a], i, t, strategy_t[information_set][a] * pi_1, pi_2, regrets, action_counts, strategy_t, strategy_t_1)
		value += strategy_t[information_set][a] * values_Itoa[a]

	# Update regrets now that we have computed the counterfactual value of the information
	# set as well as the counterfactual values of playing each action in the information set.
	# First initialise regrets with this information set if necessary.
	if not information_set in regrets:
		regrets[information_set] = {ad: 0.0 for ad in available_actions}
	if player == i:
		for a in available_actions:
			pi_minus_i = pi_1 if i == 2 else pi_2
			pi_i = pi_1 if i == 1 else pi_2
			regrets[information_set][a] += (values_Itoa[a] - value) * pi_minus_i
			if not information_set in action_counts:
				action_counts[information_set] = {ad: 0.0 for ad in available_actions}
			action_counts[information_set][a] += pi_i * strategy_t[information_set][a]

		# Update strategy t plus 1
		strategy_t_1[information_set] = compute_regret_matching(regrets[information_set])

	# Return the value
	return value

def compute_regret_matching(regrets):
	""" Given regrets r_i for actions a_i, we compute the regret matching strategy as follows.
	Define denominator = sum_i max(0, r_i). If denominator > 0, play action a_i proportionally to max(0, r_i).
	Otherwise, play all actions uniformly.
	"""

	# If no regrets are positive, just return the uniform probability distribution on
	# available actions.
	if max([v for k, v in regrets.items()]) <= 0.0:
		return {a: 1.0 / float(len(regrets)) for a in regrets}
	else:
		# Otherwise take the positive part of each regret (i.e. the maximum of the regret and zero),
		# and play actions with probability proportional to positive regret.
		denominator = sum([max(0.0, v) for k,v in regrets.items()])
		return {k: max(0.0, v) / denominator for k,v in regrets.items()}

def evaluate_strategies(game, strategy, num_iters=500):
	""" Given a strategy in the form of a dictionary from information sets to probability
	distributions over actions, sample a number of games to approximate the expected value
	of player 1.
	"""
	all_payoffs = {1: [], 2: []}
	for t in range(num_iters):
		# First reset the game
		player, information_set, terminal, payoffs = game.reset()
		while not terminal:
			# Sometimes the information set doesn't exist in the strategy (since we're chance sampling)
			if not information_set in strategy:
				action = None
			else:
				prob_dist = []
				for a in [0,1,2]:
					if a in strategy[information_set]:
						prob_dist.append(strategy[information_set][a])
					else:
						prob_dist.append(0.0)
				action = np.random.choice([0,1,2], p=prob_dist)
			player, information_set, terminal, payoffs = game.play_action(action)
		all_payoffs[1].append(payoffs[1])
		all_payoffs[2].append(payoffs[2])
	return all_payoffs