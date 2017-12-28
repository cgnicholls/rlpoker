# coding: utf-8
# This file computes a maximin strategy for Leduc Hold'em using counterfactual
# regret minimisation.

import numpy as np
from cfr import cfr
from leduc_cfr_game import LeducCFR
import pickle
from time import time
from argparse import ArgumentParser


def play_against(game, user_player, strategy):
	""" Given a strategy in the form of a dictionary from information sets to probability
	distributions over actions, the user can play against it. The user plays as 'user_player'
	(either 1 or 2).
	"""
	# First reset the game
	player, information_set, terminal, payoffs = game.reset()
	while not terminal:
		if player == user_player:
			print("Player {} to play".format(player))
			print("Information set: {}".format(information_set))
			print("Please enter action (0 = fold, 1 = call, 2 = raise):")
			action = int(input())
			#action = 1
		else:
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
	return payoffs

if __name__ == "__main__":

	parser = ArgumentParser()

	parser.add_argument("--load_file", help="Loads the file specified afterwards")
	parser.add_argument("--num_iters", help="The number of iterations to run CFR for")

	args = parser.parse_args()
	# If we don't load a file, then we compute a maximin strategy using CFR.
	if not args.load_file:
		game = LeducCFR()
		num_iters = 10000
		if args.num_iters:
			num_iters = int(args.num_iters)
		average_strategy = cfr(game, num_iters=num_iters)
		print(average_strategy)

		with open('average_strategy-' + str(int(time())) + '.strategy', 'wb') as f:
			pickle.dump(average_strategy, f)
	else:
		# Otherwise we load the specified strategy and play against it
		with open(args.load_file, 'rb') as f:
			average_strategy = pickle.load(f)
			print(average_strategy)

			# Now the user can play against the strategy
			game = LeducCFR()
			all_payoffs = {1: [], 2: []}
			while True:
				payoffs = play_against(game, 2, average_strategy)
				all_payoffs[1].append(payoffs[1])
				all_payoffs[2].append(payoffs[2])
				print("Payoffs: {}".format(payoffs))
				print("Average payoffs: {}".format({i: np.mean(all_payoffs[i]) for i in all_payoffs}))