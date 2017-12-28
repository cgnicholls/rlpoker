# coding: utf-8
# This file computes a maximin strategy for Leduc Hold'em using counterfactual
# regret minimisation.

import numpy as np
from cfr import cfr
from leduc_cfr_game import LeducCFR
import pickle
from time import time
from argparse import ArgumentParser

if __name__ == "__main__":

	parser = ArgumentParser()

	parser.add_argument("--load_file", help="Loads the file specified afterwards")
	parser.add_argument("--num_iters", help="The number of iterations to run CFR for")

	args = parser.parse_args()
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
		with open(args.load_file, 'rb') as f:
			average_strategy = pickle.load(f)
			print(average_strategy)