# coding: utf-8
# This program implements regret matching to play rock paper scissors.
import numpy as np

def compute_strategy(cumul_regrets):
    # If the cumulative regrets are all nonpositive, then play uniformly at
    # random
    if (cumul_regrets <= 0).all():
        return np.array([1.0, 1.0, 1.0]) / 3.0
    else:
        cumul_regrets[cumul_regrets < 0.0] = 0.0
        return cumul_regrets / np.sum(cumul_regrets)

# Returns the utility for player 1 if player 1 plays a1 and player 2 plays a2.
# We use rock = 0, paper = 1, scissors = 2.
def compute_utility(a1, a2):
    # If both players play the same action, it's a draw.
    if a1 == a2:
        return 0
    # Otherwise note that rock loses to paper loses to scissors loses to rock.
    # So there is cyclical symmetry. So if a2 = a1 + 1 mod 3, then player 2
    # wins, and if a2 = a1 + 2 mod 3, then player 1 wins. We thus look at a2 -
    # a1 mod 3.
    if (a2 - a1) % 3 == 1:
        return -1
    else:
        return 1

# opp_strategy: the opponent's mixed strategy. This is a length 3 numpy array
# representing the probability distribution over rock, paper and scissors.
# returns: mixed strategy found by regret matching.
def regret_matching_best_response(opp_strategy, n_iters=1000):
    # Initialise cumulative regrets to zero.
    cumul_regrets = np.array([0.0,0.0,0.0])
    # Keep track of all strategies we ever play -- we return the average.
    strategies = []
    for i in range(n_iters):
        # Compute the strategy given the cumulative regrets
        strategy = compute_strategy(cumul_regrets)
        player_action = np.random.choice([0,1,2], p=strategy)
        opp_action = np.random.choice([0,1,2], p=opp_strategy)
        utility = compute_utility(player_action, opp_action)
        regrets = np.array([compute_utility(action, opp_action) - utility for action in [0,1,2]])
        cumul_regrets += regrets
        strategies.append(strategy)
    return strategies

opp_strategy = np.array([1.0,1.0,1.0])/3.0
strategies = regret_matching_best_response(opp_strategy, n_iters=100000)
print np.mean(strategies, axis=0)

opp_strategy = np.array([0.4,0.3,0.3])
strategies = regret_matching_best_response(opp_strategy, n_iters=100000)
print np.mean(strategies, axis=0)

# returns: mixed strategy found by regret matching.
def regret_matching_self_play(n_iters=1000):
    # Initialise cumulative regrets to zero.
    cumul_regrets_1 = np.array([0.0,0.0,0.0])
    cumul_regrets_2 = np.array([0.0,0.0,0.0])

    # Keep track of all strategies we ever play -- we return the average.
    strategies_1 = []
    strategies_2 = []
    for i in range(n_iters):
        # Compute the strategy given the cumulative regrets
        strategy_1 = compute_strategy(cumul_regrets_1)
        strategy_2 = compute_strategy(cumul_regrets_2)
        player_action_1 = np.random.choice([0,1,2], p=strategy_1)
        player_action_2 = np.random.choice([0,1,2], p=strategy_2)
        utility_1 = compute_utility(player_action_1, player_action_2)
        utility_2 = -utility_1
        regrets_1 = np.array([compute_utility(action, player_action_2) - utility_1 for action in [0,1,2]])
        regrets_2 = np.array([compute_utility(action, player_action_1) - utility_2 for action in [0,1,2]])
        cumul_regrets_1 += regrets_1
        cumul_regrets_2 += regrets_2
        strategies_1.append(strategy_1)
        strategies_2.append(strategy_2)
    return strategies_1, strategies_2

strategies_1, strategies_2 = regret_matching_self_play(n_iters=10000)
print np.mean(strategies_1, axis=0)
print np.mean(strategies_2, axis=0)
