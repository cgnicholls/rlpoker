# coding: utf-8
# This program plays Kuhn Poker using the CFR algorithm.
import numpy as np

# Information sets: this has the information of that player's card, and the
# previous moves by all players. We just write this as a string for now, e.g.
# '2PB' denotes that your card is a 2, the first player passed, the second
# player bet, and now it's your turn.
# Histories. We represent this as a string also, where the first two letters are
# the cards, and the 

# Computes the information set at this history of moves. We can deduce which
# player is to move by the length of the history.
# Possible histories are: '', 'P', 'B', 'PB', 'PP', 'BB', 'BP', 'PBB', 'PBP'.
def get_info_set(cards, history):
    if len(history) % 2 == 0:
        return str(cards[0]) + history
    else:
        return str(cards[1]) + history

# Return whether the current history is terminal or not.
def is_terminal(history):
    if history == 'PP' or history == 'BB' or history == 'BP' or len(history) == 3:
        return True
    else:
        return False

# Player 1 or player 2 depending on the length of the history.
def get_player(history):
    return (len(history) % 2) + 1

# Given the cards and the history compute the utilities u1, u2 for player 1 and
# player 2, respectively.
def utility(cards, history):
    # If both players passed, then the pot contains 2 chips, and the player with
    # the highest card wins.
    if history == 'PP':
        if cards[0] > cards[1]:
            return np.array([1,-1])
        else:
            return np.array([-1,1])
    # If both players bet, then the pot contains 4 chips, so the winning player
    # gains 2 and the losing player loses 2.
    if history == 'BB' or history == 'PBB':
        if cards[0] > cards[1]:
            return np.array([2,-2])
        else:
            return np.array([-2,2])
    # If player 2 bet and player 1 didn't, then player 2 wins 3 chips, so gains
    # 1 and player 1 loses 1 chip.
    if history == 'PBP':
        return np.array([-1, 1])
    # If player 1 bet and player 2 didn't, then player 2 loses 1 chip and player
    # 1 gains 1 chip.
    if history == 'BP':
        return np.array([1, -1])

# Sample two cards out of 1,2,3.
def sample_cards():
    card1 = np.random.choice([1,2,3])
    remaining_cards = [1,2,3]
    remaining_cards.remove(card1)
    card2 = np.random.choice(remaining_cards)
    return [card1,card2]

# Computes the strategy given the cumulative regrets as the normalised positive
# cumulative regrets. Assumes given an np array.
def regret_matching(c_regret):
    if (c_regret > 0).any():
        strategy = (c_regret > 0) * c_regret
    else:
        strategy = np.ones(np.shape(c_regret))
    return strategy / np.sum(strategy)

# Computes the strategy given the cumulative regrets as the normalised positive
# cumulative regrets. Assumes given an np array.
def compute_strategy(c_regrets):
    strategy = {}
    for key, c_regret in c_regrets.iteritems():
        strategy[key] = regret_matching(c_regret)
    return strategy

# Assume the possible moves are always 0 for pass and 1 for bet.
# The cfr algorithm computes the counterfactual regret for the given player at
# the information set containing the history node.
# pi1 contains the probability of reaching this information set given that
# player 2 always plays moves to reach this history with probability 1.
# pi2 is pi1 with players interchanged.
def cfr(cards, history, player, t, pi1, pi2, strategy, c_regrets, c_strategy):
    if is_terminal(history):
        return utility(cards, history)[(player - 1 % 2)]

    # If no history so far, then we have to sample the cards
    if len(cards) == 0:
        cards = sample_cards()
        return cfr(cards, history, player, t, pi1, pi2, strategy, c_regrets,
        c_strategy)

    # Let info_set be the info set containing h.
    info_set = get_info_set(cards, history)
    
    val = 0
    val_a = {'P': 0.0, 'B': 0.0}
    for idx, a in enumerate(['P', 'B']):
        sigma_I_a = strategy[info_set][idx]
        if get_player(history) == 1:
            val_a[a] = cfr(cards, history + a, player, t, pi1*sigma_I_a, pi2,
            strategy, c_regrets, c_strategy)
        else:
            val_a[a] = cfr(cards, history + a, player, t, pi1, pi2*sigma_I_a,
            strategy, c_regrets, c_strategy)
        val += sigma_I_a * val_a[a]

    # Counterfactual pi is pi_{-i} from the paper. If player is 1, it should be
    # pi2, if player is 2, it should be pi1.
    counterfactual_pi = {1: pi2, 2: pi1}
    pi_player = {1: pi1, 2: pi2}
    if get_player(history) == player:
        for idx, a in enumerate(['P', 'B']):
            c_regrets[info_set][idx] += (val_a[a] - val) * \
            counterfactual_pi[player]
            c_strategy[info_set][idx] += pi_player[player] * \
            strategy[info_set][idx]
        strategy[info_set] = regret_matching(c_regrets[info_set])

    return val

def compute_all_info_sets_from_cards(cards, history, all_info_sets):
    if is_terminal(history):
        return
    info_set = get_info_set(cards, history)
    all_info_sets.append(info_set)
    for a in ['P', 'B']:
        compute_all_info_sets_from_cards(cards, history + a, all_info_sets)
    return
    
def compute_all_info_sets():
    all_info_sets = []
    for cards in [[1,2], [1,3], [2,1], [2,3], [3,1], [3,2]]:
        card_info_sets = []
        compute_all_info_sets_from_cards(cards, '',card_info_sets)
        all_info_sets = all_info_sets + card_info_sets
    no_duplicates = []
    for I in all_info_sets:
        if I not in no_duplicates:
            no_duplicates.append(I)
    return no_duplicates

# Initialise zeros for each information set.
def initialise_arrays():
    all_info_sets = compute_all_info_sets()
    strategy = {}
    c_regrets = {}
    c_strategy = {}
    for info_set in all_info_sets:
        strategy[info_set] = np.array([0.5,0.5])
        c_regrets[info_set] = np.array([0.0,0.0])
        c_strategy[info_set] = np.array([0.0,0.0])
    return strategy, c_regrets, c_strategy

def run_cfr(T):
    strategy, c_regrets, c_strategy = initialise_arrays()
    vals = {1: [], 2: []}
    for t in range(T):
        for player in [1,2]:
            avg_util = cfr([], '', player, t, 1.0, 1.0, strategy, c_regrets,
            c_strategy)
            vals[player].append(avg_util)
    print "Player 1 avg value", np.mean(vals[1]), " (std: ", np.std(vals[1]), ")"
    print "Player 2 avg value", np.mean(vals[2]), " (std: ", np.std(vals[2]), ")"
    return c_regrets, c_strategy

def normalise_strategy(strategy):
    normalising_sum = np.sum(strategy)
    if normalising_sum > 0:
        return strategy / normalising_sum
    else:
        uniform_dist = np.ones(np.shape(strategy))
        return uniform_dist / np.sum(uniform_dist)

c_regrets, c_strategy = run_cfr(10000)
avg_strategy = {k:normalise_strategy(v) for k, v in c_strategy.iteritems()}
print avg_strategy
