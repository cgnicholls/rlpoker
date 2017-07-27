# coding: utf-8

import tensorflow as tf
import numpy as np
from random import shuffle

# This isn't actually LeducHoldEm yet -- just the preflop showdown.

# Actions are: 0 - fold
# 1 - call
# 2 - raise

# Rounds: 0 - pre-flop
# 1 - flop
# 2 - turn
# 3 - river

# Small blind is 1, big blind is 2.
other_player = {1: 2, 2: 1}

class PreFlopLeducHoldEm:
    def __init__(self, num_values=4, num_suits=1):
        self.num_values = num_values
        self.num_suits = num_suits
        self.deck = None
        self.hole_card = None
        self.bets = None
        self.player = None
        self.terminal = None
        self.num_raises = None

    def name(self):
        return 'preflopleducholdem'

    def reset(self, first_player):
        # The first player posts the small blind (in heads up) and the second
        # player posts the big blind.
        self.first_player = first_player
        self.player = first_player
        self.bets = {self.player: 1, other_player[self.player]: 2}

        # Since the second player posted the big blind, this is the previous
        # betting player.
        self.previous_bettor = other_player[self.player]
        self.num_raises = 0

        # Initialise and shuffle the deck
        self.deck = [{'value': v, 'suit': s} for v in range(self.num_values) \
            for s in range(self.num_suits)]
        shuffle(self.deck)

        # Draw the hole cards for 1 and 2 (doesn't matter in which order).
        self.hole_card = {1: self.deck.pop(), 2: self.deck.pop()}
        self.round = 0

        self.terminal = False
        rewards = {1: 0.0, 2: 0.0}
        return self.state(rewards)

    # Returns the state, given the rewards. Uses self.player, self.round,
    # self.hole_card, self.board, self.bets and self.terminal.
    def state(self, rewards):
        state = encoding(self.hole_card[self.player], self.player==self.first_player, \
            self.num_raises, self.num_values, self.num_suits)

        return self.player, state, rewards, self.terminal

    def step(self, a):
        # Make sure we aren't playing when the game is over.
        if self.terminal:
            assert False
        next_player = other_player[self.player]

        # If the player's action is to raise, we actually change it to a call if
        # the maximum number of raises has already been reached.
        if a == 2 and self.num_raises >= 4:
            a = 1
        # If the action is to fold, then this hand is over, and we return the
        # rewards for each player as a dictionary: reward for player 1 and
        # reward for player 2.
        if a == 0:
            self.terminal = True
            winner = next_player
        # The player called.
        elif a == 1:
            # If they are also the previous bettor, then this round of betting
            # is over.
            if self.previous_bettor == self.player:
                # Since we are playing pre-flop Poker then we have a showdown here.
                self.terminal = True
                winner = self.showdown()
            # Otherwise the player called in response to the other player's bet,
            # and so the other player has a chance to bet again if they want.
            else:
                # We update the bet for the player to equal the bet for the
                # other player.
                self.bets[self.player] = self.bets[next_player]
        elif a == 2:
            # The player raised. We already checked that they are allowed to
            # (there are max 4 raises) and so we increment the number of raises
            # for this round and then set them as the previous bettor. We also
            # adjust the bet sizes.
            self.num_raises += 1
            self.previous_bettor = self.player
            # In all cases, raising is equivalent to first calling and then
            # raising.
            self.bets[self.player] = self.bets[next_player]
            self.bets[self.player] += 1

        # Then we switch players.
        self.player = next_player
        if self.terminal:
            rewards = self.rewards(winner)
        else:
            rewards = {1: 0.0, 2: 0.0}
        return self.state(rewards)

    # Returns the reward for both players given self.bets, with 1 referring to
    # the first player and 2 referring to the second player in the game.
    def rewards(self, winner):
        rewards = {}
        # The winner wins their own bet back as well as the loser's bet. So they
        # just win what the loser bet.
        rewards[winner] = self.bets[other_player[winner]]
        # The loser just loses their bet.
        rewards[other_player[winner]] = -self.bets[other_player[winner]]
        return rewards

    # Currently if there is more than one suit, then player 1 is favoured if the
    # cards have the same value.
    def compare_holes(self):
        return 1 if self.hole_card[1]['value'] > self.hole_card[2]['value'] else 2

    # Returns 0 if player 0 wins and 1 if player 1 wins
    def showdown(self):
        self.terminal = True
        return self.compare_holes()

    def state_dim(self):
        return self.num_values * self.num_suits + 2

def index_of_card(card, num_values):
    return card['suit'] * num_values + card['value']

def encoding(hole, is_first_player, num_raises, num_values, num_suits):
    # One hot encoding for cards, plus a single entry for the number of raises.
    vec = np.zeros(num_values * num_suits + 2, 'float32')
    vec[index_of_card(hole, num_values)] = 1.0
    # Also encode the number of raises
    vec[-1] = num_raises
    vec[-2] = int(is_first_player)
    return vec

# Play a game of Leduc
def play_leduc():
    leduc = LeducHoldEm()
    while True:
        first_player = np.random.choice([1,2])
        print "New Game. First player", first_player
        player, state, rewards, terminal = leduc.reset(first_player)
        while not terminal:
            print "Player", player, "to play"
            print "State:", state
            print "Rewards:", rewards
            print "Terminal:", terminal
            print "Please enter action for player", player, " (0 = fold, 1 = call, 2 = raise):"
            action = int(raw_input())
            assert action in [0,1,2]
            player, state, rewards, terminal = leduc.step(action)
        print "Game ended"
        print "Rewards", rewards

if __name__ == "__main__":
    play_leduc()
