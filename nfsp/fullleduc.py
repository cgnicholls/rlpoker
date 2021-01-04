import numpy as np
from random import shuffle

# LeducHoldEm is the following game.
# The deck consists of (J, J, Q, Q, K, K).
# Each player gets 1 card. There are two betting rounds and the number of raises in each round is at most 2. In the second round, one card is revealed on the table and is used to create a hand.
# There are two types of hands: pairs and highest card. There are three actions: fold, call and raise. Each of the two players antes 1. In the first round, the betting amount is 2 (including the ante for the first bet). In the second round it is 4.

# Betting in Poker:
# Bet: place a wager into the pot, or raise, by matching the outstanding bet and placing an extra bet into the pot.
# Call: matching the amount of all outstanding bets (or check, by passing when there is no bet)
# Fold: pass when there is an outstanding bet and forfeiting the current game.

# So keep track of whether the previous player called or not. Note that the big blind is an initial bet.

# Actions are: 0 - fold
# 1 - call
# 2 - raise

# Rounds: 0 - pre-flop
# 1 - flop

# Small blind is 1, big blind is 2.
other_player = {1: 2, 2: 1}

class LeducHoldEm:
    def __init__(self, num_values=4, num_suits=1, max_raises=2):
        self.num_values = num_values
        self.num_suits = num_suits
        # Initialise the deck
        self.deck = [{'value': v, 'suit': s} for v in range(self.num_values) for s in range(self.num_suits)]
        self.hole_card = None
        self.bets = None
        self.player = None
        self.terminal = None
        self.num_raises = {0: 0, 1: 0}
        self.board = None
        self.max_raises = max_raises
        self.first_player = None

    def name(self):
        return 'leducholdem'

    def reset(self, first_player):
        # The first player posts the small blind (in heads up) and the second
        # player posts the big blind.
        self.first_player = first_player
        self.player = first_player
        self.bets = {self.player: 1, other_player[self.player]: 2}

        # Since the second player posted the big blind, this is the previous
        # betting player.
        self.previous_player_called = False
        self.num_raises = {0: 0, 1: 0}

        # Initialise and shuffle the deck
        self.deck = [{'value': v, 'suit': s} for v in range(self.num_values) \
            for s in range(self.num_suits)]
        shuffle(self.deck)

        # Draw the hole cards for 1 and 2 (doesn't matter in which order).
        self.hole_card = {1: self.deck.pop(), 2: self.deck.pop()}
        self.round = 0
        self.board = None

        self.terminal = False
        rewards = {1: 0.0, 2: 0.0}
        return self.state(rewards)

    # Returns the state, given the rewards. Uses self.player, self.round,
    # self.hole_card, self.board, self.bets and self.terminal.
    def state(self, rewards):
        state = encoding(self.hole_card[self.player], self.board, self.player==self.first_player, self.round, self.num_raises, self.max_raises, self.num_values, self.num_suits)

        return self.player, state, rewards, self.terminal

    def step(self, a):
        # Make sure we aren't playing when the game is over.
        if self.terminal:
            assert False
        next_player = other_player[self.player]

        # If the player's action is to raise, we actually change it to a call if
        # the maximum number of raises has already been reached.
        if a == 2 and self.num_raises[self.round] >= self.max_raises:
            a = 1
        # If the action is to fold, then this hand is over, and we return the
        # rewards for each player as a dictionary: reward for player 1 and
        # reward for player 2.
        if a == 0:
            self.terminal = True
            winner = next_player
        # The player called.
        elif a == 1:
            # In all cases, we update the player's bet to equal the other player's bet.
            self.bets[self.player] = self.bets[next_player]

            # If the previous player called then it's the end of the round. The round is also over if there have been any raises in this round.
            if self.previous_player_called or self.num_raises[self.round] > 0:
                # If the round is 0 then we move to round 1.
                if self.round == 0:
                    # In the next round, no one has called yet.
                    self.previous_player_called = False

                    # Increment the round number
                    self.round = 1

                    # Switch players
                    self.player = self.first_player

                    # We also have to draw the board card
                    self.board = self.deck.pop()
                else:
                    # Else the round is 1 and we have a showdown.
                    self.terminal = True
                    winner = self.showdown()
            else:
                # Else the previous player didn't call and there haven't been any raises this round so this is a check. We set previous_player_called to True. It's now the next player's turn.
                self.previous_player_called = True

                # Switch players.
                self.player = next_player

        elif a == 2:
            # The player raised. We already checked that they are allowed to
            # (i.e. we haven't already reached the max raises) and so we increment the number of raises for this round and add to the player's bet size.
            self.num_raises[self.round] += 1

            bet_size = 1 if self.round == 0 else 2

            # In all cases, raising is equivalent to first calling and then
            # raising.
            self.bets[self.player] = self.bets[next_player]
            self.bets[self.player] += bet_size

            # Switch players
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
        # If it was a draw, return 0 for both players
        if winner == 0:
            return {1: 0.0, 2: 0.0}

        # If it's not a draw, the winner wins their own bet back as well as the loser's bet. So they just win what the loser bet.
        rewards[winner] = self.bets[other_player[winner]]

        # The loser just loses their bet.
        rewards[other_player[winner]] = -self.bets[other_player[winner]]
        return rewards

    # Returns 0 if player 0 wins and 1 if player 1 wins
    def showdown(self):
        return compare_hands(self.hole_card[1], self.hole_card[2], self.board)

    def state_dim(self):
        return 1 + 2 * (self.max_raises + 1) + 2 * self.num_values

def index_of_card(card, num_values):
    return card['suit'] * num_values + card['value']

# Ignore suits for the one hot card. Also, if the card is None then just return zeros.
def one_hot_card(card, num_values):
    vec = np.zeros(num_values)
    if card is None:
        return vec
    else:
        vec[card['value']] = 1.0
    return vec

def raise_encoding(num_raises0, num_raises1, max_raises, the_round):
    round0 = np.zeros(max_raises+1)
    round0[num_raises0] = 1.0
    round1 = np.zeros(max_raises+1)
    if the_round == 1:
        round1[num_raises1] = 1.0
    return np.concatenate((round0, round1))

def encoding(hole, board, is_first_player, the_round, num_raises, max_raises, num_values, num_suits):
    # One hot encoding for cards, plus a single entry for the number of raises.
    hole_enc = one_hot_card(hole, num_values)
    board_enc = one_hot_card(board, num_values)
    raise_enc = raise_encoding(num_raises[0], num_raises[1], max_raises, the_round)
    is_first = np.array([is_first_player]).astype('float32')
    return np.concatenate((hole_enc, board_enc, raise_enc, is_first))

def compare_hands(hole1, hole2, board):
    # If either player pairs with the board, they win
    if hole1['value'] == board['value']:
        return 1
    elif hole2['value'] == board['value']:
        return 2
    else:
        # Else, no one paired with the board, so just return who has the highest value
        if hole1['value'] > hole2['value']:
            return 1
        elif hole2['value'] > hole1['value']:
            return 2
        else:
            # If both players have the same card value and neither is a pair, it's a draw.
            return 0

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
