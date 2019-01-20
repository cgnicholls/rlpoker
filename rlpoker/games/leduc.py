# coding: utf-8
# This implements Leduc Hold'em.

import numpy as np

from collections import deque, Counter

from rlpoker.extensive_game import ExtensiveGame, ExtensiveGameNode
from rlpoker.games.card import Card


class Leduc(ExtensiveGame):
    """Leduc is a game of poker with 2 rounds. Both players initially ante 1
    chip. The raise amount in round 1 is 2 and the raise amount in round 2
    is 4. When facing a bet/raise, a player can:
    - call, matching the amount their opponent bet;
    - raise, betting the raise amount;
    - or fold, ending the hand.
    When facing a check, the player can either
    - check, ending the round;
    - or raise, betting the raise amount.
    On the first move in a round, the player can check or bet.
    """
    def __init__(self, cards, max_raises=4, raise_amount=2):
        root = Leduc.create_tree(cards, max_raises=max_raises,
                                 raise_amount=raise_amount)

        # Check all the cards are distinct
        assert len(set(cards)) == len(cards)
        self.cards = cards

        self.max_raises = max_raises
        self.raise_amount = raise_amount

        # Initialise the super class.
        super().__init__(root)

        # We first define a one-hot-encoding based on the cards.
        self.card_indices = dict(enumerate(cards))
        self.card_indices = {v: k for k, v in self.card_indices.items()}

        self.state_vectors = compute_state_vectors(self.info_set_ids.values(),
                                                   self.card_indices, self.max_raises)

        # Make sure the mappings are unique.
        assert len(set(self.state_vectors.keys())) == len({tuple(v) for v in self.state_vectors.values()})

        # Make sure all the vectors are the same length.
        assert len(set(len(tuple(v)) for v in self.state_vectors.values())) == 1
        self.state_dim = list(len(tuple(v)) for v in self.state_vectors.values())[0]
        self.action_dim = 3

    @staticmethod
    def create_tree(cards, max_raises, raise_amount):
        # Create the root node.
        root = ExtensiveGameNode(0, action_list=(), hidden_from={2})

        raise_amount = {'round_1': raise_amount,
                        'round_2': 2 * raise_amount}

        # Both players ante 1 initially.
        pot = {1: 1, 2: 1}

        to_explore = deque()
        to_explore.append((root, "deal_card_1", tuple(cards), pot))

        other_player = {1: 2, 2: 1}

        # We explore all the nodes and add their children until we have
        # created the whole game tree.
        while len(to_explore) > 0:
            current_node, state, remaining_cards, pot = to_explore.popleft()
            current_node.extra_info['pot'] = pot
            action_list = current_node.action_list
            current_player = current_node.player

            if state == "deal_card_1":
                for c, n in Counter(remaining_cards).items():
                    current_node.hidden_from = [2]
                    child_node = ExtensiveGameNode(0, action_list=action_list + (c,), hidden_from={1})
                    current_node.children[c] = child_node
                    current_node.chance_probs[c] = n / len(remaining_cards)
                    index = remaining_cards.index(c)
                    to_explore.append(
                        (child_node, "deal_card_2",
                         remaining_cards[:index] + remaining_cards[index+1:],
                         pot))
            elif state == "deal_card_2":
                for c, n in Counter(remaining_cards).items():
                    current_node.hidden_from = [1]
                    child_node = ExtensiveGameNode(1, action_list=action_list + (c,))
                    current_node.children[c] = child_node
                    current_node.chance_probs[c] = n / len(remaining_cards)
                    index = remaining_cards.index(c)
                    to_explore.append(
                        (child_node, "round_1",
                         remaining_cards[:index] + remaining_cards[index+1:],
                         pot))
            elif state == "deal_board":
                # We deal a board card from the remaining cards.
                for c, n in Counter(remaining_cards).items():
                    # Player 2 starts the second round.
                    child_node = ExtensiveGameNode(2, action_list + (c,))
                    current_node.children[c] = child_node
                    current_node.chance_probs[c] = n / len(remaining_cards)

                    index = remaining_cards.index(c)
                    to_explore.append((child_node, "round_2",
                                       remaining_cards[:index] +
                                       remaining_cards[index+1:],
                                       pot.copy()))
            elif state == "round_1" or state == "round_2":
                # The betting actions are all the actions since the last
                # card was dealt on the board.
                last_card_index = max([i for i, a in enumerate(
                    action_list) if type(a) is Card])
                betting_actions = action_list[last_card_index+1:]
                # First determine the available actions. If we are still in
                # the betting round then the previous player didn't fold.
                # There are three options:
                # - no action yet.
                # - last action was call
                # - last action was raise.
                if len(betting_actions) == 0:
                    # We may either call or bet.
                    next_player = other_player[current_player]
                    # Call
                    child_node = ExtensiveGameNode(next_player,
                        action_list + (1,))
                    current_node.children[1] = child_node
                    to_explore.append((child_node, state, remaining_cards,
                                       pot.copy()))

                    # Bet
                    child_node = ExtensiveGameNode(next_player,
                        action_list + (2,))
                    current_node.children[2] = child_node
                    next_pot = pot.copy()
                    next_pot[current_player] += raise_amount[state]
                    to_explore.append((child_node, state, remaining_cards,
                                       next_pot))
                elif betting_actions[-1] == 1:
                    # The previous player called, and this didn't end the
                    # betting round. We can either call (and end the betting
                    # round), or we can bet (and continue the betting round).

                    # Call
                    if state == "round_1":
                        next_state = "deal_board"
                        next_player = 0
                    else:
                        next_state = "showdown"
                        next_player = -1
                    child_node = ExtensiveGameNode(next_player, action_list + (1,))
                    current_node.children[1] = child_node
                    next_pot = pot.copy()
                    to_explore.append((child_node, next_state,
                                       remaining_cards, next_pot))

                    # Bet
                    next_player = other_player[current_player]
                    child_node = ExtensiveGameNode(next_player, action_list + (2,))
                    current_node.children[2] = child_node
                    next_pot = pot.copy()
                    next_pot[current_player] += raise_amount[state]
                    to_explore.append((child_node, state,
                                       remaining_cards, next_pot))
                elif betting_actions[-1] == 2:
                    # We are facing a bet. We can bet if there are fewer
                    # than max_raises so far; this continues the betting
                    # round. If we bet and there are now max_raises bets
                    # then this ends the round and we go to the next state.
                    # We can call to end the betting round. We can fold to
                    # end the betting round.

                    # Call
                    if state == "round_1":
                        next_state = "deal_board"
                        next_player = 0
                    else:
                        next_state = "showdown"
                        next_player = -1
                    child_node = ExtensiveGameNode(next_player, action_list + (1,))
                    current_node.children[1] = child_node
                    next_pot = pot.copy()
                    next_pot[current_player] += raise_amount[state]
                    to_explore.append((child_node, next_state,
                                       remaining_cards, next_pot))

                    # Fold
                    child_node = ExtensiveGameNode(-1, action_list + (0,))
                    current_node.children[0] = child_node
                    to_explore.append((child_node, "fold{}".format(
                        current_player), remaining_cards, pot.copy()))

                    # Bet
                    num_bets = Counter(betting_actions)[2]
                    if num_bets < max_raises - 1:
                        # We can bet and continue the round.
                        child_node = ExtensiveGameNode(
                            other_player[current_player],
                            action_list + (2,))
                        current_node.children[2] = child_node
                        next_pot = pot.copy()
                        next_pot[current_player] += raise_amount[state]
                        to_explore.append((child_node, state,
                                           remaining_cards, next_pot))
                    else:
                        # This bet will end the betting round.
                        if state == "round_1":
                            next_state = "deal_board"
                            next_player = 0
                        else:
                            next_state = "showdown"
                            next_player = -1

                        child_node = ExtensiveGameNode(next_player,
                                                       action_list + (2,))
                        current_node.children[2] = child_node
                        next_pot = pot.copy()
                        next_pot[current_player] += raise_amount[state]
                        to_explore.append(
                            (child_node, next_state, remaining_cards,
                             next_pot))
            elif state == 'fold1':
                # Player 1 folded. So player 2 wins the whole pot and player
                # 1 loses what they bet.
                current_node.utility = {1: -pot[1], 2: pot[1]}
            elif state == 'fold2':
                # Player 2 folded, so player 1 wins the whole pot and player
                #  2 loses what they bet.
                current_node.utility = {1: pot[2], 2: -pot[2]}
            elif state == 'showdown':
                # We have to work out who won the pot.
                cards = tuple(a for a in action_list if type(a) is Card)
                assert len(cards) == 3
                hole1, hole2, board = cards
                current_node.utility = Leduc.compute_showdown(
                    hole1, hole2, board, pot)
            else:
                # Make sure we don't get here!
                assert False

        return root

    @staticmethod
    def compute_showdown(hole1, hole2, board, pot):
        """Compute showdown returns the reward dictionary. The reward
        dictionary has keys 1 and 2 and values the reward that player 1 and
        player 2 respectively receive from the hand.

        Args:
            hole1: Card.
            hole2: Card.
            board: Card.
            pot: dict. Keys are 1 and 2 and values are the amounts that each of
            1 and 2 have bet into the pot.

        Returns:
            dict. Dictionary with the reward for player 1 and reward for
            player 2.
        """
        # First convert each card into just its value.
        hole1 = hole1.value
        hole2 = hole2.value
        board = board.value

        # We evaluate as: pair beats high card.
        pair1 = hole1 == board
        pair2 = hole2 == board

        if pair1 and pair2:
            # Draw
            return {1: 0, 2: 0}
        elif pair1 and not pair2:
            # Player 1 wins:
            return {1: pot[2], 2: -pot[2]}
        elif not pair1 and pair2:
            # Player 2 wins:
            return {1: -pot[1], 2: pot[1]}

        # No pairs, so the winner is just whoever has the higher card
        if hole1 > hole2:
            return {1: pot[2], 2: -pot[2]}
        elif hole2 > hole1:
            return {1: -pot[1], 2: pot[1]}
        else:
            return {1: 0, 2: 0}


def compute_state_vectors(info_set_ids, card_indices, max_raises):
    """Computes a state vector for each information set id. This is a unique vector
    for each information set, which encodes the information set.

    Args:
        info_set_ids: iterable. Should be the ids for each information set.

    Returns:
        dict. Dictionary with keys the information set ids and values
        the vector representation as a numpy array.
    """
    state_vectors = {}
    # Generate a vector for each information set.
    for info_set_id in info_set_ids:
        # The information set is of the form (hole1, hole2, <betting
        # actions>, board, <betting actions>.

        cards_in_play = [a for a in info_set_id if type(a) is Card]
        assert len(cards_in_play) <= 2

        # The hole card is the first card we see.
        hole = cards_in_play[0]

        # The player is determined by which hole card is hidden.
        player = 2 if info_set_id[0] == -1 else 1
        player_vec = np.array([player - 1])

        # We embed the hole card for the player to play, then the board
        # card, then the round 1 betting actions, then the round 2
        # betting actions.
        hole_vec = one_hot_encoding(len(card_indices), card_indices[hole])

        # If there is more than one card in play, then the board has been played.
        if len(cards_in_play) > 1:
            board = cards_in_play[1]
            board_vec = one_hot_encoding(len(card_indices), card_indices[board])
        else:
            board_vec = np.zeros(len(card_indices), dtype=float)

        actions1, actions2, current_round = compute_betting_rounds(info_set_id)

        if current_round == 1:
            num_raises1 = Counter(actions1)[2]
            round1_vec = one_hot_encoding(max_raises + 1, num_raises1)
            round1_vec = np.append(round1_vec, len(actions1) == 0)
            first_is_call = 0
            if len(actions1) > 0 and actions1[0] == 1:
                first_is_call = 1
            round1_vec = np.append(round1_vec, first_is_call)
            round2_vec = np.zeros(max_raises + 3, dtype=float)
        else:
            num_raises1 = Counter(actions1)[2]
            num_raises2 = Counter(actions2)[2]

            round1_vec = one_hot_encoding(max_raises + 1, num_raises1)
            round1_vec = np.append(round1_vec, len(actions1) == 0)
            first_is_call = 0
            if len(actions1) > 0 and actions1[0] == 1:
                first_is_call = 1
            round1_vec = np.append(round1_vec, first_is_call)
            round2_vec = one_hot_encoding(max_raises + 1, num_raises2)
            round2_vec = np.append(round2_vec, len(actions2) == 0)
            first_is_call = 0
            if len(actions2) > 0 and actions2[0] == 1:
                first_is_call = 1
            round2_vec = np.append(round2_vec, first_is_call)

        state_vectors[info_set_id] = np.concatenate(
            [player_vec, hole_vec, board_vec, round1_vec, round2_vec], axis=0)

    return state_vectors


def compute_betting_rounds(info_set_id):
    # We now encode the rounds.
    card_locations = [i for i, a in enumerate(info_set_id) if type(a) is Card]

    if len(card_locations) == 1:
        actions1 = info_set_id[2:]
        actions2 = ()
        current_round = 1
    elif len(card_locations) == 2:
        actions1 = info_set_id[2:card_locations[1]]
        actions2 = info_set_id[card_locations[1] + 1:]
        current_round = 2
    else:
        assert False

    return actions1, actions2, current_round


def one_hot_encoding(dim, i):
    """Returns the one hot vector length 'dim' with a 1 in the ith position (counting from zero).

    Args:
        dim: int. The length of the vector.
        i: int. The index to put a 1 in.

    Returns:
        ndarray.
    """
    x = np.zeros(dim, dtype=float)
    x[i] = 1.0
    return x
