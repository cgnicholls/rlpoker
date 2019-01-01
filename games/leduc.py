# coding: utf-8
# This implements Leduc Hold'em.

from collections import deque, Counter

from rlpoker.extensive_game import ExtensiveGame, ExtensiveGameNode


class Leduc(ExtensiveGame):
    """
    """
    def __init__(self, cards, max_raises=4, raise_amount=2):
        root = Leduc.create_tree(cards, max_raises=max_raises,
                                 raise_amount=raise_amount)

        # Initialise the super class.
        super(Leduc, self).__init__(root)

    @staticmethod
    def create_tree(cards, max_raises=4, raise_amount=2):
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
                    child_node = ExtensiveGameNode(0,
                        action_list=action_list + (c,),
                        hidden_from={1})
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
                    child_node = ExtensiveGameNode(1,
                        action_list=action_list + (c,))
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
                    action_list) if a >= 10])
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
                    child_node = ExtensiveGameNode(next_player,
                        action_list + (1,))
                    current_node.children[1] = child_node
                    next_pot = pot.copy()
                    to_explore.append((child_node, next_state,
                                       remaining_cards, next_pot))

                    # Bet
                    next_player = other_player[current_player]
                    child_node = ExtensiveGameNode(next_player,
                        action_list + (2,))
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
                    child_node = ExtensiveGameNode(next_player,
                        action_list + (1,))
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
                cards = tuple(a for a in action_list if a >= 10)
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
