# coding: utf-8
# Computes a best response for a given strategy in a given game.
import itertools
import typing

from rlpoker import extensive_game


def br(game: extensive_game.ExtensiveGame, info_set: typing.List[extensive_game.ExtensiveGameNode],
       reach_probs: typing.Dict[extensive_game.ExtensiveGameNode, float],
       strategy: extensive_game.Strategy, br_strategy, player):
    """

    Args:
        game: ExtensiveGame.
        info_set: list of nodes, all in the same information set for the player.
        reach_probs: dictionary mapping a node to the probability of reaching this node if we ignore player's
        strategy. This is often denoted pi_{-i}^strategy(node).
        strategy: Strategy. A dictionary mapping a node in the game tree (for the other player) to probabilities
            over actions (including chance).
        br_strategy: dictionary mapping information sets to probabilities over actions. The best response strategy
            we return.
        player: int. the player for whom we want to compute the best response (against the other player's strategy).

    Returns:
        best response strategy.
    """
    # Check if the information set is terminal (i.e. has a terminal node in it).
    if info_set[0].player == -1:
        # It's a terminal node, so we return the utility for player i, weighted
        # by the reach probabilities.
        utility = 0.0
        for node in info_set:
            utility += reach_probs[node] * node.utility[player]
        return utility
    # The information set is not terminal
    if info_set[0].player != player:
        # The information set belongs to an opponent of i (including chance)
        info_sets = {}
        new_reach_probs = {}
        for node in info_set:
            for action, child in node.children.items():
                # Add child to the information set corresponding to taking
                # action 'action'.
                if action not in info_sets:
                    info_sets[action] = []
                info_sets[action].append(child)
                # Update the reach probability by multiplying by the chance the
                # opponent takes this action.

                # It's a chance player:
                if node.player == 0:
                    new_prob = node.chance_probs[action]
                else:
                    new_prob = strategy[game.info_set_ids[node]][action]
                new_reach_probs[child] = reach_probs[node] * new_prob

        # Convert the information sets into a list, partitioned by actions.
        info_sets = [info_sets[a] for a in info_sets]

        # If the actions in the node are not hidden from i, then we need a new
        # information set for each action. If they are hidden from i, then all
        # the child nodes are in the same information set.
        if player in info_set[0].hidden_from:
            # Concatenate the lists in info_sets into one list. Put it inside a
            # list again, so that we have a list of information sets (i.e. a
            # list of lists).
            info_sets = [list(itertools.chain(*info_sets))]

        # For each resulting information set, compute the best response, and
        # return the sum
        br_sum = 0.0
        for I in info_sets:
            br_sum += br(game, I, new_reach_probs, strategy, br_strategy, player)
        return br_sum
    else:
        # The info set belongs to i. Player i chooses the action with maximum
        # value returned by br.
        brs = []
        actions = []
        for action in info_set[0].children:
            next_info_set = [node.children[action] for node in info_set]
            # The reach probabilities don't change, since it's i's information
            # set.
            next_reach_probs = {
                node.children[action]: reach_probs[node] for node in info_set}
            brs.append(br(game, next_info_set, next_reach_probs, strategy,
                          br_strategy, player))
            actions.append(action)

        # Player i chooses action with maximum value.
        info_set_id = game.info_set_ids[info_set[0]]
        br_strategy[info_set_id] = {a: 0.0 for a in info_set[0].children}

        # Get the maximum br and its corresponding action.
        best_action, best_br = max(zip(actions, brs), key=(lambda x: x[1]))

        # Set the best action to have probability 1
        br_strategy[info_set_id][best_action] = 1.0
        return best_br


def compute_best_response(game, strategy: extensive_game.Strategy, player: int):
    """
    Given a game (defined by an ExtensiveGame) and a strategy (defined by a dictionary from nodes in the game tree
    to probabilities over actions), returns the best response for player i against the other player.
    -

    Args:
        game: ExtensiveGame.
        strategy: Strategy.
        player: int. The player to compute the best response for.

    Returns:
        value, strategy.
    """
    br_strategy = {}
    br_value = br(game, [game.root], {game.root: 1.0}, strategy, br_strategy, player)
    return br_value, br_strategy


def compute_exploitability(game: extensive_game.ExtensiveGame, strategy: extensive_game.Strategy):
    """Computes the exploitability of a given strategy. The strategy must
    implement both player 1 and player 2 information sets.

    Args:
        game: ExtensiveGame.
        strategy: Strategy.

    Returns:
        exploitability: float.
    """
    # First compute the best response against the strategy when the strategy
    # plays as player 1. Then compute the best response against the strategy
    # when it plays as player 2.
    exploitability_1, br_against_1 = compute_best_response(game, strategy, 1)
    exploitability_2, br_against_2 = compute_best_response(game, strategy, 2)
    return exploitability_1 + exploitability_2
