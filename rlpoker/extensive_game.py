# coding: utf-8

import collections
from collections.abc import Mapping
import typing

import numpy as np


class ActionFloat(Mapping):
    """
    ActionFloat stores a float for each action.
    """

    @staticmethod
    def initialise_zero(actions: typing.List[typing.Any]):
        """
        Returns an ActionFloat containing zero for all actions.

        Args:
            actions: list of actions.

        Returns:
            ActionFloat.
        """
        return ActionFloat({action: 0.0 for action in actions})

    @staticmethod
    def initialise_uniform(actions: typing.List[typing.Any]):
        """
        Returns an ActionFloat containing 1/len(actions) for each action. This is a uniform probability
        distribution over the actions.

        Args:
            actions: list of actions.

        Returns:
            ActionFloat.
        """
        return ActionFloat({action: 1.0 / len(actions) for action in actions})

    def __init__(self, action_floats: typing.Dict[typing.Any, float]):
        self.action_floats = action_floats

    @staticmethod
    def sum(action_float1: 'ActionFloat', action_float2: 'ActionFloat') -> 'ActionFloat':
        """
        Adds the two action floats together elementwise. If an action exists in one but not the other, then it is
        treated as being zero in the one it doesn't exist in.

        Args:
            action_float1: ActionFloat.
            action_float2: ActionFloat.

        Returns:
            ActionFloat. An ActionFloat with actions given by the union of the actions in action_float1 and
            action_float2, and values being their sum.
        """
        action_float = {action: action_float1[action] for action in action_float1}
        for action in action_float2:
            if action not in action_float:
                action_float[action] = 0.0
            action_float[action] += action_float2[action]

        return ActionFloat(action_float)

    def action_list(self):
        """
        Returns: list of the actions.
        """
        return list(self.action_floats.keys())

    def __getitem__(self, action):
        return self.action_floats[action]

    def __iter__(self):
        for action in self.action_floats:
            yield action

    def __eq__(self, other):
        if not isinstance(other, ActionFloat):
            return False
        return self.action_floats == other.action_floats

    def __str__(self):
        return "ActionFloat({})".format(self.action_floats)

    def __len__(self):
        return len(self.action_floats)

    def copy(self):
        return ActionFloat(self.action_floats.copy())


class Strategy:
    """
    A Strategy is a dictionary mapping information sets to ActionProbabilities in those information sets. The
    ActionProbabilities should only contain the valid actions in the information set.
    """

    @staticmethod
    def initialise():
        """
        Initialise an empty strategy.

        Returns:
            Strategy. Empty strategy.
        """
        return Strategy(dict())

    def __init__(self, strategy: typing.Dict[typing.Any, ActionFloat]):
        self.strategy = strategy

    def set_uniform_action_probs(self, info_set: typing.Any, available_actions: typing.List[typing.Any]):
        assert info_set not in self.strategy, "Info set {} already exists".format(info_set)
        self.strategy[info_set] = ActionFloat.initialise_uniform(available_actions)

    def __getitem__(self, item):
        return self.strategy[item]

    def __setitem__(self, key, value):
        self.strategy[key] = value

    def get_action_probs(self, info_set):
        return self.strategy[info_set]

    def get_info_sets(self):
        return list(self.strategy.keys())

    def copy(self):
        return Strategy(
            {
                k: v.copy() for k, v in self.strategy.items()
            }
        )

    def __str__(self):
        lines = []
        for info_set, action_probs in self.strategy.items():
            lines += ["Info set: {}, Action probs: {}".format(info_set, action_probs)]
        return "Strategy({})".format("\n".join(lines))


class InformationSetAdvantages(typing.NamedTuple):
    info_set: typing.Any
    time: int
    advantages: typing.Dict[typing.Any, float]


class ExtensiveGameNode:
    """
    A game node in an extensive form game.
    """
    def __init__(self, player: int,
                 action_list: typing.Tuple=(),
                 children: typing.Optional[typing.Dict]=None,
                 hidden_from: typing.Optional[typing.Set]=None,
                 chance_probs: typing.Optional[ActionFloat]=None,
                 utility: typing.Optional[typing.Dict]=None):
        """
        Args:
            player: int. The player to play in the node. Use -1 for terminal, 0 for chance, 1 for player 1,
                2 for player 2.
            action_list: tuple. The sequence of actions leading to this node.
            children: Dict. Maps available actions in the node to ExtensiveGameNode objects resulting from taking
                the action in this node.
            hidden_from: Set or None. The set of players from which the actions in this node are hidden.
            chance_probs: ActionFloat or None. Store the chance probabilities for chance nodes.
            utility: Dict or None. The utility of the node to each player. Only specify this for terminal nodes.
        """
        self.player = player

        self.children = dict() if children is None else children

        self.hidden_from = set() if hidden_from is None else hidden_from

        self.utility = dict() if utility is None else utility

        self.chance_probs = dict() if chance_probs is None else chance_probs

        self.action_list = action_list

        # The node can also store extra information.
        self.extra_info = dict()

    def __str__(self):
        return "\n".join(["Player: {}".format(self.player),
                         "Actions: {}".format(list(self.actions)),
                         "Hidden from: {}".format(self.hidden_from),
                         "Utility: {}".format(self.utility),
                         "Chance probs: {}".format(self.chance_probs),
                         "Action list: {}".format(self.action_list)])

    @property
    def actions(self):
        if self.children is None:
            return []
        else:
            return list(self.children.keys())


class ExtensiveGame:
    """Create an ExtensiveGame by passing an ExtensiveGameNode to the
    constructor. This should be the root of the game, and define the entire
    game tree.
    """

    def __init__(self, root):
        # set the root node.
        self.root = root

        # Also build the information set ids.
        self.info_set_ids = self.build_info_set_ids()

    @staticmethod
    def print_tree_recursive(node, action_list, only_leaves):
        """ Prints out a list of all nodes in the tree rooted at 'node'.
        """
        if only_leaves and len(node.children) == 0:
            print(action_list, node.utility)
        elif not only_leaves:
            print("Node for action list: {}".format(action_list))
            print(node)
            print("-----")
        for action, child in node.children.items():
            ExtensiveGame.print_tree_recursive(
                child, action_list + (action,), only_leaves)

    def print_tree(self, only_leaves=False):
        """ Prints out a list of all nodes in the tree by the list of actions
        needed to get to each node from the root.
        """
        ExtensiveGame.print_tree_recursive(self.root, (), only_leaves)

    def build_information_sets(self, player):
        """ Returns a dictionary from nodes to a unique identifier for the
        information set containing the node. This is all for the given player.
        """
        info_set = {}

        # We just recursively walk over the tree using a stack to store the
        # nodes to explore.
        node_stack = [self.root]
        visible_actions_stack = [[]]

        # First build the information sets for player 1.
        while len(node_stack) > 0:
            node = node_stack.pop()
            visible_actions = visible_actions_stack.pop()

            # Add the information set for the node, indexed by the
            # visible_actions list, to the information set dictionary. Use a
            # tuple instead of a list so that it is hashable if we want later
            # on.
            info_set[node] = tuple(visible_actions)

            for action, child in node.children.items():
                # Add all the children to the node stack and also the visible
                # actions to the action stack. If an action is hidden from the
                # player, then add -1 to signify this.
                node_stack.append(child)
                if node.hidden_from is not None and player in node.hidden_from:
                    visible_actions_stack.append(visible_actions + [-1])
                else:
                    visible_actions_stack.append(visible_actions + [action])

        return info_set

    def build_info_set_ids(self):
        """ Join the two info set dictionaries. The keys are nodes in the game
        tree belonging to player 1 or player 2, and the values are the
        identifier for the information set the node belongs to, from the
        perspective of the player to play in the node.
        """
        info_sets_1 = self.build_information_sets(1)
        info_sets_2 = self.build_information_sets(2)
        info_set_ids = {}
        for k, v in info_sets_1.items():
            if k.player == 1:
                info_set_ids[k] = v
        for k, v in info_sets_2.items():
            if k.player == 2:
                info_set_ids[k] = v
        return info_set_ids

    def expected_value(self, strategy_1: Strategy, strategy_2: Strategy, num_iters: int):
        """Given a strategy for player 1 and a strategy for player 2, compute
        the expected value for player 1.
        - strategy_1: Strategy.
        - strategy_2: Strategy.
        Returns the result of each game of strategy_1 versus strategy_2.
        """
        results = []
        for t in range(num_iters):
            node = self.root
            while node.player != -1:
                # Default to playing randomly.
                actions = [a for a in node.children.keys()]
                probs = [1.0 / float(len(actions)) for a in actions]

                # If it's a chance node, then sample an outcome.
                if node.player == 0:
                    probs = [node.chance_probs[action] for action in actions]
                elif node.player == 1:
                    # It's player 1's node, so use their strategy to make a
                    # decision.
                    info_set = self.info_set_ids[node]
                    if info_set in strategy_1:
                        probs = [strategy_1[info_set][action] for action in actions]
                elif node.player == 2:
                    # It's player 2's node, so use their strategy to make a
                    # decision.
                    info_set = self.info_set_ids[node]
                    if info_set in strategy_2:
                        probs = [strategy_2[info_set][action] for action in actions]

                # Make sure the probabilities sum to 1
                assert np.isclose(sum(probs), 1.0)

                # Sample an action from the probability distribution.
                action = np.random.choice(np.array(actions), p=np.array(probs))

                # Move into the child node.
                node = node.children[action]

            # The node is terminal. Add the utility for player 1 to the results.
            results.append(node.utility[1])

        return results

    def expected_value_exact(self, strategy1: Strategy, strategy2: Strategy) -> typing.Tuple[float, float]:
        """Given a strategy for player 1 and a strategy for player 2, compute the exact expected value for player 1.

        Args:
            strategy1: Strategy.
            strategy2: Strategy.

        Returns:
            utility1, utility2. The expected utility for player 1 and for player 2.
        """
        to_explore = collections.deque([(self.root, 1.0)])

        strategies = {
            1: strategy1,
            2: strategy2
        }

        # Traverse the tree, keeping track of the probabilities of reaching each node.

        expected_value = {
            1: 0.0,
            2: 0.0
        }
        while len(to_explore) > 0:
            node, reach_prob = to_explore.popleft()

            if node.player == -1:
                expected_value[1] += reach_prob * node.utility[1]
                expected_value[2] += reach_prob * node.utility[2]
            else:
                for action, child in node.children.items():
                    if node.player == 0:
                        # Chance node
                        action_prob = node.chance_probs[action]
                    else:
                        # Player node
                        info_set = self.info_set_ids[node]
                        action_prob = strategies[node.player][info_set][action]

                    to_explore.append((child, reach_prob * action_prob))

        return expected_value[1], expected_value[2]

    def complete_strategy_uniformly(self, strategy: Strategy):
        """ Given a partial strategy, i.e. a dictionary from a subset of the
        info_set_ids to probabilities over actions in those information sets,
        complete the dictionary by assigning uniform probability distributions
        to the missing information sets.
        """
        new_strategy = strategy.copy()
        num_missing = 0
        for node, info_set_id in self.info_set_ids.items():
            if info_set_id not in new_strategy:
                actions = node.children.keys()
                new_strategy[info_set_id] = {
                    a: 1.0 / float(len(actions)) for a in actions}
                num_missing += 1
        if num_missing > 0:
            print("Completed strategy at {} information sets.".format(num_missing))
        return new_strategy

    def is_strategy_complete(self, strategy: Strategy):
        """Returns whether or not the strategy contains probabilities for
        each information set.
        """
        return set(strategy.keys()) == set(self.info_set_ids.values())

    def get_node(self, actions: typing.Tuple) -> ExtensiveGameNode:
        """
        Returns the node in the tree corresponding to the given action list, or None, if no such node exists.

        Args:
            actions: list of actions.

        Returns:
            The node with that action list.
        """
        node = self.root
        for action in actions:
            if action in node.children:
                node = node.children[action]
            else:
                return None

        return node

    def get_info_set_id(self, node: ExtensiveGameNode):
        """
        Returns the information set id for the given node. Must be player 1 or 2 node.

        Args:
            node: ExtensiveGameNode.

        Returns:
            information set id.
        """
        assert node.player > 0
        return self.info_set_ids[node]
