import random
import enum
import abc

from rlpoker.extensive_game import ExtensiveGame

# TODO: Currently I implement losing influence by dropping the last card in the player's hand, rather than the
# player having a choice.
# TODO: Currently the Ambassador action randomly chooses which cards to choose out of the ones it sees from
# the deck.
# TODO: Currently we don't enforce that you must coup if you have 10+ coins.
# TODO: The game tree.
# TODO: Challenges and counteractions.


class Card(enum.Enum):
    AMBASSADOR = 1
    ASSASSIN = 2
    CAPTAIN = 3
    CONTESSA = 4
    DUKE = 5

    @staticmethod
    def unique_cards():
        return [Card.AMBASSADOR, Card.ASSASSIN, Card.CAPTAIN, Card.CONTESSA, Card.DUKE]


class State:
    """A State in Coup is sufficient information to apply an action. An action then returns the next state
    that results from that action.

    The game tree deals with the challenges and counteractions.
    """
    
    def __init__(self, player, coins, cards, lost_cards, deck):
        self.player = player
        self.coins = coins
        self.cards = cards
        self.lost_cards = lost_cards
        self.deck = deck

    def next_player(self):
        return 1 if self.player == 2 else 2

    def copy(self):
        return State(
            player=self.player,
            coins=self.coins.copy(),
            cards=self.cards.copy(),
            lost_cards=self.lost_cards.copy(),
            deck=self.deck.copy()
        )


class Action:
    
    @abc.abstractmethod
    def apply(state):
        pass


class IncomeAction(Action):
    """The income action increases the player's coins by 1.
    """
    
    @staticmethod
    def apply(state):
        next_state = state.copy()
        next_state.coins[state.player] += 1
        next_state.player = state.next_player()

        return next_state


class ForeignAidAction(Action):
    """The foreign aid action increases the player's coins by 2. It can be blocked by the Duke.
    """
    
    @staticmethod
    def apply(state):
        next_state = state.copy()
        next_state.coins[state.player] += 2
        next_state.player = state.next_player()

        return next_state


class TaxAction(Action):
    """The income action increases the player's coins by 3.
    """
    @staticmethod
    def apply(state):
        next_state = state.copy()
        next_state.coins[state.player] += 3
        next_state.player = state.next_player()

        return next_state


class StealAction(Action):
    """The steal action takes up to 2 coins from the opponent and adds to the current player's coins.
    """
    @staticmethod
    def apply(state):
        next_state = state.copy()
        other_player = state.next_player()
        stolen_coins = min(state.coins[other_player], 2)
        next_state.coins[state.player] += stolen_coins
        next_state.coins[other_player] -= stolen_coins
        next_state.player = state.next_player()

        return next_state


class AssassinateAction(Action):
    """The assassinate action costs 3 coins and causes the other player to lose influence."""

    @staticmethod
    def apply(state):
        current_player = state.player
        # Make sure the player has enough coins.
        if state.coins[current_player] < 3:
            print("State: {}".format(state))
            raise ValueError("Player {} does not have enough coins to assassinate.".format(current_player))

        next_state = loseInfluence(state, state.next_player())
        next_state.coins[current_player] -= 3
        next_state.player = next_state.next_player()

        return next_state


class CoupAction(Action):
    """The Coup action costs 7 and causes the other player to lose influence."""

    @staticmethod
    def apply(state):
        current_player = state.player
        # Make sure the player has enough coins.
        if state.coins[current_player] < 7:
            print("State: {}".format(state))
            raise ValueError("Player {} does not have enough coins to coup.".format(current_player))

        next_state = loseInfluence(state, state.next_player())
        next_state.coins[current_player] -= 7
        next_state.player = next_state.next_player()

        return next_state


def loseInfluence(state, player):
    state = state.copy()
    lost_card = state.cards[player].pop()
    state.lost_cards[player].append(lost_card)
    
    return state


class Coup(ExtensiveGame):

    def __init__(self):
        pass

    @staticmethod
    def dealInitialCards():
        stack1 = Card.unique_cards()
        stack2 = Card.unique_cards()
        stack3 = Card.unique_cards()
        random.shuffle(stack1)
        random.shuffle(stack2)
        random.shuffle(stack3)

        return Coup.dealInitialCardsGivenStacks(stack1, stack2, stack3)

    @staticmethod
    def dealInitialCardsGivenStacks(stack1, stack2, stack3):
        """Each of stack1, stack2 and stack3 consists of an Ambassador, Assassin, Captain, Contessa and Duke.
        We deal the top card of stack1 to player 1, the top card of stack2 to player2 and then the top card of
        stack3 to player1 and the second card of stack3 to player2. The remainder of stack3 becomes the deck.
        """
        cards = dict()
        cards[1] = [stack1.pop()]
        cards[2] = [stack2.pop()]
        cards[1].append(stack3.pop())
        cards[2].append(stack3.pop())
        deck = stack3
        return cards, deck
