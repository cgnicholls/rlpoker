import random
import enum
import abc

from rlpoker.extensive_game import ExtensiveGame


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
    """
    
    def __init__(self, player, coins, cards, lost_cards, deck):
        self.player = player
        self.coins = coins
        self.cards = cards
        self.lost_cards = lost_cards
        self.deck = deck

    def next_player(self):
        return 1 if self.player == 2 else 2

class Action:
    
    @abc.abstractmethod
    def apply(state):
        pass


class Income(Action):
    """The income action increases the player's coins by 1.
    """
    
    @staticmethod
    def apply(state):
        next_player = {1: 2, 2: 1}
        coins = state.coins.copy()
        coins[state.player] += 1
        next_state = State(player=state.next_player(), coins=coins, cards=state.cards.copy(),
        lost_cards=state.lost_cards.copy(), deck=state.deck.copy())
        return next_state





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
