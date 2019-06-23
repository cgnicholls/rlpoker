import unittest

import rlpoker.games.coup as coup

class TestCoup(unittest.TestCase):
    
    def test_canDealInitialCardsToPlayers(self):
        cards, deck = coup.Coup.dealInitialCards()

        self.assertEqual(set(cards.keys()), {1, 2})
        self.assertEqual(len(cards[1]), 2)
        self.assertEqual(len(cards[2]), 2)
        self.assertEqual(len(deck), 3)

    def test_dealInitialCardsGivenStacks(self):
        stack1 = ['A', 'B', 'C', 'D', 'E']
        stack2 = ['F', 'G', 'H', 'I', 'J']
        stack3 = ['K', 'L', 'M', 'N', 'O']

        cards, deck = coup.Coup.dealInitialCardsGivenStacks(stack1, stack2, stack3)
        self.assertEqual(cards, {
                1: ['E', 'O'],
                2: ['J', 'N']
            })

        self.assertEqual(deck, ['K', 'L', 'M'])

    def test_canPlayIncomeAction(self):
        initial_coins = {1: 2, 2: 3}
        cards = {
            1: [coup.Card.DUKE, coup.Card.DUKE],
            2: [coup.Card.AMBASSADOR, coup.Card.CAPTAIN]
        }
        deck = [coup.Card.CONTESSA, coup.Card.ASSASSIN, coup.Card.ASSASSIN]
        lost_cards = {
            1: [], 2: []
        }
        state = coup.State(player=1, cards=cards, lost_cards=lost_cards, coins=initial_coins, deck=deck)

        next_state = coup.Income.apply(state)

        self.assertEqual(next_state.cards, cards)
        self.assertEqual(next_state.player, 2)
        self.assertEqual(next_state.lost_cards, lost_cards)
        self.assertEqual(next_state.coins, {
            1: 3,
            2: 3
        })
        self.assertEqual(next_state.deck, deck)
