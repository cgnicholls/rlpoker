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

        next_state = coup.IncomeAction.apply(state)

        self.assertEqual(next_state.cards, cards)
        self.assertEqual(next_state.player, 2)
        self.assertEqual(next_state.lost_cards, lost_cards)
        self.assertEqual(next_state.coins, {
            1: 3,
            2: 3
        })
        self.assertEqual(next_state.deck, deck)

    def test_canPlayForeignAidAction(self):
        initial_coins = {1: 2, 2: 3}
        cards = {
            1: [coup.Card.DUKE, coup.Card.DUKE],
            2: [coup.Card.AMBASSADOR, coup.Card.CAPTAIN]
        }
        deck = [coup.Card.CONTESSA, coup.Card.ASSASSIN, coup.Card.ASSASSIN]
        lost_cards = {
            1: [], 2: []
        }
        state = coup.State(player=2, cards=cards, lost_cards=lost_cards, coins=initial_coins, deck=deck)

        next_state = coup.ForeignAidAction.apply(state)

        self.assertEqual(next_state.cards, cards)
        self.assertEqual(next_state.player, 1)
        self.assertEqual(next_state.lost_cards, lost_cards)
        self.assertEqual(next_state.coins, {
            1: 2,
            2: 5
        })
        self.assertEqual(next_state.deck, deck)

    def test_canPlayTaxAction(self):
        initial_coins = {1: 2, 2: 3}
        cards = {
            1: [coup.Card.DUKE, coup.Card.DUKE],
            2: [coup.Card.AMBASSADOR, coup.Card.CAPTAIN]
        }
        deck = [coup.Card.CONTESSA, coup.Card.ASSASSIN, coup.Card.ASSASSIN]
        lost_cards = {
            1: [], 2: []
        }
        state = coup.State(player=2, cards=cards, lost_cards=lost_cards, coins=initial_coins, deck=deck)

        next_state = coup.TaxAction.apply(state)

        self.assertEqual(next_state.cards, cards)
        self.assertEqual(next_state.player, 1)
        self.assertEqual(next_state.lost_cards, lost_cards)
        self.assertEqual(next_state.coins, {
            1: 2,
            2: 6
        })
        self.assertEqual(next_state.deck, deck)

    def test_canPlayStealAction(self):
        initial_coins = {1: 2, 2: 3}
        cards = {
            1: [coup.Card.DUKE, coup.Card.DUKE],
            2: [coup.Card.AMBASSADOR, coup.Card.CAPTAIN]
        }
        deck = [coup.Card.CONTESSA, coup.Card.ASSASSIN, coup.Card.ASSASSIN]
        lost_cards = {
            1: [], 2: []
        }
        state = coup.State(player=2, cards=cards, lost_cards=lost_cards, coins=initial_coins, deck=deck)

        next_state = coup.StealAction.apply(state)

        self.assertEqual(next_state.cards, cards)
        self.assertEqual(next_state.player, 1)
        self.assertEqual(next_state.lost_cards, lost_cards)
        self.assertEqual(next_state.coins, {
            1: 0,
            2: 5
        })
        self.assertEqual(next_state.deck, deck)

        # Check can steal when only 1 coin left
        state = coup.State(player=2, cards=cards, lost_cards=lost_cards, coins={1: 1, 2: 3}, deck=deck)

        next_state = coup.StealAction.apply(state)

        self.assertEqual(next_state.cards, cards)
        self.assertEqual(next_state.player, 1)
        self.assertEqual(next_state.lost_cards, lost_cards)
        self.assertEqual(next_state.coins, {
            1: 0,
            2: 4
        })
        self.assertEqual(next_state.deck, deck)

        # Check can steal when no coins left
        state = coup.State(player=2, cards=cards, lost_cards=lost_cards, coins={1: 0, 2: 3}, deck=deck)

        next_state = coup.StealAction.apply(state)

        self.assertEqual(next_state.cards, cards)
        self.assertEqual(next_state.player, 1)
        self.assertEqual(next_state.lost_cards, lost_cards)
        self.assertEqual(next_state.coins, {
            1: 0,
            2: 3
        })
        self.assertEqual(next_state.deck, deck)

    def test_canPlayAssassinateAction(self):
        initial_coins = {1: 2, 2: 3}
        cards = {
            1: [coup.Card.ASSASSIN, coup.Card.DUKE],
            2: [coup.Card.AMBASSADOR, coup.Card.CAPTAIN]
        }
        deck = [coup.Card.CONTESSA, coup.Card.ASSASSIN, coup.Card.ASSASSIN]
        lost_cards = {
            1: [], 2: []
        }
        state = coup.State(player=2, cards=cards, lost_cards=lost_cards, coins=initial_coins, deck=deck)

        next_state = coup.AssassinateAction.apply(state)

        self.assertEqual(next_state.cards, {
            1: [coup.Card.ASSASSIN],
            2: [coup.Card.AMBASSADOR, coup.Card.CAPTAIN]
        })
        self.assertEqual(next_state.player, 1)
        self.assertEqual(next_state.lost_cards, {
            1: [coup.Card.DUKE],
            2: []
        })
        self.assertEqual(next_state.coins, {
            1: 2,
            2: 0
        })
        self.assertEqual(next_state.deck, deck)

        # Check we can assassinate with only one card left.
        initial_coins = {1: 2, 2: 3}
        cards = {
            1: [coup.Card.ASSASSIN],
            2: [coup.Card.AMBASSADOR, coup.Card.CAPTAIN]
        }
        deck = [coup.Card.CONTESSA, coup.Card.ASSASSIN, coup.Card.ASSASSIN]
        lost_cards = {
            1: [coup.Card.DUKE], 2: []
        }
        state = coup.State(player=2, cards=cards, lost_cards=lost_cards, coins=initial_coins, deck=deck)

        next_state = coup.AssassinateAction.apply(state)

        self.assertEqual(next_state.cards, {
            1: [],
            2: [coup.Card.AMBASSADOR, coup.Card.CAPTAIN]
        })
        self.assertEqual(next_state.player, 1)
        self.assertEqual(next_state.lost_cards, {
            1: [coup.Card.DUKE, coup.Card.ASSASSIN],
            2: []
        })
        self.assertEqual(next_state.coins, {
            1: 2,
            2: 0
        })
        self.assertEqual(next_state.deck, deck)

    def test_canPlayCoupAction(self):
        initial_coins = {1: 2, 2: 7}
        cards = {
            1: [coup.Card.ASSASSIN, coup.Card.DUKE],
            2: [coup.Card.AMBASSADOR, coup.Card.CAPTAIN]
        }
        deck = [coup.Card.CONTESSA, coup.Card.ASSASSIN, coup.Card.ASSASSIN]
        lost_cards = {
            1: [], 2: []
        }
        state = coup.State(player=2, cards=cards, lost_cards=lost_cards, coins=initial_coins, deck=deck)

        next_state = coup.CoupAction.apply(state)

        self.assertEqual(next_state.cards, {
            1: [coup.Card.ASSASSIN],
            2: [coup.Card.AMBASSADOR, coup.Card.CAPTAIN]
        })
        self.assertEqual(next_state.player, 1)
        self.assertEqual(next_state.lost_cards, {
            1: [coup.Card.DUKE],
            2: []
        })
        self.assertEqual(next_state.coins, {
            1: 2,
            2: 0
        })
        self.assertEqual(next_state.deck, deck)

        # Check we can assassinate with only one card left.
        initial_coins = {1: 2, 2: 8}
        cards = {
            1: [coup.Card.ASSASSIN],
            2: [coup.Card.AMBASSADOR, coup.Card.CAPTAIN]
        }
        deck = [coup.Card.CONTESSA, coup.Card.ASSASSIN, coup.Card.ASSASSIN]
        lost_cards = {
            1: [coup.Card.DUKE], 2: []
        }
        state = coup.State(player=2, cards=cards, lost_cards=lost_cards, coins=initial_coins, deck=deck)

        next_state = coup.CoupAction.apply(state)

        self.assertEqual(next_state.cards, {
            1: [],
            2: [coup.Card.AMBASSADOR, coup.Card.CAPTAIN]
        })
        self.assertEqual(next_state.player, 1)
        self.assertEqual(next_state.lost_cards, {
            1: [coup.Card.DUKE, coup.Card.ASSASSIN],
            2: []
        })
        self.assertEqual(next_state.coins, {
            1: 2,
            2: 1
        })
        self.assertEqual(next_state.deck, deck)

        # Check we can't coup with fewer than 7 coins.
        initial_coins = {1: 2, 2: 6}
        cards = {
            1: [coup.Card.ASSASSIN],
            2: [coup.Card.AMBASSADOR, coup.Card.CAPTAIN]
        }
        deck = [coup.Card.CONTESSA, coup.Card.ASSASSIN, coup.Card.ASSASSIN]
        lost_cards = {
            1: [coup.Card.DUKE], 2: []
        }
        state = coup.State(player=2, cards=cards, lost_cards=lost_cards, coins=initial_coins, deck=deck)

        with self.assertRaises(ValueError):
            next_state = coup.CoupAction.apply(state)
