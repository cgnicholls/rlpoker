
from rlpoker.games.leduc import Leduc
from rlpoker.games.card import Card
from rlpoker.games.one_card_poker import OneCardPoker
from rlpoker.cfr import cfr
from rlpoker.best_response import compute_exploitability


def test_best_response_cfr():
    """Test we can run 10 iterations of CFR on Leduc and then compute a best
    response.
    """
    cards = [Card(value, suit) for value in range(3) for suit in range(2)]
    game = Leduc(cards)

    strategy, exploitabilities = cfr(game, num_iters=10,
                                     use_chance_sampling=False)

    exploitability = compute_exploitability(game, strategy)

    print("Exploitability: {}".format(exploitability))
    assert exploitability > 0.0


def test_best_response_cfr_one_card_poker():
    game = OneCardPoker.create_game(n_cards=4)

    strategy, exploitabilities = cfr(game, num_iters=10,
                                     use_chance_sampling=False)

    exploitability = compute_exploitability(game, strategy)

    print("Exploitability: {}".format(exploitability))
    assert exploitability > 0.0
