# rlpoker
Reinforcement learning algorithms to play Poker.

NFSP: implemented but not yet working.

## Create a game of Leduc Hold em with 3 cards:
```
leduc = Leduc.create_game(3)
```

Run NFSP on this:
```
python rlpoker/nfsp.py
```

## Solve Leduc Hold Em using cfr
```
strategy = cfr(leduc, num_iters=100000, use_chance_sampling=True)
```

Run this using
```
python examples/leduc_cfr.py
```

