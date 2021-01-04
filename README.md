# rlpoker
Reinforcement learning algorithms to play Poker.

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

You can also use external sampling cfr instead:
```
python -m examples.cfr --game Leduc
```

You can also use external sampling cfr instead:
```
python -m examples.cfr --cfr_algorithm external --game Leduc
```

## Solve Leduc Hold Em using deep cfr
```
python -m examples.deep_cfr --game Leduc
```
# Installation
Use the rlpoker conda environment.
