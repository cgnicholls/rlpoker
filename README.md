# rlpoker
Reinforcement learning algorithms to play Poker

## Create a game of Leduc Hold em with 3 cards:
```
leduc = Leduc.create_game(3)
```

## Solve Leduc Hold Em using cfr
```
strategy = cfr(leduc, num_iters=100000, use_chance_sampling=True)
```
