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


# TODO:
Investigate why NFSP isn't yet working.

I fixed a bug where I was computing the transitions incorrectly. But the
exploitability still doesn't decrease.

At the moment I don't think the Q-loss is decreasing, so we might need to try a
different learning rate, or to debug that separately.

Other ideas:
* Try training on a smaller game, e.g. OneCardPoker.
* Look through exactly what transitions we are training on.
* Can we just train the deep Q-network separately, to verify it is working?


# 28/12/2019
* Working on refactoring CFR to use a different information set representation.
