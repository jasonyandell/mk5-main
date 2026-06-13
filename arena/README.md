# Arena — full-game Texas 42 harness

The measuring stick for the champion line ([wiki](../wiki/entities/champion.md),
GitHub milestone **Champion**, issue #20). Four seats, real auctions, hands
until a team reaches 7 marks. Replaces forced bid=30: every player in the
stack becomes comparable at the level the game is actually played.

## Run a match

```bash
python -u -m arena.cli --team-a heuristic+lens:ev --team-b bid30+lens:ev \
    --n-games 64 --n-samples 10 --device mps
```

A player is `<bidder>+<play>`:

| bidder | what it does |
|---|---|
| `heuristic` | Roberson risk-budget: bid the minimum legal raise while it stays within `42 − unique_exposed_points` for the best ≥3-tile trump; never bids marks. `heuristic:<min_trumps>,<caution>` to tune. |
| `bid30` | Opens 30 if nothing has been bid, else passes — the arena analogue of the historical forced-bid-30 evals, with a sane declaration. |
| `random` | Bids a uniform legal point bid with probability p (`random:<p>`); noise floor. |

| play | what it does |
|---|---|
| `lens:<utility>` | Oracle E[Q] utility-greedy (`ev`, `p_make`, `mark_ev`, `cvar_10`, `robust_q25`, `disaster`), with **real per-game bid values** feeding the utility thresholds. |
| `random` | Uniform legal. |

## Design

- **Paired-seed halves**: half 1 plays A as seats {0,2}, half 2 as {1,3},
  over identical deal seeds (`hand_seed` is independent of team assignment).
  Card luck cancels; auction divergence between halves is the players' own
  doing and is part of what is measured.
- **Engine split**: `forge/zeb/game.py` owns the play phase; the arena owns
  the auction (`auction.py`, rules.md §4), marks (`score_hand`, §6-§7), and
  the game loop (`engine.py`). A hand enters play as a fully-formed PLAYING
  state — the `_force_bid_30` construction from `w42/lens_v1`, minus the force.
- **Lockstep batching**: every tick, each live game's decision routes to the
  owning side's play policy as one batched call, so GPU players amortize
  across all games (same pattern as `w42/lens_v1/parallel_match.py`).
- **Pass-outs**: tournament rule — reshake, next player shakes. After
  `max_redeals` consecutive pass-outs the shaker is forced to open 30
  (rules.md "common variation"), so the engine is total even between
  never-bid policies.
- **The oracle never sees the bid** (`GameStateTensor` carries decl and
  bidder only), so auction-won contracts are in-distribution for the Q
  model; the bid value enters only through utility thresholds.

v0 scope: point bids 30–42 plus mark bids by the rules (84, then +42); the
shipped bidders declare pip trumps only and never bid past 42. Special
contracts (nello, sevens, plunge) and model-backed bidding are Champion
rungs #21–#22.

## Tests

```bash
python -m pytest arena/ -q
```

CPU-only, no checkpoint needed. `test_hand_metrics.py` checks the
risk-budget port against the original wave-2.B `hand_eval` arithmetic;
`test_engine.py` covers marks bookkeeping, dealer rotation, determinism,
and deal/auction pairing across halves.
