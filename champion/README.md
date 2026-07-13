# champion — the unified belief-state player

The assembly point for the project's player work: one belief-state player that
**bids and plays full games to 7 marks**, built from organs that already existed
as separate artifacts (the [forge](../forge) oracle, [Gus](../gus), the bidding
evaluators, the mark-utility machinery). Vision + ladder live in
[`wiki/entities/champion.md`](../wiki/entities/champion.md); this package is the
code. The measuring stick is [`arena/`](../arena) (full games, paired-seed).

## The decision loop (the target)

At every decision — bid or play:

1. Maintain a posterior over the 21 hidden tiles from **all** evidence.
2. Sample worlds from that posterior (not uniformly).
3. Evaluate each world with the perfect-info oracle (E[Q]) or its student Gus.
4. Choose under **marks-to-7 win probability conditioned on the score**.

## Modules

| File | What it is | Rung |
|---|---|---|
| `utility.py` | `race_wp` (Pascal-recursion marks-to-7 WP table), `MarkEV`, `MarksToSeven` (now with an optional equilibrium-aware pass model), `score_to_utility` | #27 |
| `bidder.py` | `GusBidder` — minimum positive-utility bid over a P(make) table; `GusPointsEvaluator` (live Gus sim) and `NetPointsEvaluator` (the <1ms distilled net) supply the table | #21, #22 |
| `bid_net.py` | `BidNet` — distills `hand → (9 decl × 13 bid) P(make)` from a Gus-backed corpus; `featurize_hand` (63-dim) | #22 |
| `play_risk.py` | `ScoreConditionedLensPlay` — picks a lens per game by score; `select_by_score` | #27 v2 |
| `belief.py` | `belief_weights_for_worlds` — per-world importance weights from the Gus belief posterior; the ZebGameState→Gus-token bridge | #25 |
| `play.py` | `BeliefLensPlay` — belief-weighted world sampling; overrides only LensPlay's marginalization | #25 |

## What's measured (all in the arena, identical-play or identical-bidder paired seeds)

| Question | Result | Read |
|---|---|---|
| Gus bidder vs static heuristic | **wins 84/128 (+1.09 marks/game, CI excl. 0)** — make-rate + doubles-trump access | rung #21 |
| Distilled **net** bidder vs heuristic | **wins 85/128 (+1.29 marks/game, CI excl. 0)** at **0.68 ms/hand** (~1500× faster than the sim; ≥ the sim bidder's own +1.09 edge) | rung #22 |
| Score-conditioned **play** risk vs plain EV | **loses −1.20 marks/game (CI excl. 0)** — risk-shaped lenses sacrifice contracts | rung #27 v2 |
| Belief-**weighted** sampling vs uniform | **null −0.13 (CI incl. 0)** — weights active (ESS ~5–8/10) but play-evidence belief too weak | rung #25 |

The two negatives are the point: they are direct, CI-backed confirmations of the
marginal-value ranking **auction ≫ belief ≫ score-utility ≫ card-play polish**.
Within a 42 hand, marks-optimal play ≈ EV-greedy (marks are quantized), so play
risk-shaping hurts; and belief-weighting is wired + validated but waiting on a
stronger, **auction-conditioned belief (#24)** to pay off.

## Reproduce

```bash
# Auction v0 — Gus bidder vs heuristic (slow: live Gus sim per hand)
python -u -m arena.cli --team-a gus:32,wp+lens:ev --team-b heuristic+lens:ev \
    --n-games 128 --n-samples 10 --device mps

# Net bidder — same edge, ~1500x faster bidding
python -u -m arena.cli --team-a net:wp+lens:ev --team-b heuristic+lens:ev \
    --n-games 128 --n-samples 10 --device mps

# Score-conditioned play risk (measured the wrong lever)
python -u -m arena.cli --team-a heuristic+scorelens --team-b heuristic+lens:ev \
    --n-games 192 --n-samples 10 --device mps

# Belief-weighted world sampling (measured null; needs #24)
python -u -m arena.cli --team-a heuristic+belieflens:ev --team-b heuristic+lens:ev \
    --n-games 128 --n-samples 10 --device mps

# Retrain the bid-strength net (corpus is gitignored; regenerate first)
python -u -m forge.cli.bidding_continuous --samples 32 --limit 500 --output data/bidding-results
python -u -m champion.bid_net --epochs 80

pytest champion/          # unit tests for every module above
```

## Next

**#24 — auction-conditioned belief** is the unlock the #25 null points to: condition
the Gus belief head on the auction (partner's 31 means something) so the
belief-weighted sampler (#25) starts paying off. It needs new training data from
arena self-play (#26) and a tokenizer change that retrains the gus adapters — the
heavy-training frontier. **#28 — teaching battery** runs the champion against the
book's claims with receipts (the *Winning 42, 2nd edition* artifact).
