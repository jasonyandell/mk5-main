---
title: "Bourne-0: a field net written directly, without training"
kind: experiment
first_seen: 2026-11-02
last_updated: 2026-11-02
status: complete
---

## Result

**A functioning field net can be written from zero without training.** Bourne-0
implements a modest public-information policy entirely in integer weights and
ReLU gates. Its first frozen version wins **68.90% [67.31%, 70.49%]** against
uniform random legal play and **41.27% [39.88%, 42.66%]** against the trained
field, over 1,008 paired deals (2,016 hands) per opponent. The trained field is
stronger in this test. These are standalone field-policy trials, not complete
[[scenario-player]] matches or evidence for replacing its deployed net.

Jason requested the construction after reading
[Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/).
The experiment uses tool-assisted tensor construction. It is not the article's
tool-free setting, a blinded replication, or a discovery of hidden abilities:
the author had already examined the trained field in [[zeb-weight-space-2026-11-02]].
No trained tensor, example, forward pass, optimizer, or loss enters the writer.
The network is newly constructed; its policy is explicit algorithm compilation.

## What was written

`zeb/scripts/written_field.py` creates zero arrays and sets **1,990 nonzero
parameters out of 53,332**. The architecture is **343 → 111 → 108 → 28**, using
the existing `field-v2` encoder, two integer ReLU layers with shift zero, and
the existing legal-action mask and lowest-real-tile tie rule. The serialized
network is 108,175 bytes, SHA-256
`090b667daa24dcad8a9875752267374bbe6881d339c46f32077d731a2ffd3731`.

The first layer encodes the led suit, absence of higher-ranked tiles in the
current trick, opponent winning, last-seat partner winning, leading for the
bidding side, tile count, and absence of higher suit followers outside the
actor's hand and played tiles. The second layer combines those predicates:

| Circumstance | Output contribution for a tile |
|---|---:|
| Can take the current trick from an opponent | +256 |
| Count on the tile, in units of five points | −4 per unit |
| Playing last with partner winning | +64 per count unit |
| Leading with no higher suit follower outside own hand or played tiles | +64 |
| Bidding side leading a trump | +32 |
| Always, for canonical tile `(a,b)` | `−a−b−12[a=b]−20[a=6]` |

“Suit follower” excludes a possible off-suit ruff: the leading bonus is not a
claim that the trick is secure. Canonicalization maps trump to pip 6, but count
values come from the count-feature block, preserving the real tile's points.

The ordinary forward pass executes these rules; it calls no handwritten policy
callback. For valid Boolean inputs, `ReLU(x+y−1)` implements conjunction. Count
gating uses `ReLU(c+2g−2)` for `c∈{0,1,2}` and `g∈{0,1}`. Named neurons and the
full recipe are preserved in the generator and manifest.

Two useful properties follow directly from the output scale. When an opponent
is winning and a legal tile can take the trick, +256 dominates every applicable
resource penalty. In last seat with partner winning, each five count contributes
net +60, exceeding the largest difference in the base resource penalty (44).
The net therefore takes an available opposing trick and feeds maximum legal
count in those respective last-seat cases. These are local properties, not
optimal full-hand play.

## Frozen evaluation

The model and protocol were saved at **2026-11-03 05:16:41 UTC** (November 2
locally), before the first forward pass. Only one set of weights was evaluated;
there was no post-result tuning. `zeb/scripts/eval_written_field.py` loads that
blob, checks its hash, and preserves it byte for byte.

- **10,192 legal positions**, from 364 complete hands split between random and
  candidate trajectories, cover seven pip trumps and bids 30–42. All **27,017
  legal tile scores** match an independently coded symbolic policy exactly.
- **552** last-seat positions with an available win against an opponent take
  the trick; **612** last-seat positions with partner winning feed maximum
  legal count. Counts include forced choices.
- All 28 integer outputs and selected tiles match native Rust on **599**
  positions. A uniform-draw, 32-world, floor-16 planner using the net returns
  legal actions on **28** late positions (1,152 nodes); this is an integration
  smoke test, not a planner strength estimate.
- Changing bid 30 to 42 preserves all scores on all 10,192 positions. More
  generally, bid/need/slack and void rows are zero, so their values cannot
  affect this policy.
- Against each opponent, the same **1,008 physical deals** are played twice,
  with Bourne-0 on each team once. Seed 20261102; declaration cycles 0–6,
  bidder cycles all four seats, fixed bid 30, no auction. Each pair includes
  one bidding and one defending hand for the candidate. The environment owns
  the deal; each policy receives only its actor's hand and the public record.

| Opponent field | Wins / hands | Win rate, paired 95% interval | Mean point difference |
|---|---:|---:|---:|
| Uniform random legal | 1,389 / 2,016 | 68.90% [67.31%, 70.49%] | +12.37 |
| Trained `margin_d250k_h128x64` | 832 / 2,016 | 41.27% [39.88%, 42.66%] | −6.93 |

Intervals use 4,000 percentile bootstrap resamples of whole deal pairs. The
same-policy written/written, trained/trained, and random/random controls each
split 112 hands exactly 50/50. The random tape repeats within a physical-deal
pair. The trained baseline has SHA-256
`6fa3723dcae04a2f30839efb091969830c8cc65a7668c3526bd47c9fde0275d5`.

## Scope and artifacts

The network covers pip trumps only. It has no auction, search, contract-aware
strategy, learned conventions, or use of void evidence. Artificial fixed bids
and cyclic declarations do not model a live bidding population. Strength with
a bidder, match scoring, or inside the full planning player remains unmeasured.
Its integer score scale is far smaller than the trained field's: existing
posterior-reading margins must not be reused without a separate calibration.
Plunge and the shipped Zeb model remain unchanged.

Source: writer and evaluator at
[`ad823ebe`](https://github.com/jasonyandell/mk5-main/commit/ad823ebe6fc5d98a89ad5792fa6187f9725e826a),
based on `f1970713`. The original evaluator used that base's compiled core;
the worktree quality gate also builds its own identical core sources.
Data: [frozen model, named-neuron manifest, results, all game replays, and checksums](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-11/tree/34df5ae9c0ca05b5484061baf1b0f16bbbaf7d2a/weight-space/from-scratch-2026-11-02).
Publication verifies every filename and fresh downloads of the model, results,
and complete game ledger. The generator and evaluator can replay the experiment
using the commands in the artifact README.

## Related

[[scenario-player]] · [[zeb-weight-space-2026-11-02]] · [[strategy-fusion-fence]]
