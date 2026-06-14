---
title: Champion #24 — Auction-Conditioned Belief
kind: experiment
status: complete
task_id: champion-24-auction-belief
first_seen: local-2026-06-13
last_updated: local-2026-06-13
---

# w42-champion-auction-belief

## Summary

**Question:** Does conditioning the [[gus]] belief head on the **completed auction** (who
won, at what level, declaring what) measurably improve its posterior over the hidden tiles —
the unlock the rung-#25 belief-weighting null pointed at?

**Answer: yes, robustly. +2.59pp held-out belief accuracy** (auction student vs an otherwise
identical voids-only control), consistent across three independent corpora and shown to be
auction *information*, not added parameters. The improvement does **not** move arena marks
under oracle play (a clean null), exactly as [[champion]]'s thesis predicts for the
belief→play link when card play is already near-double-dummy.

This is the first *measured win* in the champion's heavy-training frontier (#24/#26).

## Method

- **Mechanism (rung #24):** the auction enters as an explicit **side feature**, mirroring the
  proven `VoidsEncoder` — `gus/model/auction.py::auction_feature_vector` (per-relative-seat
  bid/pass/winner + winning-bid level + a 10-wide declared-trump one-hot, current-player POV)
  → `BidsEncoder` → added to the pooled state embedding (`StudentTransformerFullVoidsAuction`).
  **No tokenizer change**, so existing adapters (the #25 belieflens, the gus bidder) load and
  behave byte-identically. (The handoff feared bid *tokens* would grow the vocab and break
  every adapter; the side-feature seam dissolves that — the same "find the lower-risk seam"
  move as #25's `compute_eq_weighted_mean`.)
- **Data (rung #26):** the existing belief corpus has no real auction (seed deal + imposed
  bid), so conditioning on it teaches nothing. `arena.cli --emit-snapshots` dumps each
  contracted hand's real deal + per-seat auction; `forge.cli.generate_eq_from_snapshots` runs
  the same oracle E[Q] generation on those deals — **with the declarer leading the first
  trick** — and stamps the auction onto each `GameRecordGPU`. Bidder = `net:wp` (0.68 ms/hand).
- **Measurement:** train an auction student and a **voids-only control** on the *same*
  real-auction corpus, identical hyperparams and matched seed; compare best held-out belief
  accuracy. The control isolates the auction feature's contribution. ~1300 train / ~440 eval
  hands per corpus, 35 epochs, on MPS. (The (seed × model) trains are independent → run in
  parallel across cores; the full matrix lands in minutes.)

## Results

**Belief accuracy (auction − voids), generalizing across independent corpora:**

| corpus | deals from | delta | per-seed |
|---|---|---|---|
| A | base-seed 1000 | **+2.42pp** (n=5) | 1.41, 3.38, 3.11, 2.17, 2.05 |
| B | base-seed 2000 | **+2.21pp** (n=3) | 1.56, 2.42, 2.66 |
| C | base-seed 3000 | **+3.12pp** (n=3) | 2.26, 3.82, 3.28 |

Corpus-level mean **+2.59pp**, 95% CI [+1.41, +3.76] (n=3 corpora); **11/11** corpus×seed
deltas positive (min +1.41pp). The auction student is also strikingly *consistent* (47.0–47.5%,
σ≈0.2) where the voids control wanders more (σ≈0.66) — the auction stabilizes belief as well as
raising it. Both sit well above the play-evidence [[belief-bayes-ceiling]] (~39%), because the
corpus is oracle-played and voids-aware; the auction adds on top of that.

**Capacity control (decisive):** a *shuffled-auction* arm — the auction student with its
auction feature sourced from a different game (same BidsEncoder, real feature distribution,
but decorrelated from the deal) — performs at **voids level**:

```
real auction − voids = +2.63pp        shuffled-auction − voids = −0.39pp
```

If the +2.6pp were from the extra ~10k BidsEncoder parameters, the shuffled arm would also be
~+2.6pp. It isn't — it's *slightly below* voids (the useless feature is mild noise the model
learns to ignore). **The gain is auction information, not capacity.**

**No leakage** (adversarial check): the auction feature encodes only public auction scalars,
never per-domino positions; `decl_id` comes from the winner's own hand (arena/auction.py:147),
and the belief target comes from the true deal via a separate path. Empirically, on the real
corpus one auction-feature key maps to 22 distinct belief targets — a leaking feature would
force one-to-one.

**Arena marks: clean null.** Auction belieflens vs uniform `lens:ev` (96 games, belief active,
ESS≈4.8/10): **−0.29 marks/game, 95% CI [−0.92, +0.36]**, make-rate 68.2% vs 70.2%. Sharper
belief does not move the scoreboard under oracle play — the same lesson as rung #25's
belief-weighting null.

## Interpretation

The champion's marginal-value thesis is **auction ≫ belief ≫ utility ≫ play-polish**. #24
confirms the *auction→belief* link is real and strong (+2.59pp, generalizing, information-not-
capacity), and re-confirms that *belief→marks* is weak when card play is already near-oracle.
So #24's value is the **belief quality itself** — which feeds bidding and defense, and is the
substrate the self-play loop (#26 full iteration) compounds — not a direct marks win in an
oracle-play arena. The negative arena result is not a shortfall; it is the same honest
"the lever is elsewhere" finding as #25 and #27, now with the belief link measurably improved.

## Artifacts

- Code: `gus/model/auction.py`, `StudentTransformerFullVoidsAuction` + `BidsEncoder`
  (`gus/model/student.py`), `gus/train/train_v2_voids.py --auction/--shuffle-bids`,
  `champion/belief.py` (live `bid_state`), `forge/cli/generate_eq_from_snapshots.py`,
  `arena.cli --emit-snapshots`. Commits on `forge`: d84e22e, b0b35d3, b4baec7, 5691795.
- Adversarial review (pre-train): 9 findings fixed incl. the CRITICAL declarer-leads bridge
  bug; (post-train) 4-skeptic refutation workflow → no leakage, control/stat caveats answered
  by the multi-corpus + capacity matrix here.
- Raw: `scratch/champion-run/run24_RESULTS.txt`, `run24_measure/`, `run24_corpusB/`,
  `run24_corpusC/` (logs + adapters; gitignored).

## Links

- [[champion]] — rung #24 (Belief v2) + #26 (self-play bridge)
- [[belief-bayes-ceiling]] — the ~39% play-evidence ceiling the auction beats
- [[arena]] — the marks A/B (null, #25-consistent)
- [[gus]] — the belief head this conditions
