---
title: Belief Bayes Ceiling
kind: topic
first_seen: 2026-04-22
last_updated: 2026-07-15
status: complete
---

## The finding

Gus's belief head has reached the Bayesian optimum for top-1 accuracy given the available information. `gus/eval/belief_ceiling.py` computes the theoretical best top-1 achievable on any eval corpus: for each unseen domino, take `argmax_seat P(seat | oracle sampled worlds)`. Since the oracle's worlds ARE the posterior, this argmax is Bayes-optimal (548d32a).

**Current number (clean deck, 2026-07-15): 40.119%** on the regenerated
`corpus_eval_20.pt` ([[otis-phase-r]] R2, registered band [39.3, 41.5]).
The long-standing 39.184% was measured on the April corpus, whose stored
worlds were 27–67% malformed per decision (issue #52,
[[world-sampler-mrv-audit]]) — contamination suppressed the measured
ceiling by ~0.9pp. History:

| corpus | top-1 |
|---|---:|
| Bayes-optimal, clean deck (regen 2026-07-15, 9 decls) | **40.119%** |
| Bayes-optimal, April corpus (contaminated) | 39.184% |
| Gus v3 belief head (measured on the April corpus) | ~38-39% |

Gus matched the contaminated ceiling within noise; whether it also reaches
the clean 40.1% is an open re-measurement (the belief head itself trained
on contaminated worlds). Zeb's 39% baseline wasn't a plateau to overcome —
it was the information ceiling of the game state as then measurable
(548d32a).

## Per-decision breakdown

| d_idx range | Bayes top-1 | interpretation |
|---|---:|---|
| 0-5 (pre-first-trick) | ~33% | pure prior — 1-in-3 over 3 opp seats, no play info |
| 6-15 (mid-hand) | ~36-40% | voids + lead signaling begin to sharpen |
| 18-25 (endgame) | ~50-75% | played dominoes narrow the field deterministically |
| 26 (penultimate) | 100% | only one domino unseen |

Early-hand belief is fundamentally un-sharpenable without more observations. Late-hand belief sharpens because each play rules out seats by void inference and direct exclusion (548d32a, PRACTICALITIES §21).

## How this differs from the earlier "propagation gap"

The [[belief-propagation-gap]] (G6) was about Q_head not using the belief signal — Q_head was world-blind, so calibrating belief had no downstream effect. This ceiling finding is a different and stronger claim: even if Q_head perfectly consumed the belief, top-1 accuracy cannot be improved further. The information is not in the visible state.

These are complementary findings: the propagation gap is an architectural problem (fixable), the Bayes ceiling is an information-theoretic constraint (not fixable).

## Architectural implications

**Top-1 accuracy is a dead lever.** Any belief head exceeding ~39.2% on this corpus is overfitting; lower is underfitting. Gus is tuned.

**The remaining lever is posterior shape (calibration).** §15 already showed distribution-target training closed 47% of the KL gap (0.078 → 0.062). That improvement stalled because downstream heads weren't co-trained on the new distribution. This was tested in [[belief-co-train]]; the answer is no — jointly training belief + world_encoder + Q_head with a distribution target did propagate calibration (KL improved) but Q_head was already at a sweet spot for the old belief's distribution, so downstream regret got slightly worse, not better.

**LAMIR / PIMC / BMCS** all consume the full `P(seat | domino)` distribution, not argmax. Better tail calibration → better-weighted world samples → potentially better look-ahead aggregation. But the co-train experiment showed this doesn't propagate additively in the project's distillation pipeline — and the whole LAMIR/PIMC-lookahead line was subsequently abandoned in favor of [[jud]] (see [[lamir1-ceiling]]).

**Corpus-specific caveat**: the ceiling is a property of the eval corpus.
The 2026-07-15 re-derivation ([[otis-phase-r]]) is the same 20 eval seeds
under the repaired sampler and the 9-declaration mix (decl 8 purged, issue
#51) — a diverse-seed re-measurement per the original [[gen-fleet]] intent
remains not run.

## Diagnostic artifact

`gus/eval/belief_ceiling.py` — ~60 lines, reproducible, reads any eval corpus and reports Bayes-optimal top-1 per-decision-idx. Permanently available as a sanity check (548d32a).

## Links

[[belief-propagation-gap]] [[belief-co-train]] [[consistency-regularizer]] [[lamir1-ceiling]] [[student-distillation]] [[gus]] [[gen-fleet]] [[jud]]
