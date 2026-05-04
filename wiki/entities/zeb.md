---
title: Zeb — Learned Belief Model
kind: entity
first_seen: 8d26e0d
last_updated: d858781
status: active
---

## What it is

Zeb is a 3.3M-parameter transformer that predicts opponent-hand distributions from visible
game state. It lives in [[forge]] at `forge/zeb/`. (burl/OVERVIEW.md @ 8d26e0d)

## Output

For each of 28 dominoes: `P(opponent ∈ {L, partner, R} | visible state)` — a probability
vector over the three non-self seats.

## Training

Trained via self-play + oracle distillation. 72% top-1 accuracy on opponent-of-each-domino
prediction. (burl/OVERVIEW.md @ 8d26e0d)

## Role for Burl

[[burl]] calls Zeb via a tool wrapper (`get_belief(player, domino) → [P_L, P_P, P_R]`).
Zeb is the authority on hidden-state beliefs; the [[engine]] is the authority on visible
facts. The composite tool `conditional_outcome(play, assume)` uses Zeb to compute
P(assume | visible), then the E[Q] framework evaluates outcome conditional on that
assumption. (burl/OVERVIEW.md @ 8d26e0d)

## Role for LEM

LEM did not use Zeb directly. The forge pipeline (`generate_eq_games_gpu`) used an E[Q]
policy at N=10 for game generation; Zeb's specific belief model was not a LEM dependency.
Zeb is Burl-first infrastructure, though it is part of [[forge]].

## Parked for Burl (calibration eval, 2026-04-18)

Zeb's advertised 72% top-1 was calculated over all 28 dominoes, including dominoes already
played (trivially known from visible state). These inflate the number.

Hidden-only calibration (dominoes actually uncertain at time of prediction):

| Metric | Value |
|---|---|
| Top-1 accuracy | ~39% |
| Brier score | 0.224 |
| ECE | 0.067 |

39% top-1 on the hidden dominoes (the only ones that matter for belief-based reasoning) is
not reliable enough to anchor decisions on. See [[experiments/zeb-calibration-eval]] and
[[decisions/zeb-parked-eq-primitive]]. (commit message @ d9baf3b)

**Status**: Zeb's tool wrapper (`get_belief`) is shipped in `burl/tools/zeb.py` but parked
behind a flag in Burl's default tool list. The default checkpoint was also corrected from
`large-belief-bootstrap.pt` (untrained, std 0.036) to `lb-v-eq-3740-bootstrap.pt`. Both
fixes are available if Zeb is re-enabled once a better belief model is trained.

**Replacement at d9baf3b**: E[Q] N=10 outcome PDF (`burl/tools/eq_distribution.py`) became
Burl's belief primitive. Counterfactual shift validated: Δmean +15, p_make 0.6→1.0 on seed 900013.

## Superseded for Burl (2026-04-23, commit d858781)

Burl's production belief primitive is now [[belief-trajectory]], which exposes [[gus]]'s
`v3_consistency_10000g` adapter: per-domino posterior, shift-since-last, V (state value),
CLS attention. Gus provides a higher-quality, calibrated belief model.

Zeb remains parked — not deleted. The `get_belief` tool wrapper in `burl/tools/zeb.py` is
still present and re-enableable. Gus is the production path. (commit message @ d858781)

## Potential role for [[book-strategy-player]] (2026-05-03)

The book-strategy-player architecture (designed 2026-05-03; build pending) opens two
candidate roles for Zeb:

1. **Training-data generator.** Zeb's self-play infrastructure could supply structured
   training data for the strategy-selector model (Model A) or end-to-end policy (Model C).
2. **Model C target.** Zeb's policy head, retrained on strategy-labeled data, has a much
   smaller structured output space (15-30 named strategies vs raw 7-domino choice).
   Faster to train, more interpretable, natively produces strategy attributions.

Speculative until the book-strategy framework is built and recording starts.
