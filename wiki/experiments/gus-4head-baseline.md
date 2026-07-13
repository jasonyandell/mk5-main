---
title: Gus 4-Head Baseline (100g)
kind: experiment
first_seen: 2026-04-20
last_updated: 2026-07-13
status: complete
---

## Summary

First run of the full 4-head student (belief + V + π_me + world-conditioned Q) on the
100-game corpus. π_me bot-match reaches 57.9% with no overfit, proving the architecture.
Next bottleneck is data. (commit message @ da21f52)

## Setup

- **Corpus**: 100 games (seeds 0-99 train, 900000-900019 held-out eval)
- **Architecture**: shared transformer encoder + belief_head + V_head + π_me_head +
  world_encoder + Q_head
- **Training**: 20 epochs, MPS (M5 Max)
- **Key change from v1 belief-only**: Q supervision is ~3400× denser per decision than
  belief alone (avg M≈3400 oracle-labeled sampled worlds per decision, each with a 7-action Q vector, vs a single belief target)

## Results

| Metric | Value | Baseline |
|---|---|---|
| π_me bot-match (held-out) | 57.9% | 25% chance |
| belief top-1 (held-out) | 34.5% | 33.3% chance |
| V MAE | 11.9 | Q scale: [-42, +42] |
| Q MAE (legal actions) | 18.4 | — |

**Critical finding**: train and eval π_me track within 1-2 points — no overfit.
The dense [[dense-q-supervision]] regularizes the shared encoder, preventing the
collapse seen in belief-only training. (commit message @ da21f52)

## Per-decision π_me

| Decision | π_me bot-match | Interpretation |
|---|---|---|
| 0 (first play) | ~20% | Near chance, no info |
| 13 (mid-game) | 60% | Clear learning signal |
| 24-27 (end-game) | 100% | Certain — few legal moves remain |

## Architecture status

Proven. Five components work end-to-end: state encoder, belief head, V head, π_me head,
world encoder + Q head. The datasets now accept multiple `.pt` paths or globs so chunked
corpora merge transparently at load time.

## Next bottleneck

Data. 100 games insufficient for belief and value to converge. Chunked 1000-game generation
(MPS can't handle n_games=1000 in one batch — INT_MAX dim limit) enables overnight scaling.
See [[gus-v2-voids-1000g]] for the 1000g result.

## Links

[[gus]] · [[dense-q-supervision]] · [[gus-v0-v1-belief]] · [[gus-v2-voids-1000g]]
