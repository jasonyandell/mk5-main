---
title: Gus v2 at 1000g with Explicit Voids
kind: experiment
first_seen: 2026-04-21
last_updated: 2026-07-13
status: complete
---

## Summary

Adds engine-computed explicit void features (VoidsEncoder) to the v1 architecture and
scales corpus to 1000 games. Voids provide a modest +1.4pp belief gain; π_me is flat.
The transformer was already inferring voids attentionally from play tokens. (commit message @ 3c02d10)

## Setup

- **Corpus**: 1000 games (chunked generation; glob-load via [[sources/2e4f586]] fix)
- **Architecture**: v1 transformer + VoidsEncoder projecting `[24]`-dim void indicator
  (3 opponents × 8 suits) into d_model, added to pooled state_emb before all four heads
- **Model size**: d=192, 4 layers, 40 epochs
- **Void computation**: per-trick, led suit identified from leader's domino; any non-leader
  not playing a containing-led-suit (non-trump) domino is marked void in that suit

## Results: v1 → v2

| Metric | v1 (1000g) | v2 (1000g) | Delta |
|---|---|---|---|
| π_me bot-match | 66.1% | 65.2% | ~flat |
| belief top-1 | 37.2% | 38.6% | +1.4pp |
| V MAE | 7.6 | 7.7 | flat |
| Q MAE | 12.3 | 12.2 | flat |

Note: comparing v1 vs v2 both at 1000g. The 100g → 1000g scaling alone lifted π_me from
57.9% ([[gus-4head-baseline]]) to 66.1%.

## Interpretation

The transformer was already inferring voids attentionally from the play token sequence.
Explicit void features close a small gap (+1.4pp belief) but are not a large unlock.
v2 is kept as a cleaner architectural baseline; the real next lever is more data and
a larger model. (commit message @ 3c02d10)

## Links

[[gus]] · [[gus-4head-baseline]]
