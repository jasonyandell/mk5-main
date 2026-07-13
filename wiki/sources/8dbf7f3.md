---
title: "Source: 8dbf7f3"
kind: source
commit: 8dbf7f3
date: 2026-04-20
author: Jason Yandell
first_seen: 2026-04-24
last_updated: 2026-04-24
---

## Commit message

> feat(gus): v1 transformer belief student — architecture works, data-bound
>
> Replaces v0 MLP's bag-of-masks with a transformer over tokenized play
> sequences (CLS + DECL + 7 MINE + 24 PLAY = 33-token fixed layout, five
> channels per position: token/type/trick/pos/player_rel). Enables
> attention-based void inference that the MLP's flat features couldn't
> express.
>
> Empirical (100-game corpus, seeds 0-99 train, 900000-900019 eval):
> - v0 MLP: peak eval 34.6% (~chance 33.3%), train reaches 100% — severe overfit
> - v1 transformer: peak eval 37.5%, train reaches 74%, still overfits but
>   per-decision breakdown shows the right shape:
>       decision  0 (first play, 21 unseen): 33% (no info available, chance floor)
>       decision 10 (mid-game, 13 unseen):   42%
>       decision 22 (late-game, 4 unseen):   45%
>       decision 26 (last play, 1 unseen):   75%
>
> Late-game belief IS learning — the floor at decision 0 just mathematically
> caps the aggregated number. The ceiling is data, not architecture.
>
> Moving on to the full 4-head student (belief + V + π_me + world-conditioned
> Q) where Q supervision is ~3400× denser per decision. See BUILD_PLAN.md v1.

## Links

[[experiments/gus-v0-v1-belief]]
