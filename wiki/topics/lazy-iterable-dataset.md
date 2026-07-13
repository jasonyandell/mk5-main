---
title: Lazy IterableDataset (streaming corpus loader)
kind: topic
first_seen: 2026-04-21
last_updated: 2026-07-13
status: complete
---

## Overview

`JointWorldFullIterable` is a streaming PyTorch `IterableDataset` for [[gus]] training corpora. It loads joint-world tensor chunks one at a time through a shuffle buffer, bounding memory to `(one chunk + buffer)` regardless of total corpus size (f138069).

## Motivation

`train_v3_consistency` OOM'd when eagerly loading the full 10k-game corpus (~110 GB). The eager `JointWorldFullDataset` class requires the entire corpus in RAM, which caps usable corpus size to available unified memory (f138069).

## Design

- Streams chunks sequentially; shuffle buffer provides stochasticity within and across chunk boundaries.
- Memory ceiling measured at ~3.4 GB for 5 chunks — does not grow with chunk count.
- `--lazy` flag on training scripts opts in; default behavior (eager class) unchanged for small corpora.
- `--buffer-size` and `--length-cache` flags for tuning.
- `__len__` precomputes length from a cache file so the trainer can report epochs correctly without loading data.

Smoke-tested: 2 epochs over 5 chunks (14k items) in 6s wall, memory flat across epochs, item counts match (f138069).

## What it unlocks

The loader shipped and works; the fleet it was sized for never launched. [[gen-fleet]] (the
Vast.ai distributed generator) never ran past its pre-launch fix list — the 10k-game corpus
actually used to train [[consistency-regularizer]] (v3) was generated locally, not on fleet
infrastructure. What the loader unlocked in practice:

- **10k+ game corpora**: training [[consistency-regularizer]] (v3) on the locally-generated 10k corpus.
- **N-declaration-per-seed**: each seed generates up to 10 declaration variants; the iterable loader handles the resulting chunked layout transparently.
- **Scale experiments**: arbitrary corpus growth without architecture changes to the training scripts — the growth path this enabled was local, not distributed.

## Links

[[gus]] [[dense-q-supervision]] [[consistency-regularizer]] [[gen-fleet]]
