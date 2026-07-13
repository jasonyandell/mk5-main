---
title: "Source: b4e8ecd"
kind: source
commit: b4e8ecd
date: 2026-04-22
author: Jason Yandell
first_seen: 2026-04-24
last_updated: 2026-04-24
---

## Commit message

> fix(train_pi_opp): eliminate NaN loss from 0 * (-inf) in CE computation
>
> Illegal slot log_probs are -inf after log_softmax masking; zeroing them
> before the dot-product with target avoids IEEE 0 * (-inf) = NaN.

## Files changed

| File | Change |
|---|---|
| `gus/train/train_pi_opp.py` | Fix — zero illegal log_probs before dot-product |

## What this commit establishes

One-line fix for the IEEE 0 × (−∞) = NaN trap in legal-masked cross-entropy. Without
this, the entire gradient is NaN whenever an illegal slot appears in the batch.

## Links

[[gus-pi-opp-training]]
