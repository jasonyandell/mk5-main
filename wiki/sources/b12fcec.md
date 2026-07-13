---
title: "Source digest: b12fcec — relax scratchpad validation to hand-only (counts informational)"
kind: source
first_seen: 2026-04-11
last_updated: 2026-04-11
status: active
---

## Commit

- **SHA:** b12fcec3a15f73a5f49afa0bb42f491dcb325ed2
- **Date:** 2026-04-11
- **Author:** Jason Yandell

> fix(lem): relax scratchpad validation to hand-only (counts informational)
>
> 64.5% invalid rate on first iteration — model hasn't seen scratchpad format.
> Hand validation is the critical check (prevents hallucinated-hand bug).
> Counts check is now informational, logged to wandb but doesn't reject traces.
> Also: lenient hand check when model doesn't output HAND label but plays from
> correct hand.

## Files modified

| Path | Change |
|---|---|
| `lem/gemma_star/star_loop.py` | 25 insertions, 11 deletions — counts validation demoted to informational; lenient hand check added |

## First retreat

After 64.5% invalid rate on iteration 0, the COUNTS check is removed as a hard gate. Only HAND must match the remaining dominoes exactly. Counts errors are logged to wandb for diagnostic purposes but do not cause a trace to be discarded.

An additional leniency: if the model does not output a HAND label but plays from the correct hand, the trace is accepted. This handles models that follow the spirit but not the letter of the scratchpad format.

Superseded minutes later by [[78ba940]], which reverts entirely to simple [[k1-grading]].

## Related pages

[[scratchpad-validation]] · [[k1-grading]] · [[star-harness]] · [[scratchpad-v2-iter0]] · [[380f3fa]] · [[78ba940]]
