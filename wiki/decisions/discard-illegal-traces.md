---
title: Discard Illegal Traces (don't rationalize them)
kind: decision
first_seen: fb47ab3
last_updated: fb47ab3
status: active
---

## Decision

In the [[star-harness]] (all three runners including the single-GPU loop), traces graded `illegal` or `parse_fail` are **discarded**. They are not passed to [[r1-rationalization]]. Only traces graded `fail` (legal but suboptimal) are rationalized. Traces graded `pass` are kept as-is.

## Why

> traces that arrive at impossible states (illegal moves, unparseable actions) are poison — the reasoning chain is corrupted even if intermediate steps looked reasonable. Don't rationalize, just discard.

— commit message, [[sources/fb47ab3]]

A rationalized illegal trace would teach the model to reason toward an impossible state and then correct course — embedding the corrupted reasoning path in the training data alongside the correction. Discarding the whole trace is the cleaner signal.

## Free diagnostic

The illegality rate is a diagnostic for rules comprehension, logged separately as `illegal_rate` in wandb:

- `illegal_rate` ~40% → Stage 0 rules work is insufficient; the model does not yet reliably know what moves are legal.
- `illegal_rate` ~5% → Model has internalized the rules; focus on strategy improvement via [[k1-grading]] / [[r1-rationalization]].

## Tracking

`illegal_rate = (illegal + parse_fail) / total * 100` is now a separate wandb metric. `n_discarded` is also logged per iteration.

## Reversal note

The [[star-harness]] introduced in [[sources/7538016]] rationalized illegal traces alongside legal failures. This decision supersedes that behavior. The prior frontier was short-lived (hours, same day: 2026-04-10).

## Related pages

[[star]] · [[star-harness]] · [[k1-grading]] · [[r1-rationalization]] · [[sources/fb47ab3]] · [[sources/7538016]]
