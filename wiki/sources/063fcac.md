---
title: "Source digest: 063fcac — turn-budget extension on reject + forced-commit on cap"
kind: source
first_seen: 2026-04-24
last_updated: 2026-04-24
status: active
---

## Commit

- **SHA:** 063fcac3aa15cc25eb3e40f5d5dc367c66caf2d7
- **Date:** 2026-04-24
- **Author:** Jason Yandell

> feat(burl/wax_museum): turn-budget extension on reject + forced-commit on cap
>
> Phase A of the Burl harvest workflow. Two guards on `run_decision_waxed`:
>
> 1. Illegal-commit budget extension. When the engine rejects a commit
>    (retry path), bump max_turns by +2 so the rejection message has room
>    to land on the next prompt. Cap at 3 extensions (max +6 turns).
>
> 2. Turn-cap forced commit. If the loop exits without a legal commit,
>    pick the highest-E[Q] play we already know about — probed plays first
>    (via ctx.caches), then an oracle scan over legal plays, then the
>    first legal play as a last resort. Mark forced_commit=True in the
>    result. final=-1 is reserved for internal errors only.
>
> Tests in test_harness_guards.py exercise both paths with scripted stubs.
> Pre-Phase-A, 3/29 blunder decisions committed illegally; post-Phase-A
> the rerun is in progress.

## What this commit establishes

Phase A of the harvest workflow: the [[wax-museum]] harness will never produce an illegal commit on a clean run, and a model that runs out of turns lands on a justified play rather than nothing. Both are prerequisites for the 2000-decision corpus harvest (see [[burl-2000-harvest]]).

The two guards:

- **Illegal-commit budget extension** — `max_turns_extensions` field added to the trace, bounded at 3. Each extension adds +2 turns. Used when the engine rejects a commit and the model needs another turn to digest the rejection. Audit on the v2 harvest: 733 total extensions across 2000 decisions, max 3 per decision. The guard fires routinely but stays bounded.
- **Forced-commit fallback** — `forced_commit=True` + `forced_commit_reason` fields. Three-tier fallback: probed plays in `ctx.caches` (preferred — model already saw the E[Q]), oracle scan over `legal_plays` (next), first legal play (last resort). 219 / 2000 decisions on the v2 harvest hit the fallback (10.9%); zero illegal commits.

## Files changed

| File | Lines | Purpose |
|---|---|---|
| `burl/wax_museum/harness.py` | +120 | Both guards, force-commit selector, trace fields |
| `burl/wax_museum/test_harness_guards.py` | new, +280 | Scripted-stub coverage of both branches |

## Related pages

[[wax-museum]] · [[burl]] · [[burl-2000-harvest]] · [[max-tokens-2048-floor]] · [[batched-harvest-resilience]] · [[1bf1885]] · [[d858781]]
