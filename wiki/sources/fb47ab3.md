---
title: "Source digest: fb47ab3 — discard illegal traces instead of rationalizing them"
kind: source
first_seen: fb47ab3
last_updated: fb47ab3
status: active
---

## Commit

- **SHA:** fb47ab3ffe90a269ab500f0abf2c6423125ab671
- **Date:** 2026-04-10
- **Author:** Jason Yandell

> fix(lem): discard illegal traces instead of rationalizing them
>
> ClaudeAI insight: traces that arrive at impossible states (illegal moves,
> unparseable actions) are poison — the reasoning chain is corrupted even
> if intermediate steps looked reasonable. Don't rationalize, just discard.
>
> The illegality rate becomes a free diagnostic:
> - High (~40%) = Stage 0 needs more rules work
> - Low (~5%) = model knows rules, focus on strategy
>
> Also: inlined grading functions to fix Modal module mount issue,
> applied to both star_loop.py and star_harness.py.

## Files modified

| Path | Change |
|---|---|
| `lem/gemma_star/star_harness.py` | Grading branch split: `illegal`/`parse_fail` now fall through to a `discarded` counter instead of the rationalization path |
| `lem/gemma_star/star_loop.py` | Same grading branch split; `parse_play`/`grade_k1` inlined (previously imported from `star_harness.py`); wandb metrics extended; console output updated |

## Key changes

**Grading branch restructure** (both files): the previously unified `("fail", "illegal", "parse_fail")` branch is now split into:
- `"pass"` → kept as training trace (unchanged)
- `"fail"` → [[r1-rationalization]] path (unchanged)
- `"illegal"` / `"parse_fail"` → `discarded` counter, not rationalized

**Grading functions inlined** (`star_loop.py`): `parse_play` and `grade_k1` were previously imported from `star_harness.py`. Inlining fixes a [[modal]] module mount issue where the `lem` package was not available on the worker at import time. Functions are identical to the originals.

**Wandb metrics extended**:
- `illegal_rate = (illegal + parse_fail) / total * 100` — new metric, logged separately
- `n_discarded` — count of discarded traces per iteration
- `pass_rate` and existing counters unchanged

**Console output**: now distinguishes `"Rationalizing N legal failures (discarding M illegal/unparseable)"`.

## Attribution note

The commit message credits "ClaudeAI insight" for the policy change. This is the first external-to-project reasoning contribution captured verbatim in a commit message in the [[lem]] history.

## Related pages

[[decisions/discard-illegal-traces]] · [[star]] · [[star-harness]] · [[r1-rationalization]] · [[k1-grading]] · [[modal]] · [[sources/7538016]] · [[sources/8c5fbca]]
