---
title: "Source digest: abb1b3d — thread enable_rules_tools + iter-3-rules launcher"
kind: source
first_seen: 2026-04-19
last_updated: 2026-04-19
status: active
---

## Commit

- **SHA:** abb1b3d32ba949e149e723ce5bfe388846ad728b
- **Date:** 2026-04-19
- **Author:** Jason Yandell

> feat(burl): iter-3 prep — thread enable_rules_tools + iter-3-rules launcher
>
> run_move4_star_rollout.py: flag plumbed through _run_one_rollout,
> _gate_one, run_star_rollout. Stats surface tool_histogram_gate +
> dedicated rules_tool_histogram_rollouts/gate. --enable-rules-tools CLI.
>
> run_move4_spike.py: kwarg + flag threaded through; help text flags
> must-match-training-shape constraint.
>
> star_iter3_rules.py: new launcher, adapter burl-iter3-rules.
> 101/101 tests green. Rollout Phase 1 firing in background.

Threads `enable_rules_tools` through the full STaR rollout + eval paths so rollout and EQ-gate see the same tool surface. Adds `star_iter3_rules.py` launcher targeting `jasonyandell/gemma-4-e2b-texas42-burl-iter3-rules`. The eval flag MUST match the adapter's training-time shape — help text documents this constraint.

## Related pages

[[rules-as-tools]] · [[burl]] · [[eq-gate-star]] · [[sources/80704f0]] · [[sources/c698091]]
