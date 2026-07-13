---
title: "Source digest: faefca7 — thread enable_primer into run_move4_spike (eval path)"
kind: source
first_seen: 2026-04-19
last_updated: 2026-04-19
status: active
---

## Commit

- **SHA:** faefca70f1616a9cf3a7805951a0c9c93cbd7110
- **Date:** 2026-04-19
- **Author:** Jason Yandell

> feat(burl): iter-3 prep — thread enable_primer into run_move4_spike
>
> --no-primer CLI flag on run_move4_spike, mutually exclusive with
> --enable-rules-tools. enable_primer threaded through _loop,
> run_move4_spike, debug_one. Help text flags must-match-training-shape.
>
> Unblocks eval-time prompt alignment without another threading pass.

Companion to [[sources/65c749c]]: threads `enable_primer` into the eval path (`run_move4_spike.py`) so iter-3-v2 eval can use the same no-primer prompt shape as training. Without this, evaluating a no-primer adapter with the default (trimmed-primer) eval path would silently mismatch.

## Related pages

[[burl]] · [[sources/65c749c]] · [[sources/c698091]]
