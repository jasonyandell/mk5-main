---
title: "Source digest: aeafe22 — PRACTICALITIES.md split from OVERVIEW"
kind: source
first_seen: 2026-04-20
last_updated: 2026-04-20
status: active
---

## Commit

- **SHA:** aeafe22e2ff75a782c9cb71f9224b099fc1ee9a7
- **Date:** 2026-04-20
- **Author:** Jason Yandell

> docs(burl): PRACTICALITIES.md split — vision stays in OVERVIEW, receipts grow separately
>
> Eight practicalities logged: native tool-call format, dual-use primer,
> model-invented idioms, conditional_outcome zero-shot invisibility,
> max_seq_length truncation, rank-16 LoRA sweet spot, 43→1334 tok/s
> batch ceiling, M5-Max-as-multiplier. Each entry: assumption / observed
> / adapted / evidence.
>
> OVERVIEW.md refreshed surgically — Pareto frontier table, tool surface
> split into 4 subtables, candlewax hints + spike_drivers +
> what_would_change_my_mind surfaced, conditional_outcome flagged.

New `burl/PRACTICALITIES.md` captures eight operational receipts that belong in a living "what we actually learned" log rather than the vision document. OVERVIEW refreshed with current Pareto frontier (iter-3-rules 90% robustness vs iter-1 −0.16 eq-delta) and updated tool surface table.

## Related pages

[[burl]] · [[iter3-rules-adapter]] · [[conditional-outcome-structural-nonuse]] · [[sft-max-seq-length]]
