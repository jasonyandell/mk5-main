---
title: "Source digest: dbadb5f — session 2026-04-19 capture, iter-3 winner, iter-4 null, forward plan"
kind: source
first_seen: 2026-04-19
last_updated: 2026-04-19
status: active
---

## Commit

- **SHA:** dbadb5ffe05d2b6f0d862ebdf993783f916d8d79
- **Date:** 2026-04-19
- **Author:** Jason Yandell

> docs(burl): session 2026-04-19 capture — iter-3 winner, iter-4 null, forward plan
>
> Promotes 14 write-ups from scratch/ to burl/experiments/ as durable
> scientific record. Extends SPIKE_REPORT.md ~400 lines covering Phases
> 5-8. New burl/ITER4_PLAN.md — forward plan for M5 Max local GPU.
> Updates burl/OVERVIEW.md with "Current state" section. No code changes.
>
> Cumulative spike spend: ~$20.50/$40.

Pure documentation commit. Promotes 14 working write-ups to `burl/experiments/` and extends SPIKE_REPORT.md with ~400 lines covering:

- **Phase 5:** iter-2 coverage regression + `strip_thinking()` schema surprise (thought blocks invisible to SFT by default)
- **Phase 6:** iter-3-v2 primer-load-bearing finding — 32% retry-exhausted without the primer
- **Phase 7:** iter-3-rules winner — `trick_winner_if` usage went UP after SFT, validating tools-replace-memorization
- **Phase 8:** iter-4-thoughts byte-identical A/B; LoRA capacity disambiguation plan
- **`conditional_outcome=0` across 145+ decisions** — session's cleanest single datum; every model tested reaches 0 zero-shot

New `burl/ITER4_PLAN.md` documents the ranked experiment queue (E1: LoRA rank sweep locally on M5 Max; E2: candlewax redesign; E3: Pareto-closing scale bump; E4: richer EQ-gate), decisions NOT to make (no Haiku-authored demos), and M5-Max-specific cost/speed tradeoffs.

## Related pages

[[burl]] · [[iter3-rules-adapter]] · [[rules-as-tools]] · [[experiments/iter3-comparison]] · [[experiments/iter4-null-preserve-thoughts]] · [[conditional-outcome-structural-nonuse]] · [[burl-selfplay-arena]]
