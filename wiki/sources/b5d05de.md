---
title: "Source digest: b5d05de — Haiku N=30 reference-trace runner"
kind: source
first_seen: b5d05de
last_updated: b5d05de
status: active
---

## Commit

- **SHA:** b5d05de7fd2f9fb9df6f6eb9a892e135c6eaf3db
- **Date:** 2026-04-19
- **Author:** Jason Yandell

> feat(burl): iter-2 prep — Haiku N=30 reference-trace runner
>
> 30-decision run against Haiku 4.5: $0.78, 29/30 complete, 72.4%
> bot-match. Three iter-3 implications:
> 1. conditional_outcome: 0 uses — even at Haiku's ceiling, not reached zero-shot.
> 2. 4 big-gap misses all skipped eq_outcome_distribution (prose-only reasoning).
> 3. Haiku uses 7 distinct tools/decision vs Gemma iter-1's 3, zero retries.
>
> Part of burl-iter2-prep team (T8).

Production N=30 [[reference-trace-distillation]] run of [[haiku-4-5]] (29/30 complete, 72.4% bot-match, $0.78). Three findings with direct iter-3 implications: (1) `conditional_outcome` never used — STaR must synthesize demos to get Gemma to reach for it; (2) all big-gap misses skipped `eq_outcome_distribution`; (3) Haiku uses 7 distinct tools/decision vs Gemma's 3, zero retries — these "go with the grain" patterns are candidates for distillation.

## Related pages

[[reference-trace-distillation]] · [[haiku-4-5]] · [[burl]] · [[sources/1f13f92]]
