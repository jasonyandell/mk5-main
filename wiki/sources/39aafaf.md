---
title: "Source digest: 39aafaf — arena --tag + Opus vs Haiku head-to-head"
kind: source
first_seen: 39aafaf
last_updated: 39aafaf
status: active
---

## Commit

- **SHA:** 39aafafd7a8010f9e2a75dd9c963698e2bc167ad
- **Date:** 2026-04-19
- **Author:** Jason Yandell

> feat(burl): arena — --tag for head-to-head game artifacts
>
> Appends suffix to output stems so multiple runs against the same seed
> don't clobber each other. Opus vs Haiku on seed 900010: Haiku 0/42
> $0.84 7m22s; Opus 7/35 $5.58 8m50s. trump_declared: Haiku 24× vs
> Opus 1×. is_legal: Haiku 92× vs Opus 27×. conditional_outcome: 0×
> both (4th session observation). Lock fix held clean across 28 Opus turns.

`--tag` flag prevents output file clobbery for multi-run arena comparisons. Runs the Opus vs Haiku head-to-head establishing decision quality and tool-use efficiency deltas. Full analysis: [[experiments/opus-vs-haiku-arena]].

## Related pages

[[experiments/opus-vs-haiku-arena]] · [[selfplay-arena]] · [[conditional-outcome-structural-nonuse]] · [[haiku-4-5]] · [[burl]]
