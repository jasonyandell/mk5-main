---
title: Stage 0 Progression — Kerry STaR + v3 STaR
kind: experiment
first_seen: a2498e4
last_updated: 8c1bb14
status: active
---

## Summary

Controlled comparison of [[star]] pass/illegal rates across Stage 0 adapter versions, testing whether the [[experiments/star-10-iterations]] plateau was K1-structural (an inherent ceiling of the grading criterion) or curriculum-bounded (fixable by improving the Stage 0 adapter).

Result: curriculum improves the floor confirmedly (illegal rate 33% → 13%, replicated across
Kerry and v3), and nudged the ceiling once, unreplicated (peak pass 42% → 48%, a single data
point at v3 iter-2 that was never repeated across its own 5 iterations). The floor claim is
double-confirmed; the ceiling claim is not.

([sources/a2498e4](../sources/a2498e4.md), [sources/601f622](../sources/601f622.md), [sources/8c1bb14](../sources/8c1bb14.md))

## Motivation

After 15 STaR iterations with [[stage-0-adapter]] (v1), the pass rate plateaued at 38–42% and the [[decisions/discard-illegal-traces]] diagnostic showed ~33% illegal. The proposed ceiling hypothesis (from [[sources/908773a]]) was that K1 grading without fact-verification had a structural ~40% limit. This experiment tests that hypothesis by running fresh STaR iterations starting from better Stage 0 adapters.

## Kerry STaR (a2498e4) — iters 0-2 on [[kerry-adapter]]

| Iter | Pass | Illegal |
|------|------|---------|
| 0 | 44% | 13% |
| 1 | 42% | 10.5% |
| 2 | 44% | 13% |

vs v1 baseline: 30% pass, 33% illegal at iter 0.

> Kerry's curriculum + public state = floor above the old ceiling. Illegal rate cut in half (33% → 13%). Following-suit exercises worked.

— commit message, [[sources/a2498e4]]

## v3 STaR (8c1bb14) — iters 0-4 on [[v3-adapter]]

[[v3-adapter]] = [[kerry-curriculum]] 15k + [[trump-drilling]] 5k = 20k examples, 200 steps on B200.

| Iter | Pass | Illegal |
|------|------|---------|
| 0 | 44% | ~13% |
| 1 | 42% | ~13% |
| 2 | **48%** | ~13% |
| 3 | 47% | ~13% |
| 4 | 38% | ~13% |

New peak: 48% at iter 2. Best adapter: v3 STaR iter-2.

## Stage 0 progression summary

| Stage 0 | Avg pass | Peak | Illegal |
|---------|----------|------|---------|
| v1 (3.5k Q&A) — [[stage-0-adapter]] | ~37% | 42% | 33% |
| Kerry (15k) — [[kerry-adapter]] | ~43% | 46% | 12% |
| v3 (20k + trump drill) — [[v3-adapter]] | ~44% | 48% | 13% |

> Each curriculum round raises the floor.

— commit message, [[sources/8c1bb14]]

## Interpretation

The ingest-10 "K1 ceiling" narrative was too strong. Better Stage 0 lifts both the floor (illegal rate) and the ceiling (peak pass). The illegal rate halved from v1 to Kerry (Section C's following-suit exercises worked), then held at ~13% once trump drilling was added (drilling did not cut illegals further but did raise peak pass).

Concretely: the v1 plateau at ~40% was curriculum-bounded. With Kerry + v3, the plateau rises to ~44–48%.

## Open questions — none pursued

The v3 peak of 48% is itself plateau-like — 5 iters never repeated the 48%. Whether the v3
plateau is rules-bounded or strategy-bounded is unknown. Neither candidate next step
(another Stage 0 improvement round, or bootstrapping [[scratchpad-validation]] for
fact-verified K1) happened. Five days later the project replaced the base model entirely
(Gemma → Qwen 3 1.7B, [[base-model-pivot-qwen]]), and the v4 game-context pivot
two days after that made the flashcard-format v3 plateau moot rather than resolving it.

## Related pages

[[stage-0-adapter]] · [[kerry-adapter]] · [[v3-adapter]] · [[rules-adapter]] · [[trump-drilling]] · [[kerry-curriculum]] · [[star]] · [[k1-grading]] · [[learned-by-playing]] · [[star-10-iterations]] · [[a2498e4]] · [[601f622]] · [[8c1bb14]]
