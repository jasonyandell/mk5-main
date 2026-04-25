---
title: "Phase 2: STaR Corpus (N=50)"
kind: experiment
first_seen: fd6032b
last_updated: fd6032b
status: active
---

## Summary

Layer-1 endpoint rolled over N=50 held-out decisions to produce Burl's first STaR training corpus. 54% K1 pass rate. 100% rationalization convergence — which is suspicious. First real STaR corpus for Burl, but the rationalizations are cheap: Gemma restates the hint rather than deriving it.

([burl/eval/run_move4_star_rollout.py @ fd6032b](../sources/fd6032b.md))

## Setup

- **Model:** [[gemma-4-e2b]] with [[experiments/burl-phase1-primer]] prompt (primer + framing block)
- **Decisions:** N=50, seeds 900010+, balanced 5-per-declaration, trick-6, `|legal|>=2`, `eq_gap>=1.0`
- **K1 filter:** keep traces where `E[Q][gemma] >= E[Q][bot]`
- **Rationalization:** re-prompt legal losses with the ground-truth play, collect resulting trace
- **Output:** `burl/data/star_iter0_corpus.jsonl` (gitignored; regenerable from code + endpoint)

## Rollout stats

| Metric | Value |
|---|---|
| K1 pass (wins) | 54% (27/50) |
| Legal losses (rationalized) | 46% (23/50) |
| mean_eq_delta | -5.07 |
| retry_exhausted | 0 |
| illegal | 0 |
| is_legal calls | 72 |
| is_trump calls | 8 |
| eq_outcome_distribution calls | 8 |

## Rationalization finding

23/23 legal losses converged to ground-truth on the first hinted re-prompt. **100% convergence is suspicious.** Gemma behaves as a structured formatter given the answer rather than a second-chance reasoner — it follows the hint through the tool-call protocol rather than deriving the answer independently. Flagged for iter-1.

## Corpus format

```json
{"messages": [
  {"role": "user", "content": "<prompt>"},
  {"role": "assistant", "content": "<concatenated raw_completion across turns>"}
]}
```

Assistant content preserves tool-call envelopes, tool responses, and the terminal `commit_play` call as emitted by the model.

## Primer cost inflation

- Warm decision time: ~16s (spike v2) → ~55–65s (Layer 1) = 3.3×
- Tokens out/decision: 1479 → 5517 chars = 3.7×
- Phase 2 run total: 3099s = $0.69 of $0.91 total
- Trimming the primer to iter-1 would plausibly cut rollout cost to ~$0.25

## Significance

First real STaR corpus for Burl. Shape is healthy (27 wins + 23 rationalizations) but the rationalizations are low-signal — Gemma restates the hint rather than independently justifying the play. The corpus also inherits Layer 1's eq-shy pathology: only 8 `eq_outcome_distribution` calls across 50 decisions, vs 15 across 10 in spike v2. The adapter trained on this corpus will learn "Layer-1 Gemma." See [[experiments/burl-iter0-eval]].

## Related pages

[[burl]] · [[star]] · [[r1-rationalization]] · [[tool-orchestration]] · [[burl-iter0-adapter]] · [[experiments/burl-phase1-primer]] · [[experiments/burl-iter0-eval]] · [[sources/fd6032b]]
