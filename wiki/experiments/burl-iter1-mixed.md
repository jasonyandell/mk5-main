---
title: "Burl iter-1: Trimmed Primer, Mixed Result"
kind: experiment
first_seen: 09b841e
last_updated: 09b841e
status: active
---

## Summary

Full second STaR iteration on [[burl]]. Trimmed the LEM primer from 1549 → ~400 words, ran N=30 corpus rollout, SFT → [[burl-iter1-adapter]], evaluated on the same 10 held-out decisions. Mixed signal: 5/10 retry-exhausted, but the 5 that completed hit 80% bot-match and mean E[Q] delta within 1pp of spike v2. Meaningful scientific finding; not a ship-ready adapter.

([SPIKE_REPORT.md @ 09b841e](../sources/09b841e.md))

## Setup

- **Primer:** trimmed from 1549 words → ~400 words (`_TRIMMED_PRIMER`, 386 words), inlined into `agent_runner_native.py`
- **Corpus rollout:** N=30 decisions (was 50 in iter-0), seeds 900010+
- **SFT:** same recipe as iter-0 on [[modal]] B200; adapter pushed as `jasonyandell/gemma-4-e2b-texas42-burl-iter1` ([[burl-iter1-adapter]])
- **Eval:** same 10 held-out decisions as all prior runs

## Three-step finding sequence

**Step 1 — Dropping the primer entirely: killed.** Framing block only (no primer): 0% wins, 50% retry-exhausted at 6 decisions. The primer was carrying behavioral load beyond rules teaching. See [[decisions/commit-discipline]].

**Step 2 — Trimmed primer rollout: deeper reasoning, lower K1.** N=30 at 43% K1 (below iter-0's 54%). 0 retry-exhausted. Rollouts ran ~3× slower per-decision than iter-0 (~180 s vs ~56 s wall). Healthier tool diversity: `is_legal` 44, `trump_declared` 4, `is_trump` 1, `eq_outcome_distribution` 2.

**Step 3 — SFT on trimmed corpus: amplifies depth, loses commit discipline.** iter-1 adapter goes 5/10 retry-exhausted on the held-out eval. The 5 completions: 80% bot-match, mean_eq_delta −0.76 — within 1pp of spike v2.

## 4-baseline comparison (same 10-decision held-out throughout)

| Metric | spike v2 | Layer 1 | iter-0 | iter-1 |
|---|---|---|---|---|
| n_completed | 9/10 | 10/10 | 10/10 | 5/10 |
| n_retry_exhausted | 1/10 | 0 | 0 | **5/10** |
| bot_match | 88.9% | 70% | 60% | 80%* |
| mean_eq_delta | −1.92 | −3.00 | −3.33 | −0.76* |
| first_legal_rate | 90% | 100% | 100% | 50% |
| eq_outcome_dist | 15/10 | 2/10 | 2/10 | 0/5 |
| trump_declared | 9/10 | 2/10 | 0/10 | 11/5 |
| cost | $0.05 | $0.15 | $0.42 | $0.21 |

*measured only on 5 completed decisions; not directly comparable to full-10 figures.

## Interpretation

Trimmed primer + SFT amplified "think deeply about the position" at the cost of "remember to emit `commit_play`." The trimmed rules text removed a load-bearing behavioral scaffold. The quality of the committed traces is good; the failure is getting to a commit at all.

## iter-2 options

1. Re-harvest from spike v2 prompt shape (no framing, no primer — just tool menu + protocol). Trim `max_turns`. Retrain.
2. Keep trimmed primer + add explicit "you MUST emit `commit_play` to end your turn" line in system prompt.
3. N=50 corpus; `max_retries=7` at eval-time.

iter-2 was never a standalone adapter page — its results were folded directly into
[[experiments/iter3-comparison]]'s variant table on the way to [[iter3-rules-adapter]].
See that page for how these options actually resolved.

## Related pages

[[burl-iter1-adapter]] · [[burl-iter0-adapter]] · [[experiments/burl-iter0-eval]] · [[decisions/primer-tradeoff]] · [[decisions/commit-discipline]] · [[experiments/iter3-comparison]] · [[burl]] · [[gemma-4-e2b]] · [[sources/09b841e]]
