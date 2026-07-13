---
title: Burl iter-1 LoRA Adapter (mixed)
kind: entity
first_seen: 2026-04-19
last_updated: 2026-04-19
status: superseded
---

## What it is

The Burl iter-1 adapter is the second STaR iteration LoRA fine-tune of [[gemma-4-e2b]] for
[[burl]]. Trained on a trimmed-primer 30-trace STaR corpus. HF repo (private):
`jasonyandell/gemma-4-e2b-texas42-burl-iter1`. Results are mixed: 5/10 retry-exhausted on
the held-out eval, but the 5 completed decisions reach 80% bot-match and mean E[Q] delta
-0.76 — within 1pp of spike v2. (commit message @ 09b841e)

## Training provenance

- **Base**: [[gemma-4-e2b]] (`google/gemma-4-E2B-it`)
- **Corpus**: 30 entries from trimmed-primer rollout (43% K1 on N=30, 0 retry-exhausted)
- **Primer**: ~500 words (trimmed from 1549-word full primer). See [[decisions/primer-tradeoff]].
- **Predecessor**: [[burl-iter0-adapter]] trained on the full-primer corpus

(commit message @ 09b841e)

## Eval results (10-decision held-out)

See [[experiments/burl-iter1-mixed]].

| Metric | Spike v2 | Layer 1 | iter-0 | **iter-1** |
|---|---|---|---|---|
| Completed | 9/10 | 10/10 | 10/10 | **5/10** |
| Retry-exhausted | 1/10 | 0 | 0 | **5/10** |
| Bot-match | 88.9% | 70% | 60% | **80%\*** |
| Mean E[Q] delta | −1.92 | −3.00 | −3.33 | **−0.76\*** |
| `eq_outcome_distribution` | 15 | 2 | 2 | 0 |
| `trump_declared` | 9 | 2 | 0 | 11 |

\* Measured on completed 5 only — not directly comparable to the full-10 baselines.

## Interpretation

Trimmed primer amplified "reason deeply about the position" at the cost of "remember to
emit `commit_play`." The trimmed rules text removed a load-bearing commit-discipline
scaffold present in both the full-primer (Layer 1) and no-framing (spike v2) baselines.

The 5 completed decisions are promising (80% bot-match, mean Δ -0.76). The 5 exhausted
decisions are the blocker.

## Options for iter-2

1. Re-harvest corpus from spike v2 prompt shape (no framing, no primer — tool menu +
   protocol only), trim max_turns, retrain.
2. Keep trimmed primer but add explicit "you MUST emit `commit_play` to end your turn" line
   in the system prompt.
3. N=50 corpus instead of 30; max_retries=7 at eval-time.

(commit message @ 09b841e)

## Superseded

Option 3's spirit was folded into [[topics/rules-as-tools]] and shipped as
[[iter3-rules-adapter]] (90% bot-match, dbadb5f) — the documented successor on this
adapter lineage.
