---
title: Burl iter-3-rules Adapter (winner)
kind: entity
first_seen: dbadb5f
last_updated: dbadb5f
status: active
---

## What it is

The iter-3-rules adapter is the current best [[burl]] adapter — **90% bot-match, 0
retry-exhausted, 100% first-legal** on the 10-decision held-out set. It is trained on a
corpus harvested with `enable_rules_tools=True` and `enable_primer=off`, meaning rules
content is available only as callable tools, not as a text block in the prompt. See
[[topics/rules-as-tools]]. (commit message @ dbadb5f)

## Training configuration

- **Base**: [[gemma-4-e2b]] (`google/gemma-4-E2B-it`)
- **Prompt shape**: `enable_rules_tools=True`, `enable_primer=off` — ~645-byte rules-as-tools
  preamble + 42-aware framing block; no text primer
- **Recipe**: same as iter-2 (rank 16, 3 epochs, LR 1e-4, bf16, sdpa, B200)
- **Predecessor**: [[burl-iter1-adapter]]

## Eval results (10-decision held-out)

| Metric | Spike v2 | iter-0 | iter-1 | **iter-3-rules** |
|---|---|---|---|---|
| Bot-match | 88.9% | 60% | 80%\* | **90%** |
| Retry-exhausted | 1/10 | 0 | 5/10 | **0** |
| First-legal rate | 90% | 100% | 50% | **100%** |
| Mean E[Q] delta | −1.92 | −3.33 | −0.76\* | — |

\* iter-1 measured on completed 5 only.

See [[experiments/iter3-comparison]]. (commit message @ dbadb5f)

## Key validation: tools replace memorization

After SFT on the rules-tools corpus, `trick_winner_if` usage INCREASED relative to the
rollout base. The adapter learned to call the rules-tools more aggressively, not less —
directly validating the [[topics/rules-as-tools]] hypothesis that callable tools can
replace memorized primer content. (commit message @ dbadb5f)
