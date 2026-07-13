---
title: Burl iter-3-rules Adapter (winner)
kind: experiment
first_seen: 2026-04-19
last_updated: 2026-07-13
status: retired
---

Receipt in the [[burl-adapter-line]] — its terminal adapter.

## What it is

The iter-3-rules adapter is the best-scoring [[burl]] adapter — **90% bot-match, 0
retry-exhausted, 100% first-legal** on the 10-decision held-out set. It is trained on a
corpus harvested with `enable_rules_tools=True` and `enable_primer=off`, meaning rules
content is available only as callable tools, not as a text block in the prompt. See
[[rules-as-tools]]. (commit message @ dbadb5f)

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

See [[iter3-comparison]]. (commit message @ dbadb5f)

## Key validation: tools replace memorization

After SFT on the rules-tools corpus, `trick_winner_if` usage INCREASED relative to the
rollout base. The adapter learned to call the rules-tools more aggressively, not less —
directly validating the [[rules-as-tools]] hypothesis that callable tools can
replace memorized primer content. (commit message @ dbadb5f)

## Why the chain stops here

Nothing newer supersedes iter-3-rules — no iter-4 was attempted. How the line ended
([[burl]] dormancy since `dbadb5f`, jud v1's zero-adapter endpoint) is on
[[burl-adapter-line]].

## Caveat: measured under a confound, never re-tested

[[gemma-tool-response-shape]] (first_seen `54f7776`, 17 commits and one day
after this adapter shipped) found that Gemma 4's chat template silently drops `role="tool"`
messages — every Burl rollout through B9, including the one that produced this
adapter's training corpus and its 90% eval, ran with tool outputs invisible to the
model. [[chat-template-fix-validation]] posed the obvious next step — does
iter-3-rules behavior change under a working tool-response harness? — and no re-run of
this adapter appears anywhere in the [[burl]] family. The 90% bot-match figure above is
real as measured, but it is unconfirmed under the conditions the adapter was designed
for: whether the LoRA learned genuine tool-orchestration reasoning or learned to
pattern-match the prompt shape without tool input remains open. The [[burl]] line went
dormant before this was resolved.
