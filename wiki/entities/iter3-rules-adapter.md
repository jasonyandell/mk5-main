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

## Why the chain stops here

Nothing newer supersedes iter-3-rules — [[burl]] went dormant (`local-2026-05-07`) before
an iter-4 was attempted, dormant since `dbadb5f` (Apr 19) alongside the rest of Burl. As of
jud v1, the project's play mechanism no longer consumes any LoRA adapter at all:
[[champion]] (line ~166) states the jud v1 capstone runs "zero adapter," with `judplay`
replacing `lens:ev` with greedy 1-ply value play, oracle-free at runtime. The chain stopped
because the mechanism it fed was abandoned, not because an iter-4 or winner exists.

## Caveat: measured under a confound, never re-tested

[[decisions/gemma-tool-response-shape]] (first_seen `54f7776`, three commits after this
adapter shipped) found that Gemma 4's chat template silently drops `role="tool"`
messages — every Burl rollout through B9, including the one that produced this
adapter's training corpus and its 90% eval, ran with tool outputs invisible to the
model. [[experiments/chat-template-fix-validation]] posed the obvious next step — does
iter-3-rules behavior change under a working tool-response harness? — and no re-run of
this adapter appears anywhere in the [[burl]] family. The 90% bot-match figure above is
real as measured, but it is unconfirmed under the conditions the adapter was designed
for: whether the LoRA learned genuine tool-orchestration reasoning or learned to
pattern-match the prompt shape without tool input remains open. The [[burl]] line went
dormant before this was resolved.
