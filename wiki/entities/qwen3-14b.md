---
title: Qwen 3 14B (capacity experiment)
kind: entity
first_seen: 2026-04-17
last_updated: 2026-04-17
status: complete
---

## What it is

Qwen 3 14B is the larger base model used in a capacity-scaling experiment alongside the
[[v10-adapter]] joint-training work. It is NOT the LEM production base — [[qwen3-1.7b]]
remains the primary base. The experiment answers whether capacity alone can break the
~68/100 rationalization ceiling observed on 1.7B. (commit message @ 0c7392f)

## Capacity experiment results

Adapter: `jasonyandell/qwen3-14b-texas42-stage0-v9` — trained on the same v9 comprehension
data as [[v9-adapter]] (no joint rationalizations, for clean comparison).

| Metric | 1.7B v9 | 14B v9 |
|---|---|---|
| Comprehension overall | 83% | **86%** (+3pp) |
| Final loss | 0.205 | **0.125** |
| `partner_response` | 48% | **75%** (+27pp) |
| `beaters_in_unseen` | 46% | **61%** (+15pp) |
| Rationalization | 68/100 | **97/100** (+43pp) |
| `visibility_audit` | 0% | **0%** (identical) |

List-enumeration categories improve substantially. Lucid multi-factor reasoning emerges
in rationalizations. (commit message @ 0c7392f)

## Key findings

**Capacity helps enumeration, not structure.** `visibility_audit` at 0% on both 1.7B and
14B confirms long-enumeration answer format is a structural problem — autoregressive
truncation, not a knowledge gap. See [[topics/single-fact-enumeration]].

**The mask fix later closed the comprehension gap.** After [[decisions/sft-completion-only-loss]]
was applied to the 1.7B v10 trainer, 1.7B-maskfix reached 86% comprehension (same as 14B
v9) at one-third the compute cost. The gap between 1.7B and 14B on comprehension was
largely a gradient-allocation artifact, not a fundamental capacity limit.

**Rationalization gap is real.** The 14B reaches 97/100 rationalization vs 1.7B's 68 —
a 43pp jump the mask fix does not replicate. This was a one-shot capacity probe (single
commit, 0c7392f) — it never ran as a production system, and LEM ended (pivoted to [[burl]])
before any follow-up experiment used it. See [[lem]] and [[v10-adapter]] for the
end-of-replay state, and [[lem-to-burl-handoff]] for the pivot.
