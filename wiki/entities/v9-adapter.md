---
title: Stage 0 v9 Adapter (14 categories + verifier)
kind: entity
first_seen: b857299
last_updated: b857299
status: superseded
---

## What it is

The Stage 0 v9 Adapter is the LoRA fine-tune of [[qwen3-1.7b]] trained on an expanded
14-category [[topics/game-context-qa]] corpus. It extends the v5/v7/v8 lineage by adding
structured-reasoning-template categories and introduces the [[topics/rationalization-verifier]].
HF repo: `jasonyandell/qwen3-1.7b-texas42-stage0-v9`. Superseded by [[v10-adapter]].
(commit message @ b857299)

## Category evolution (v7→v9)

| Version | Categories added |
|---|---|
| v7 | `conditional_beat` — "Can X beat Y when Z is led?" with engine-derivable sub-steps |
| v8 | `beaters_in_unseen`, `partner_response`, `intervention_check` |
| v9 | `visibility_audit`, `highest_unseen_in_suit` |

Total at v9: 14 categories (was 5 in [[v5-adapter]]). Each new generator produces
engine-verified ground truth; the structured template in `conditional_beat` propagates
into rationalization contexts the model was never trained on — confirmed real transfer.
(commit message @ b857299)

## Comprehension results

Overall: **83%** on 14 categories after 3 epochs on [[modal]] B200.

Notable findings:
- `visibility_audit` at **0%** on both 1.7B and [[qwen3-14b]] — long-enumeration answers
  fail autoregressive truncation. Structural problem, not capacity. See [[topics/single-fact-enumeration]].
- `highest_unseen_in_suit` (single-fact supporting category) at **100%** — the reliable pattern.
- Compound categories (`beaters_in_unseen`, `partner_response`) did not improve from adding
  atomic supporting categories; composition does not auto-emerge.

## Rationalization verifier

`verify_rationalization.py` applies 6 engine-derived checks: domino validity,
references-visible, hand claims, trump declaration, trump membership, action match.
Filters hallucinations out of training data. See [[topics/rationalization-verifier]].
(commit message @ b857299)

## Rationalization ceiling

Pass rate plateaus at ~68/100 across all 1.7B versions (v7, v8, v9) — motivating the
[[v10-adapter]] joint training and [[qwen3-14b]] capacity experiment.
