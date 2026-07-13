---
title: Kerry Adapter (Stage 0 v2)
kind: experiment
first_seen: 2026-04-11
last_updated: 2026-07-13
status: superseded
superseded_by: v3-adapter
---

Receipt in the [[stage-0-adapter-line]].

## What it is

The Kerry Adapter is the Stage 0 v2 LoRA fine-tune of [[gemma-4-e2b]], trained on the
[[kerry-curriculum]] (15,000 examples) and published at HuggingFace as
`jasonyandell/gemma-4-e2b-texas42-stage0-kerry`. It supersedes [[stage-0-adapter]] as the
starting point for Kerry-branch [[star]] iterations. (commit messages @ f8cdbe7, 43009a4)

## Training provenance

- **Base model**: [[gemma-4-e2b]] (`google/gemma-4-E2B-it`)
- **Training data**: [[kerry-curriculum]] — 15,000 engine-generated examples structured
  after Kerry Newberry's Learner's Guide (sections A/B/C/D; section C weighted at 40% for
  following-suit tricky cases)
- **Method**: [[lora-unsloth]] LoRA fine-tune on [[modal]] B200, 150 steps
- **HF repo**: `jasonyandell/gemma-4-e2b-texas42-stage0-kerry`

(commit message @ f8cdbe7)

## Third contact results

Evaluated on the same prompt used for first and second contact (seed 42, fives trump,
trick 6). See [[third-gemma-contact]].

| Dimension | Kerry adapter |
|---|---|
| Hand tracking | Correct (maintained from v1) |
| Led suit | Correct |
| Void recognition | Correct ("you hold no fours") |
| Trump non-membership | **Fixed** — correctly says "no fives, no trump" for 6-2, 6-1 |
| Trump membership | Still partially wrong — 6-4 still called trump under fives |
| Strategic reasoning | Dramatically deeper (evaluates both options) |
| Final answer | Legal and correct |

The 6-4-under-fives error is the same stubborn error from first contact, now narrowed to
one case. Trump non-membership (the broader class) is fixed. (commit message @ 43009a4)

## Relationship to [[lem]] pipeline

Kerry Adapter is the intermediate Stage 0 step between [[stage-0-adapter]] and [[v3-adapter]].
It was run through 3 STaR iterations (44/42/44% pass, 12–13% illegal) before being
superseded by [[v3-adapter]] (Kerry 15k + 5k [[trump-drilling]] = 20k total, 200 steps).
Kerry alone established that following-suit drilling cuts illegal rate from 33% to ~12%.
(commit message @ a2498e4)
