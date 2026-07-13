---
title: "Source digest: df73c8d — Stage 0 complete"
kind: source
first_seen: 2026-04-10
last_updated: 2026-04-10
status: active
---

## Commit

- **SHA:** df73c8d42516b5fdfeccf9642c860693df65ba4d
- **Date:** 2026-04-10
- **Author:** Jason Yandell

> feat(lem): Stage 0 complete — adapter trained, second contact shows hand-tracking fixed
>
> Training run completed on Modal L4 (1 epoch, 208 steps, ~60 min):
> - Loss: 32 → 0.001 in 40 steps, 100% token accuracy by step 50
> - Adapter pushed to jasonyandell/gemma-4-e2b-texas42-stage0
> - Wandb: jasonyandell-forge42/lem-stage0
>
> Second Gemma contact (same prompt as first contact, with adapter):
> - FIXED: hand tracking — model correctly reads "remaining: 6-2, 6-1"
>   instead of pulling from initial hand (was the #1 error in first contact)
> - FIXED: final answer is LEGAL and CORRECT (sluff 6-2 or 6-1)
> - REMAINING: trump membership errors (4-4 and 6-4 called trumps under
>   fives-trump). Q&A format didn't fully transfer to narration context.
>
> Infrastructure fixes:
> - ClippableLinear patch needed in BOTH training and inference functions
> - Eval OOMs on L4; disabled for this run (100% train accuracy = converged)
> - Timeout bumped to 4 hours for full training runs

## Files modified

| Path | Change |
|---|---|
| `lem/gemma_star/modal_app.py` | ClippableLinear patch added to inference path; inference now consistent with training |
| `lem/gemma_star/train_stage0.py` | Minor fixes; eval disabled |

## Training evidence

- 1 epoch, 208 steps, ~60 min on [[modal]] L4
- Loss: 32 → 15 → 2.8 → 0.001 in 40 steps
- 100% token accuracy by step 50
- Artifact: `jasonyandell/gemma-4-e2b-texas42-stage0` ([[stage-0-adapter]])
- Wandb: `jasonyandell-forge42/lem-stage0`

## Second contact observations

Same prompt as [[experiments/first-gemma-contact]] (seed 42, fives trump, trick 6), now with [[stage-0-adapter]] loaded and merged:

- Hand tracking: FIXED. Model reads "remaining: 6-2, 6-1" correctly (was #1 error in first contact).
- Final answer: LEGAL and CORRECT ("sluff 6-2 or 6-1").
- Trump membership: REMAINING. 4-4 and 6-4 still called trumps under fives-trump. Q&A format did not fully transfer to narration-context reasoning.

Full analysis: [[experiments/second-gemma-contact]].

## Infrastructure note

The `Gemma4ClippableLinear` monkey-patch (introduced in [[sources/9571a7b]] for training) must also be applied in the inference path. This commit adds it to `modal_app.py`. Inference without the patch would fail to load the LoRA adapter correctly.

## Related wiki pages

[[lem]] · [[gemma-4-e2b]] · [[stage-0-adapter]] · [[rules-adapter]] · [[modal]] · [[experiments/stage-0-v1-training]] · [[experiments/second-gemma-contact]] · [[sources/9571a7b]] · [[sources/24ae55a]]
