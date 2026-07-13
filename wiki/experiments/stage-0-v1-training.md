---
title: Stage 0 v1 Training Run
kind: experiment
first_seen: 2026-04-10
last_updated: 2026-07-13
status: complete
---

## Summary

First successful [[lem]] training run. [[lora-unsloth]]-style LoRA fine-tune of [[gemma-4-e2b]] on the [[rules-adapter]] Q&A corpus, executed on [[modal]] L4 GPU. Run completed 2026-04-10. Adapter pushed to HuggingFace as [[stage-0-adapter]].

([lem/OVERVIEW.md @ 24ae55a](../sources/24ae55a.md))

## Setup

- **Platform:** [[modal]] L4 GPU (22GB VRAM)
- **Duration:** ~60 min wallclock, 1 epoch, 208 steps
- **Data:** 3500-example Q&A corpus (`lem/rules/qa_corpus.jsonl`), 7 categories — trump membership, led suit, legal moves, trick winners, count math, hand tracking, void inference
- **Method:** LoRA via PEFT (not Unsloth, though Unsloth was the original plan)
- **Precision:** bf16
- **Gradient checkpointing:** enabled, `use_reentrant=False`
- **Eval:** disabled (`eval_strategy="no"` — eval forward pass OOMs on L4)

([lem/gemma_star/train_stage0.py @ df73c8d](../sources/df73c8d.md))

## Recipe

The working training configuration required several non-obvious fixes, documented here as ground truth for reproducibility:

1. Monkey-patch `Gemma4ClippableLinear` to inherit from `nn.Linear` before loading. Without this patch, PEFT cannot target LoRA layers and grad_norm stays at 0.
2. Use bf16 (not fp16 — GradScaler fails with gradient checkpointing).
3. `gradient_checkpointing=True` with `use_reentrant=False` — model is ~20GB in bf16, L4 has 22GB; activations cannot be retained without checkpointing.
4. Skip eval — redundant given that training converges to 100% token accuracy.
5. 1 epoch is sufficient.

Fix source: huggingface/peft#3129. First identified in [[sources/9571a7b]].

## Results

- Loss trajectory: 32 → 15 → 2.8 → 0.001 in 40 steps
- Token accuracy: 100% by step 50
- Considered converged; eval disabled

## Artifacts

- **Adapter:** HuggingFace `jasonyandell/gemma-4-e2b-texas42-stage0` ([[stage-0-adapter]])
- **Wandb run:** `jasonyandell-forge42/lem-stage0`

## Significance

First successful LEM training run. Establishes that [[gemma-4-e2b]] can be LoRA fine-tuned on [[texas-42]] game-engine ground truth via [[modal]], and that the [[rules-adapter]] Q&A corpus produces strong in-format accuracy. Training recipe is now considered reproducible.

Results from [[experiments/second-gemma-contact]] confirm that factual state-tracking (hand tracking) transferred to narration context; compositional rule application (trump membership) did not fully transfer. This motivates Stage 1 [[star]].

## Related pages

[[lem]] · [[gemma-4-e2b]] · [[rules-adapter]] · [[stage-0-adapter]] · [[modal]] · [[lora-unsloth]] · [[star]] · [[k1-grading]] · [[experiments/second-gemma-contact]] · [[experiments/first-gemma-contact]]
