---
title: "Base Model Pivot: Gemma 4 E2B → Qwen 3 1.7B"
kind: decision
first_seen: 2026-04-16
last_updated: 2026-04-16
status: active
---

## Decision

LEM's base model changes from [[gemma-4-e2b]] to [[qwen3-1.7b]] as of 3465e29 (2026-04-16). All new Stage 0 training targets Qwen. The Gemma pipeline code remains as legacy (still runnable).

## Why

Dual motivation: accuracy and speed.

**Accuracy:** [[qwen3-1.7b]] reaches 100% on `comprehension_eval_v5` after Stage 0 training on the same game-context Q&A corpus where [[gemma-4-e2b]] reached 60%. A 40-point absolute gap on the same data is a strong signal that the base model, not the curriculum, was the remaining bottleneck after [[experiments/stage-0-v4-comprehension-eval]].

**Speed:** Qwen + Unsloth + xformers achieves 36K tok/s on [[modal]] B200 and completes a full training run in ~19 min. Gemma on the same hardware was substantially slower due to three architectural quirks:
- PLE (parameter-efficient-embedding) — non-standard embedding layout
- KV-sharing across layers 15–34 — reuses KV states, complicates PEFT targeting (see [[sources/9571a7b]])
- No flash-attention-2 support — required the HF generate + SDPA + `torch.compile` recipe (see [[sources/26f5ddf]]) and still underperformed

## Scope

- All new Stage 0 training targets Qwen.
- Gemma pipeline code (`train_comprehension.py`, `eval_comprehension.py`, `star_loop.py`, etc.) remains in the repo and is still runnable.
- All Gemma-based adapters (stage-0-adapter, kerry-adapter, v3-adapter, v4-adapter, 15 STaR-iter adapters) remain on HuggingFace as historical checkpoints. They are not deleted.

## Bead resolved

t42-hv08 — the "B200 underutilization" bead had been chased as a GPU/kernel tuning problem. The real answer was model choice: Gemma's architecture does not feed B200 efficiently regardless of kernel configuration.

## Broader lesson

When a GPU looks underutilized despite all standard kernel/batching fixes, consider whether the model itself is architecturally ill-suited to the hardware. Sometimes the fastest path to more throughput is a different model.

## Related pages

[[gemma-4-e2b]] · [[qwen3-1.7b]] · [[v4-adapter]] · [[v5-adapter]] · [[rules-adapter]] · [[modal]] · [[sources/3465e29]] · [[sources/26f5ddf]] · [[sources/9571a7b]] · [[experiments/stage-0-v4-comprehension-eval]]
