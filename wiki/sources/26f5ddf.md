---
title: "Source digest: 26f5ddf — SDPA + torch.compile + batching on B200 (120 tok/s)"
kind: source
first_seen: 2026-04-10
last_updated: 2026-04-10
status: active
---

## Commit

- **SHA:** 26f5ddfafb53aed23004b54610912eaaf62e1ef4
- **Date:** 2026-04-10
- **Author:** Jason Yandell

> perf(lem): SDPA + torch.compile + batching on B200 (120 tok/s)
>
> Drop vLLM — Gemma4ForConditionalGeneration doesn't support LoRA in
> vLLM 0.19.0, and the multimodal weight layout is incompatible with
> merged checkpoints. Instead, use HF generate() with the three fixes
> that actually matter:
>
> 1. attn_implementation="sdpa" (PyTorch native flash attention)
> 2. torch.compile(model, mode="reduce-overhead")
> 3. Left-padded batching (all prompts in one generate() call)
>
> Smoke test on B200: 120 tok/s batch inference, full 5-example iteration
> in 151s. Model loaded once, reused for inference + rationalization.
>
> Also updates modal_app.py (vLLM offline, simpler) and modal skill
> (monitoring docs with modal app logs streaming).

## Files modified

| Path | Change |
|---|---|
| `lem/gemma_star/star_loop.py` | vLLM removed (again); HF `generate()` with SDPA, `torch.compile`, left-padded batching; model loaded once, reused |
| `lem/gemma_star/modal_app.py` | vLLM removed from single-prompt path; simpler HF generate |
| `.claude/skills/modal/SKILL.md` | Monitoring docs added: `modal app logs` streaming commands, dashboard link pattern |

## The winning recipe

Three fixes that replace vLLM's benefit for this use case:

1. `attn_implementation="sdpa"` — PyTorch native flash attention; no vLLM required for memory-efficient attention.
2. `torch.compile(model, mode="reduce-overhead")` — kernel fusion via inductor; amortized over multiple calls since model is loaded once.
3. Left-padded batching — all prompts submitted in a single `generate()` call with left-padding; eliminates per-prompt overhead.

## Smoke test results

- Throughput: 120 tok/s on B200 (batch)
- Full 5-example iteration: 151s
- Model loaded once, reused across inference phase and rationalization phase

## Why vLLM was abandoned (second time)

`Gemma4ForConditionalGeneration` is a multimodal model class; vLLM 0.19.0 does not support LoRA on this class, and the multimodal weight layout is incompatible with merged checkpoints. The [[star-harness]] loop requires loading an updated adapter after each LoRA training step — an operation vLLM cannot perform for this model. Inference-only vLLM would work but would require a separate training process and adapter reload mechanism, adding complexity with no clear benefit.

## Narrative

This is the third vLLM removal across the project's history: [[8724e93]] (H100, version conflict), [[d913932]] (cleanup), [[2c2b851]] (B200 attempt), and now this. The conclusion at this frontier: HF `generate()` + SDPA + `torch.compile` + left-padded batching is the stable path for [[gemma-4-e2b]] at this frontier. vLLM will not be viable until its LoRA support catches up with `Gemma4ForConditionalGeneration`.

## Related pages

[[star-harness]] · [[modal]] · [[gemma-4-e2b]] · [[2c2b851]] · [[8724e93]] · [[d913932]]
