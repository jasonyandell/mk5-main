---
title: "Source digest: efad16e — capture 10 STaR iterations results — 30% to 42% pass rate"
kind: source
first_seen: 2026-04-11
last_updated: 2026-04-11
status: active
---

## Commit

- **SHA:** efad16ec87003623c28b5a4d2dd7667cd84ca734
- **Date:** 2026-04-11
- **Author:** Jason Yandell

> docs(lem): capture 10 STaR iterations results — 30% to 42% pass rate

## Files modified

| Path | Change |
|---|---|
| `lem/OVERVIEW.md` | +48 lines (net): "10 STaR iterations" section, inference optimization section, LoRA key mismatch resolved, scratchpad validation documented, compute setup rewritten to B200 |

## Key content added

**Compute setup rewritten to B200 all-in:**
- Training: [[modal]] B200 ($6.25/hr), LoRA step in ~13s
- Inference: B200, batch HF generate with SDPA + `torch.compile`, 120+ tok/s
- Full STaR iteration (5 examples): ~2.5 min, ~$0.26
- Local debugging: llama.cpp CPU, 11 tok/s, free

**vLLM "what didn't work" updated — 3-blocker analysis:**
1. vLLM EngineCore subprocess swallows download progress (looks stuck 5–7 min)
2. `Gemma4ForConditionalGeneration does not support LoRA yet` in vLLM
3. Merged model weight layout (`ClippableLinear` `.linear.weight` suffix) incompatible with vLLM's Gemma4 loader

**Inference optimization recipe documented:**
- `attn_implementation="sdpa"` — PyTorch native flash attention (no `flash-attn` package needed)
- `torch.compile(model, mode="reduce-overhead")` — fused kernels
- Left-padded batching — all prompts in one `generate()` call
- Model loaded once per iteration, reused across inference and rationalization phases

**LoRA adapter key mismatch RESOLVED:** Gemma 4 E2B uses KV-sharing for layers 15–34 — those layers reuse KV states from layers 0–14 and have no `k_proj`/`v_proj`. The adapter is complete; the missing keys were architectural, not a training bug.

**Scratchpad validation documented as attempted and deferred.** See [[scratchpad-validation]] and [[scratchpad-v2-iter0]].

**10-iteration results table and key observations.** See [[star-10-iterations]] for full treatment.

**Next steps documented:** held-out eval on iter 5/7, scratchpad format SFT bootstrap, or increase data diversity.

## Related pages

[[star-10-iterations]] · [[gemma-4-e2b]] · [[star-harness]] · [[scratchpad-validation]] · [[star]] · [[modal]] · [[ff0d0d2]] · [[26f5ddf]]
