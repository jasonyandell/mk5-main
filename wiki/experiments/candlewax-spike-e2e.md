---
title: "Candlewax Spike: Multimodal PDFs + Engine Fact-Checker + MLX LoRA STaR (E2E)"
kind: experiment
first_seen: 0545342
last_updated: 0545342
status: active
---

## Summary

End-to-end spike under `burl/candlewax_spike/`: candlewax PDF rendering, Opus/Haiku rollout harness, local Qwen3.6-35B-A3B via mlx-vlm, engine fact-checker (`post_commit_sim`), and local LoRA STaR. Closes "can we STaR a small VL model on 42 decisions locally" — yes on plumbing. Pivots away from LLM-as-reasoner as Burl's primary path.

([burl/candlewax_spike/ @ 0545342](../sources/0545342.md))

## Pipeline

- **render.py / render_minimal.py:** candlewax PDF rendering (v1 feature-rich, v2 minimal with win-region highlight)
- **agent.py / batch_live.py:** Opus/Haiku rollout via Claude Agent SDK with prediction block in system prompt
- **qwen_local.py / qwen_batch.py:** local Qwen3.6-35B-A3B via mlx-vlm with Hermes-style + commit-shortcut parser
- **post_commit_sim.py:** engine rolls trick forward with oracle-argmax to verify model's structured predictions against ground truth
- **star_filter.py:** K1 soft-margin + prediction-winner-match gate
- **qwen_lora_train.py / qwen_eval.py:** mlx-vlm LoRA trainer + held-out eval

## Key receipts

- Image-as-alignment: Haiku d000 13→21 flipped by the candlewax image
- v7 adapter beats base +15% bot-match on two independent held-out ranges (33 examples, lr=3e-6, 10 iters, early-stop at loss ~3)
- STaR iter-2 plateaus without reasoning verifier
- Training collapse mode: any fit past loss ~0.5 on this corpus shape destroys output (repetition loops). LR=1e-5 always collapses; 1e-6/3e-6 with early stop work

## Key pivot

Reasoning-coherence verification is the bottleneck: distinguishing correct reasoning from lucky commit requires an engine-based verifier that is a multi-week subproject, not a weekend. The plumbing works; the blocker is [[reasoning-coherence-verification]]. See [[experiments/iter5-e2-candlewax-null]] for the simpler Burl path hitting the same wall.

## Related pages

[[candlewax-spike]] · [[candlewax]] · [[reasoning-coherence-verification]] · [[mlx-lm]] · [[haiku-4-5]] · [[burl]] · [[sources/0545342]]
