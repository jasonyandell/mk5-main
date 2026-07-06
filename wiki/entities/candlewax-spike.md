---
title: Candlewax Spike (multimodal VL + local MLX LoRA STaR)
kind: entity
first_seen: 0545342
last_updated: 0545342
status: retired
---

## What it is

The Candlewax Spike is a subproject under `burl/candlewax_spike/` that closes the question:
"Can we STaR a small VL model on Texas 42 decisions locally?" It combines candlewax PDF
rendering, Claude Agent SDK rollout, local Qwen VL inference via [[mlx-lm]], and an engine
fact-checker into one end-to-end pipeline. (commit message @ 0545342)

**Disambiguation**: this project is not born in [[eq-genesis]] (era 3, January 2026), despite
a charter-level claim to that effect. The string "candlewax" does not appear anywhere in the
repo or conversation corpus until 2026-04, naming this project specifically. The January
mechanical ancestor is `ef199b0` (Jan 24), which stores the full 85-bin E[Q] outcome histogram —
a distribution-over-Q the later name happened to echo, not the same artifact and not a rename.
See [[candlewax]]'s concordance section for the full dating.

## Pipeline

| Module | Role |
|---|---|
| `render.py` + `render_minimal.py` + `snapshot*.py` | Candlewax PDF rendering — v1 feature-rich, v2 minimal with win-region highlight |
| `agent.py` + `batch_live.py` + `live_runner.py` | Opus/[[haiku-4-5]] rollout harness via Claude Agent SDK; prediction block in system prompt |
| `qwen_local.py` + `qwen_batch.py` | Local Qwen 3.6-35B-A3B via `mlx-vlm`; Hermes-style + commit-shortcut tool-call parser |
| `post_commit_sim.py` | Engine rolls trick forward with oracle-argmax; verifies model's structured predictions (winner_seat, count_to_my_team, count_to_opponents) against ground truth |
| `star_filter.py` | K1 soft-margin + prediction-winner-match gate, configurable strictness |
| `qwen_lora_train.py` | mlx-vlm LoRA trainer for Qwen 3.6-MoE |
| `qwen_eval.py` | Held-out eval harness with adapter support |
| `index.html` + `viewer.html` | Live dashboard for multi-run browsing |

(commit message @ 0545342)

## Receipts

- Image-as-alignment: [[haiku-4-5]] d000 13→21 score flipped by including the candlewax image.
- v7 adapter beats base +15% bot-match on two independent held-out ranges (33 examples,
  lr=3e-6, 10 iterations, early-stop at loss ~3).
- STaR iter 2 plateaus without reasoning verifier — confirms the bottleneck.
- Training collapse: loss < ~0.5 destroys output generation (repetition loops). LR=1e-5
  always collapses; LR 1e-6/3e-6 with early stop work.

## Key pivot

**Away from LLM-as-reasoner.** Reasoning-coherence verification emerged as the bottleneck —
checking whether a model's reasoning chain is actually consistent with the game state
requires a multi-week verifier subproject, not a weekend spike. See
[[topics/reasoning-coherence-verification]].

The spike closes the VL question and surfaces the verifier gap. Future work on
[[topics/candlewax]] must address this before further STaR iterations compound on
incoherent traces.
