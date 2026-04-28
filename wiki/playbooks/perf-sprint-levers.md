---
title: Perf Sprint — Lever Ladder
kind: playbook
first_seen: fbe798f
last_updated: fbe798f
status: active
---

Levers ordered by expected ROI for an M5-Max edge-class Burl-style workload. **Suggestions, not procedure** — the loop ([[perf-sprint-loop]]) decides keep/discard on `wall_s_per_decision`. This page is hypothesis fuel: read it for ideas to seed the next iteration, freelance is fine. Append wins, retire dead ones, evolve the ladder.

Lever notes age. mlx-lm releases ship frequently and "untested" or "could be already-applied" notes are stale by default — web-search the upstream changelog or model card before assuming the note is current.

## Active ladder (try in order)

| # | Lever | Notes |
|---|---|---|
| 1 | `--batch` sweep: 5 → 8 → 10 → 12 → 16 at PLE-safe quant | Q4's memory headroom (peak ~6.2 GB at batch=8) admits this. Wall improvement is real and material: iter 2 saw 49% (45.8s → 23.2s) at temp=0.6, iter 4 confirmed 44% (44.6s → 25.0s) at temp=0 with decode 408 tok/s. Use bf16 baseline at temp=0 for the comparison anchor. Pre-condition: a quant whose argmax matches bf16 at temp=0 — Q4 UD-MLX-4bit deterministically flips gi=0 (see PLE landmine table). Try Q8 (`FakeRockert543/gemma-4-e2b-it-MLX-8bit`) next, or disentangle quant damage from batch-prefill numerics by re-running Q4 at batch=5 temp=0. |
| 2 | Q4 PLE-safe Unsloth UD + continuous + max-batch stack | DEPRIORITIZED. Sprint 2 iter 4 showed Q4 UD-MLX-4bit fails the temp=0 gate on gi=0 (deterministic play=2 → play=6 flip, regret jump 0 → 7.632). The iter 3 multimodal-sampler hypothesis was right about temp=0.6, but it was wrong to assume Q4 would converge with bf16 at temp=0. Sprint 1's byte-equivalence claim doesn't survive temp=0 scrutiny on this subset. Move to Q8 or disentangle the batch-vs-quant confound before stacking further on UD-MLX-4bit. |
| 3 | mlx-lm version bump | Check release notes for `dynamic_roll` / cache fixes. Sprint 1 was on 0.31.2; sprint 2 iter 1 confirmed [#1139](https://github.com/ml-explore/mlx-lm/issues/1139) was still open as of 2026-04-28 (no fix landed yet). |
| 4 | `mx.compile` audit on inference hot path | Untested. Could be substantial; could be already-applied. |
| 5 | Spec-decode self-speculation (Q4 draft, bf16 verifier, single-stream lateral) | Doesn't stack with continuous batching but useful for the single-decision path. |
| 6 | Cohort abstraction implementation | Cohort-as-quarantine-unit pattern. Unblocks reliable full-N attribution and full-batch-560 production runs. |
| 7 | Direct mlx Metal kernel audit | nvtx-style profiling — what kernels dominate? Last-resort lever; needs lower-level mlx familiarity. |

## Closed levers (don't re-try without a pre-condition met)

- **Bench resilience (try/except quarantine + `prefill_batch_size=2`).** Applied in sprint 2 iter 1 (commit `bc4fbd9` on `perf/aggressive`). Sync-wave `_apply_step` is now wrapped; both BatchGenerator constructions pin `prefill_batch_size=2`. Variant ran clean at 38.9s (≈baseline 38.3s, gate passed). Pre-condition for re-opening: a future bench rewrite that re-introduces an unprotected `_apply_step` call site or removes the `prefill_batch_size` pin.
- **Parallel tool dispatch within a turn.** Empirical 1434/1434 one-call-per-turn under [[wax-museum]]. Pre-condition for re-opening: a future Burl SFT round trains a parallel-tool-call rhythm into the model.
- **LRU prompt cache (turn-to-turn KV reuse).** Two structural failures: `_merge_caches` heterogeneous-pad penalty + chat-template alignment bug. Pre-condition: prefix reuse at wave-start (system+user only, never bridging assistant turns).
- **Spec-decode through `BatchGenerator`.** mlx-lm 0.31.2's `speculative_generate_step` doesn't plumb through. Pre-condition: mlx-lm version that exposes batched spec-decode.
- **Gemma 3 270M IT as cross-family draft model.** Tokenizer collapses Gemma 4 special tokens (`<|tool_call>`, `<|channel>`, etc) to byte-fallback subword sequences. Pre-condition: a draft model whose tokenizer respects Gemma 4's special tokens.

## PLE quant landmine — known sets

- **Broken (do not use):** `mlx-community/gemma-4-*-{4,8}bit`, `unsloth/gemma-4-*-MLX-{4,8}bit` (the non-UD ones). Quantize per-layer-embedding (PLE) layers, produce garbage outputs.
- **PLE-safe but NOT byte-equivalent:** `unsloth/gemma-4-E2B-it-UD-MLX-4bit`. Sprint 2 iter 4 (2026-04-28) at temp=0 batch=8 deterministically flipped gi=0 from bf16's play=2 (regret 0) to play=6 (regret 7.632) on the 5-row subset. Earlier "byte-equivalence on the 5-row corpus is confirmed" claim from sprint 1 didn't hold; either it was at temp=0.6 with a fortunate landing or batch=5 vs batch=8 prefill-numerics also matter. Q4 UD-MLX-4bit gives ~44% wall win + ~50% peak-mem reduction but the equivalence gate fails. Open question: pure quant damage vs batch-width interaction.
- **Likely-safe (untested at temp=0):** `FakeRockert543/gemma-4-e2b-it-MLX-{4,8}bit`. Q8 in particular is the next floor candidate if Q4 UD-MLX-4bit's gi=0 flip turns out to be quant damage.

## Append a lever

When a sprint discovers a new viable lever, add a row above with name, expected ROI, and a link to the experiment that proved it. When a lever closes, move it down with the pre-condition for re-opening.

## Links

[[perf-sprint]] [[perf-sprint-traps]] [[perf-sprint-history]] [[perf-on-the-table]]
