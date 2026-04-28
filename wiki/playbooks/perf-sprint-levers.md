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
| 1 | `--batch` sweep: 5 → 8 → 10 → 12 → 16 at PLE-safe quant | CLOSED for current PLE-safe options. Q4 (UD-MLX-4bit) and Q8 (FakeRockert543) BOTH flip gi=0 deterministically at temp=0 (Q4 in iters 2/4/5, Q8 in iter 6 — same play=6 / regret 7.632); Q8 also adds gi=104 damage (play=19 vs bf16 play=6, regret 0.025). Wall improvement is real (Q4 ~42–44%, Q8 ~32%) but the gate is structurally unreachable on this subset for any current PLE-safe quant. Pre-condition for re-opening: a different PLE-safe quant set (different recipe), or a re-frozen subset that excludes gi=0/gi=104, or a widened gate definition. |
| 2 | Q4 PLE-safe Unsloth UD + continuous + max-batch stack | CLOSED. Sprint 2 iter 5 (commit `dac9d28`, reset) ran the bisect: re-ran Q4 UD-MLX-4bit at batch=5 temp=0 (matching the bf16 baseline's batch width). gi=0 STILL deterministically flipped to play=6 with byte-identical regret 7.632 and 60% K1_match — same numbers as iter 4's batch=8 run. Verdict: **pure quant damage**, not batch-prefill-numerics. UD-MLX-4bit's logits on gi=0 prefer play=6 to play=2 regardless of batch width. The 42–44% wall win + ~50% peak-mem reduction is real but the gate is structurally unreachable on this quant. Pre-condition for re-opening: a different PLE-safe Q4 set (different quantization recipe), or a re-frozen subset that doesn't include gi=0-style logit-cliff decisions, or a widened gate definition. |
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
- **PLE-safe but NOT byte-equivalent:** `unsloth/gemma-4-E2B-it-UD-MLX-4bit`. Sprint 2 iter 4 (2026-04-28) at temp=0 batch=8 deterministically flipped gi=0 from bf16's play=2 (regret 0) to play=6 (regret 7.632) on the 5-row subset. Sprint 2 iter 5 (2026-04-28) closed the open question with the bisect: re-ran at batch=5 temp=0; gi=0 still flips to play=6 with byte-identical regret. **Verdict: pure quant damage on gi=0's logit landscape, not batch-prefill numerics.** Q4 UD-MLX-4bit gives ~42–44% wall win + ~50% peak-mem reduction but the equivalence gate is structurally unreachable on this subset.
- **PLE-safe but NOT byte-equivalent:** `FakeRockert543/gemma-4-e2b-it-MLX-8bit`. Sprint 2 iter 6 (2026-04-28, commit `1e82482`, reset) at temp=0 batch=5 deterministically flipped gi=0 to play=6 with **byte-identical** regret 7.632 to Q4 UD-MLX-4bit's flip — gi=0's logit landscape is fragile to ANY quant noise on this checkpoint, not Q4-specific. Q8 also introduced NEW damage at gi=104 (bf16 play=6 K1=True regret=0 → Q8 play=19 K1=False regret=0.025), giving Q8 a wider damage footprint than Q4 (2 K1 flips vs Q4's 1). Wall win 32% (32.8s vs 47.9s baseline, decode 233 tok/s vs 142 tok/s); peak-mem reduction only ~17% (10.16GB vs 12.34GB) — Q8 keeps most of bf16's footprint, so the batch-headroom rationale Q4 had is mostly absent. The "~16× smaller quant noise" prior did NOT hold on gi=0. Verdict: Q8 PLE-safe is dead for the gate-passing path on this subset.
- **Likely-safe (untested at temp=0):** `FakeRockert543/gemma-4-e2b-it-MLX-4bit` (the non-UD Q4). Lower priority than non-quant levers given Q4 UD + Q8 both failed identically on gi=0; would expect the same flip.

## Append a lever

When a sprint discovers a new viable lever, add a row above with name, expected ROI, and a link to the experiment that proved it. When a lever closes, move it down with the pre-condition for re-opening.

## Links

[[perf-sprint]] [[perf-sprint-traps]] [[perf-sprint-history]] [[perf-on-the-table]]
