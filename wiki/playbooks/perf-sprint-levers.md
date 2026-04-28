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
| 1 | `--batch` sweep: 5 → 8 → 10 → 12 → 16 at Q4 PLE-safe | Q4's memory headroom (peak ~5 GB) admits this. Sprint 2 iter 2 lifted the default to 8 and confirmed wall improves ~49% (45.8s → 23.2s) at batch=8 + Q4. **But equivalence gate failed** — Q4 flipped final_play on gi=0 vs bf16. Either real Q4 damage or temp=0.6 sampling jitter; need flip-rate characterization on bf16-only paired runs before re-attempting (see traps note + active row in sprint 2 results.tsv). |
| 2 | Q4 PLE-safe Unsloth UD + continuous + max-batch stack | Q4 "byte-equivalence on the 5-row corpus" was claimed in sprint 1 — sprint 2 iter 2 is the first temp=0.6 paired run vs bf16 and saw a final_play flip on gi=0 (Q4 picked play=6, bf16 picked play=2, signed_delta -7.63). The byte-equivalence claim may need the temp=0 / sampling caveat. Stack everything that doesn't conflict, but only after the gate is re-grounded. |
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
- **Safe:** `FakeRockert543/gemma-4-e2b-it-MLX-{4,8}bit`, `unsloth/gemma-4-E2B-it-UD-MLX-4bit` (UD variant only). Byte-equivalence on the 5-row corpus is confirmed.

## Append a lever

When a sprint discovers a new viable lever, add a row above with name, expected ROI, and a link to the experiment that proved it. When a lever closes, move it down with the pre-condition for re-opening.

## Links

[[perf-sprint]] [[perf-sprint-traps]] [[perf-sprint-history]] [[perf-on-the-table]]
