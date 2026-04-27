---
title: Perf Sprint — Lever Ladder
kind: playbook
first_seen: fbe798f
last_updated: fbe798f
status: active
---

Levers ordered by expected ROI for an M5-Max edge-class Burl-style workload. Append wins, retire dead ones, evolve the ladder over time. Each sprint reads from this and writes back to it.

## Active ladder (try in order)

| # | Lever | Notes |
|---|---|---|
| 1 | Bench resilience: per-decision try/except quarantine + `prefill_batch_size=2` | 5-line patch covers both Phase 4 crash modes from sprint 1. Without this, full-N attribution is fragile. |
| 2 | `--batch` sweep: 5 → 8 → 10 → 12 → 16 at Q4 PLE-safe | Q4's memory headroom (peak ~5 GB) admits this. The bench's hard-coded `--batch 5` ceiling needs lifting. |
| 3 | Q4 PLE-safe Unsloth UD + continuous + max-batch stack | [[burl-perf-phase3]] established Q4 byte-equivalence. Stack everything that doesn't conflict. |
| 4 | mlx-lm version bump | Check release notes for `dynamic_roll` / cache fixes. Sprint 1 was on 0.31.2. |
| 5 | `mx.compile` audit on inference hot path | Untested. Could be substantial; could be already-applied. |
| 6 | Spec-decode self-speculation (Q4 draft, bf16 verifier, single-stream lateral) | Doesn't stack with continuous batching but useful for the single-decision path the user also needs. |
| 7 | Cohort abstraction implementation | Per [[harvest-cohort-abstraction]]. Unblocks reliable full-N attribution and full-batch-560 production runs. |
| 8 | Direct mlx Metal kernel audit | nvtx-style profiling — what kernels dominate? Last-resort lever; needs lower-level mlx familiarity. |

## Per-lever budget

30-min wall budget per lever attempt. If a positive ledger row lands, **go deeper before next lever** (e.g., for batch sweep, find the elbow before moving to mlx-lm version bump). If 30 min produces no signal, document the dragon in [[perf-sprint-traps]] and move on.

## Closed levers (don't re-try without a pre-condition met)

- **Parallel tool dispatch within a turn.** Empirical 1434/1434 one-call-per-turn under [[wax-museum]]. See [[burl-perf-phase1]]. Pre-condition for re-opening: future Burl SFT round trains a parallel-tool-call rhythm into the model.
- **LRU prompt cache (turn-to-turn KV reuse).** Two structural failures documented in [[burl-perf-phase2]]: `_merge_caches` heterogeneous-pad penalty + chat-template alignment bug. Pre-condition for re-opening: prefix reuse at wave-start (system+user only, never bridging assistant turns).
- **Spec-decode through `BatchGenerator`.** mlx-lm 0.31.2's `speculative_generate_step` doesn't plumb through. Pre-condition: mlx-lm version that exposes batched spec-decode.
- **Gemma 3 270M IT as cross-family draft model.** Tokenizer collapses Gemma 4 special tokens (`<|tool_call>`, `<|channel>`, etc) to byte-fallback subword sequences. Pre-condition: a draft model whose tokenizer respects Gemma 4's special tokens.

## PLE quant landmine — known sets

- **Broken (do not use):** `mlx-community/gemma-4-*-{4,8}bit`, `unsloth/gemma-4-*-MLX-{4,8}bit` (the non-UD ones). Quantize per-layer-embedding (PLE) layers, produce garbage outputs.
- **Safe:** `FakeRockert543/gemma-4-e2b-it-MLX-{4,8}bit`, `unsloth/gemma-4-E2B-it-UD-MLX-4bit` (UD variant only). [[burl-perf-phase3]] confirms byte-equivalence on the 5-row corpus.

## Append a lever

When a sprint discovers a new viable lever, add a row to the active ladder above with: name, expected ROI, link to the experiment that proved it. When a lever closes, move it to the appropriate closed section with the pre-condition for re-opening.

## Links

[[perf-sprint]] [[perf-sprint-traps]] [[perf-sprint-history]] [[perf-on-the-table]] [[burl-perf-phase1]] [[burl-perf-phase2]] [[burl-perf-phase3]] [[harvest-cohort-abstraction]]
