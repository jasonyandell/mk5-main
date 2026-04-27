---
title: max_tokens=2048 is the floor for batched Burl harvest
kind: decision
first_seen: 063fcac
last_updated: 160ed1c
status: active
---

## Decision

`MODEL_MAX_TOKENS=2048` is the minimum per-turn generation cap for [[burl]] harvest runs that batch [[gemma-4-e2b]] inference via [[mlx-lm]]'s `batch_generate`. 1024 is too tight; 4096 and 8192 are unnecessary and cost KV-cache budget.

## Context

The 560-decision sequential pilot used `max_tokens=8192`. The first batched 2000-decision attempt (v1, `harvest_batched_20260425_031033`) cut to `max_tokens=1024` on the assumption that the sequential headroom was wasteful. It wasn't.

## Evidence

Length-stats comparison across three harvests (sequential 560 @ 8192, batched-560-rerun @ 1024, batched-2000-killed @ 1024):

| harvest | n turns | mean | median | p90 | p99 | max | %≥1024 chars | %≥2800 chars (≈1024 tok cap) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sequential_560 | 3371 | 262 | 54 | 807 | 1770 | 2639 | 6.1% | 0.0% |
| batched_560_rerun (1024 tok) | 3496 | 317 | 54 | 911 | 2814 | 3220 | 8.4% | 1.2% |
| batched_2000_killed (1024 tok) | 8709 | 303 | 54 | 867 | 2795 | 3193 | 7.6% | 1.0% |

Sequential's p99 turn is 1770 chars (~600 tokens); max 2639 chars (~900 tokens). 8192 was 9× over-provisioned. 1024 is *below* the natural distribution — ~1.2% of turns at the 1024-token cap are mid-thought truncations. 2048 covers the full sequential range with comfortable margin.

The downstream cost of truncation was not "1.2% of turns are short." It was that 11.4% of v1 decisions had `belief_called_turns` not starting at turn 1 because the model had run out of budget mid-thinking on turn 1 and never reached the [[belief-trajectory]] call. The harness re-prompted on turn 2, the model recovered, and the trace looked plausible — but a per-decision parity audit against the sequential baseline (see [[burl-2000-harvest]]) showed 43% of decisions had moved between buckets. Bucket-parity at the distribution level was a **false pass**.

## v2 result

`max_tokens=2048` on the v2 2000-decision harvest:

| Gate | Result | vs v1 |
|---|---|---|
| Truncation-at-cap rate | 0.0% | was 1.2% |
| `belief_turns` not starting at turn 1 | 0.0% | was 11.4% |
| Bucket distribution | within ±2pp of sequential | was 5pp+ on `BURL_BREAKS_CONSENSUS` |

Override flag `--max-tokens N` exists for opportunistic re-tuning, but the constant default is now 2048.

## Why not 4096 or 8192

KV-cache memory is `O(batch × seq_len × n_layers × hidden_dim)`. At batch=6 on the M5 Max 48 GB, the resident model + KV reservation runs ~11.3 GB stable. Doubling `max_tokens` to 4096 doubles the worst-case KV reservation; sequential's max of 2639 chars (~900 tokens) is the true ceiling, well under 2048. The data does not justify spending memory we don't need.

## Why not less than 2048

Sequential never produced a turn over 2639 chars. 2048 tokens (≈5500–6500 chars worst case) covers it with margin. 1500 would handle the median + p90 cleanly but would still truncate the p99 tail — and the project's reasoning quality lives in the tail.

[[burl-perf-phase1]] revisited this from the perf side. Two attempts at gate-state-keyed sub-2048 caps (768/768/2048 and 1024/1024/2048) **changed the multi-turn trajectory** on the perf-subset-5: clean-commits flipped to forced-commits and turn counts doubled because Gemma's [[wax-museum]] turn-1 reasoning is consistently 500-700 tokens at temp=0.6. The 2048-flat floor stands. The per-prompt `max_tokens: List[int]` plumbing that Phase 1 wired through `mlx_lm.batch_generate` and `GemmaLocalNativeBatched.step_batch` is reusable; the per-state knob in `burl/wax_museum/schemas.py:max_tokens_for_state` is a future re-tuning surface rather than a current win.

## Companion: per-wave OOM resilience

Doubling `max_tokens` raises the per-batch memory footprint enough that a Metal OOM is plausible in heavy-thinking waves. Mitigated by [[batched-harvest-resilience]] — the resilience layer landed in the same scratch script and quarantines the offending wave rather than killing the run.

## Links

[[burl-2000-harvest]] · [[batched-harvest-resilience]] · [[burl]] · [[gemma-4-e2b]] · [[mlx-lm]] · [[wax-museum]] · [[sources/063fcac]]
