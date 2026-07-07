---
title: Burl 2000-decision batched harvest (D_required_first)
kind: experiment
first_seen: 063fcac
last_updated: 02d9096
status: complete
---

## What

A 2000-decision [[burl]] harvest collected on fresh seeds from `gus/data/corpus_train_chunk_0-99.pt`. Every decision runs the [[wax-museum]] tool-loop against [[gemma-4-e2b]] (LoRA-free, base) with the [[belief-trajectory]] tool wired in. Each decision's final play is tagged into one of 14 buckets against the K=200 belief-sampled E[Q] oracle (see [[lamir1]]). Output: `scratch/belief_trajectory_rollout/harvest_batched_20260425_072910/` (gitignored — corpus is 2000 decision dirs).

## Why

The 560-decision sequential pilot ([[gus-v3-consistency-full-run]] downstream products) gave [[burl]] a non-trivial-gold pool of 52 rows for [[star]] training. STaR on a corpus that small had previously collapsed (71-row run, loss 0.1, 0/3 eval). 2000 decisions on `D_required_first` was the planned scale-up — same prompt, fresh seeds, batched inference for ~3.6× per-decision speedup.

## v1 contamination + v2 success

A first 2000-decision run kicked off (`harvest_batched_20260425_031033/`, batched, max_tokens=1024) and SIGKILLed at 1398/2000. Bucket parity against the sequential 560 looked plausible at the distribution level — within ±5pp everywhere. The user's morning intuition flagged a suspicion: "we didn't have short-decision problems before batched mode." Three parallel investigation agents reproduced the data (diagnostics were run on a batched 560-decision rerun with the same 1024-token settings, paired against the sequential 560 — see `PARITY_AUDIT.md` / `LENGTH_STATS_COMPARISON.md`):

- 11.4% of batched decisions (64/560) had `belief_called_turns` not starting at turn 1 — i.e. the model had exhausted its 1024-token budget mid-thinking-block on turn 1 and never reached the [[belief-trajectory]] call. The harness re-prompted, the model recovered on turn 2, and the trace looked superficially fine. But the chain of reasoning that had been mid-flight when the budget hit was permanently lost as unstructured `assistant_text` (gi=42 sample: 2730 chars of prose ending mid-sentence at "*Checking for*", no tool call emitted).
- 1.2% of batched turns hit the 1024-token cap (≥2800 chars of generated text); 0% of sequential turns did. Sequential's p99 turn was 1770 chars / ~600 tokens; max 2639 / ~900. 1024 was below the model's natural distribution.
- 43% of decisions (241/560) had moved between buckets relative to the sequential baseline. Aggregate parity was a false pass — distribution-level errors cancelled.

Fix: `MODEL_MAX_TOKENS` 1024 → 2048 (see [[max-tokens-2048-floor]]) plus a per-wave OOM-resilience layer (see [[batched-harvest-resilience]]). Reverted run launched 2026-04-25 07:29:10 as v2. Finished 2026-04-25 13:15:28, wall 5h 46m, zero quarantine fires across 333 batches.

## Quality gates (v2)

| Gate | Sequential 560 | batched @1024 tok (560 rerun, contaminated) | v2 (2048 tok) |
|---|---:|---:|---:|
| `belief_turns` not starting at turn 1 | ~0% | 11.4% | **0.0%** |
| Truncated at cap (≥2800 chars) | 0.0% | 1.2% | **0.0%** |
| Bailed (no commit at all) | 0 | 0 | 0 |
| Illegal commits | 0 | 0 | 0 |
| Forced commits | 12.3% | ~12% | 10.9% |
| n_turns p95 / max | 10 / 16 | contaminated | 11 / 14 |
| Wall (median per decision) | 37s seq | — | 57s batched-of-6 |
| Quarantine fires (5h46m) | n/a | n/a | **0** |

Forced-commit dropped −1.4pp from the sequential baseline. n_turns mean is 6.28 (sequential 5.5) — the larger token budget surfaces a slightly chattier model, not pathologically so.

## Bucket distribution (n=2000)

| Bucket | n | % | Δ vs seq 560 (pp) | Class |
|---|---:|---:|---:|---|
| ALL_AGREE_CORRECT | 860 | 43.0 | −0.2 | gold (trivial) |
| BURL_ALONE_FIXES | 14 | 0.7 | −0.2 | gold (sharp) |
| BOTH_FIX | 52 | 2.6 | +1.0 | gold |
| BURL_INDEPENDENT_RIGHT | 88 | 4.4 | −0.1 | gold (sharp) |
| BURL_FOLLOWS_PI_RIGHT | 48 | 2.4 | +0.1 | gold |
| **STRICT POOL TOTAL** | **1062** | **53.1** | **+0.6** | — |
| → non-trivial gold | 202 | 10.1 | +0.8 | — |
| BURL_BREAKS_CONSENSUS | 299 | 14.9 | −2.4 | loss (sharp STaR target) |
| BURL_INDEPENDENT_WRONG | 148 | 7.4 | +1.5 | loss |
| ALL_AGREE_WRONG | 100 | 5.0 | +0.9 | loss |
| BURL_PARROTS_PI_WRONG | 57 | 2.9 | +0.6 | loss |
| QMEAN_ALONE_FIXES | 37 | 1.9 | +0.1 | loss |
| BURL_PARROTS_QMEAN_WRONG | 41 | 2.0 | 0.0 | loss |
| BURL_DRIFTS_FROM_PI | 37 | 1.9 | +0.1 | loss |
| FORCED_COMMIT | 219 | 10.9 | −1.4 | guarded |
| ILLEGAL | 0 | 0.0 | 0.0 | guarded |
| OTHER | 0 | 0.0 | 0.0 | guarded |

Per-decision strict pool is 1062 (vs 294 from the sequential 560 → +268% absolute, +0.6pp rate). Non-trivial gold (excluding ALL_AGREE_CORRECT) is 202 rows, 3.9× the 52-row sequential count. Sharpest [[r1-rationalization]] target — `BURL_BREAKS_CONSENSUS` — is 299 rows (3.1× the 97-row sequential count). The −2.4pp decline in `BURL_BREAKS_CONSENSUS` directionally confirms the v1 truncation bug was inflating that bucket; some residual delta vs sequential remains and is worth a per-decision-trajectory comparison before declaring full parity. (No paired test against sequential is possible: v2 ran on fresh seeds 0–71 from chunk_0-99, disjoint from the sequential 560's seeds 900000–900019. The closest paired design is the 1392 aligned decisions shared by the killed v1 run and v2 — same seed/declaration/seat — which isolates the 1024→2048 token-cap effect within batched mode; true parity with sequential would need a 2048-token batched rerun on the 900000-seed set.)

## Runtime characteristics

- Batch size 6, `max_tokens=2048`, `D_required_first` variant.
- Wall total 5h 46m (= 333 batches × ~62s each median; 333 batches × 6 = 1998 + 2 final wave).
- Memory peaked ~11.3 GB on M5 Max 48GB, stable. mlx-lm 0.31.2 broadcast-shapes bug at batch ≥14 with default `prefill_batch_size=8` is documented; batch=6 is well below.
- 733 total `max_turns_extensions` across 2000 decisions (avg 0.37/decision, max 3 — never hits the cap of 3). The guard fires routinely but stays bounded.

## Outputs

- `harvest_batched_20260425_072910/D_required_first/decision_<gi>/` per decision: `trace_summary.json`, `events.jsonl`, `transcript.live`. Indexed by global decision index `gi ∈ [0, 1999]`.
- `harvest_batched_20260425_072910/corpus_index.jsonl` — 2800 rows (per-decision file from chunk_0-99 has 2800 entries; only gi<2000 carry harvest data; gi≥2000 are tagged `bucket="ILLEGAL"` synthetically because no harvest exists for them). Filter to `global_idx<2000` for harvest-only counts.
- `harvest_batched_20260425_072910/HARVEST_SUMMARY.md` — bucket counts + sample transcript paths per interesting bucket.
- `scratch/belief_trajectory_rollout/HARVEST_REVIEW.html` and `HARVEST_REVIEW_FAMILY.html` — engineer- and family-audience static review pages bundled to https://burl-42-review.pages.dev (read-only public URL).
- `scratch/belief_trajectory_rollout/PARITY_AUDIT.md` and `LENGTH_STATS_COMPARISON.md` — agent-produced artifacts that diagnosed v1.

### Footgun caught (2026-04-25)

A pre-built min-300 corpus from a prior session was inadvertently sourced from `harvest_20260424_133611` (the 560-decision held-out eval set), not this 2000-decision harvest. Train/eval leakage averted by manifest-check during a [[star]] Run-3 scout pass; bad dir renamed `..._FROM_HELD_OUT_EVAL_DO_NOT_TRAIN`. The replacement built from `harvest_batched_20260425_072910` lives at `scratch/belief_trajectory_rollout/star/corpus_strict_min300_FROM_HARVEST_BATCHED_20260425_072910/` (1025 surviving decisions; 2686 train + 665 val rows). Lesson: any plan that says "X already exists from a prior session" must be paired with a manifest verification step before consumption.

## Next

Corpus is ready for [[star]] run-3:
1. Filter-only on the 1062 strict pool (with `--min-assistant-chars 300` per the prior session's findings — strips ~40% of templated short rows uniformly across buckets, no gold-bucket loss).
2. If clean, [[r1-rationalization]] on the 299 `BURL_BREAKS_CONSENSUS` rows.
3. Eval on a held-out sample (sequential 560 baseline is the obvious candidate).

Recipe (rank=8 LoRA, lr=3e-5, 1 epoch, val-loss + early-stopping) follows from [[iter5-e1-rank-sweep]] + the prior session's `star_mlx.py` instrumentation work. Conservative hyperparams chosen because the prior 71-row STaR collapsed at rank=16 + lr=1e-4.

## Status

This corpus fed [[burl-star-run3]] and, upstream of it, [[burl-harvest-2]] — the
harvest's job is done and its outputs are consumed by name in both of those. Last
`burl/` commit in this window is 02d9096 (2026-04-25); the family has had no commits
since 2026-05-07.

## Links

[[burl]] [[wax-museum]] [[gemma-4-e2b]] [[belief-trajectory]] [[gus]] [[star]] [[r1-rationalization]] [[batched-harvest-resilience]] [[max-tokens-2048-floor]] [[sources/063fcac]] [[sources/1bf1885]] [[sources/d858781]]
