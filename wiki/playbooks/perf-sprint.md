---
title: Perf Sprint — Playbook
kind: playbook
first_seen: fbe798f
last_updated: fbe798f
status: active
---

The "let's friggin rock this perf problem" entry point.

## Kickoff

Paste into a fresh session at the project root:

> Read [[perf-sprint]]. Run a perf sprint targeting `<GOAL>`. e.g. *"drive Burl per-decision latency from current baseline toward ~2s on this M5 Max."*

## The contract

**Metric (one number, lower is better):** `wall_s_per_decision` on `perf_subset_5`, with a paired baseline run immediately before each variant. End-to-end — token generation, tool calls, harness overhead, all of it.

**Equivalence gate (binary, must pass):** `K1_match >= 60%` AND `regret_delta` within ±10% on that same paired run. If the gate fails, the row is `discard` regardless of wall.

**Cycle:** modify, run paired baseline + variant, log one row to the ledger, `keep` or `discard` (advance branch or `git reset`). See [[perf-sprint-loop]].

**Don't give up.** Crashes are work, not a stop sign. Stop only when the metric hits the target or the user types stop.

## Editable surface (likely-candidate set, not strict)

The agent is free to modify anything that helps. These are where wins typically live:

- `burl/eval/bench_decision_latency.py` (lives on the `perf/bench` worktree; bring it forward when needed) — the bench harness, batch flag, paired-baseline plumbing.
- `burl/eval/bench_batch_throughput.py` — the raw-inference reference bench.
- `burl/eval/run_move4_star_rollout_batched.py` — the batched eval harness; sync-wave loop + tool dispatch live here.
- `burl/wax_museum/harness.py`, `schemas.py` — gate state machine + tool dispatch.
- `burl/requirements-mlx.txt` — mlx / mlx-lm version pin.
- Bench flags: `--batch`, `--model-repo`, `--max-tokens`, `prefill_batch_size`.

Read source freely. Add files to this list as the territory teaches you where the wins are.

## The ledger

`scratch/<sprint>/results.tsv`, tab-separated. One row per iteration:

```
commit  wall_s  k1_match  regret_delta  peak_gb  status  description
```

- `commit` — short SHA after the variant edit.
- `wall_s` — the contract metric. Use `0.0` for crashes.
- `k1_match`, `regret_delta` — the equivalence gate values, logged so you can see why a row was discarded.
- `peak_gb` — advisory. Useful because memory headroom unlocks larger batches. Use `0.0` for crashes.
- `status` ∈ {`keep`, `discard`, `crash`}. `keep` requires `wall_s` improved AND gate passed.
- `description` — short freelance text. Hypothesis tried, what was learned, idea for next.

The TSV *is* the digest. The user wakes up, reads the keep rows, picks winners.

## How to work

1. Write `scratch/<sprint>/PERF_GOAL.md` from [[perf-sprint-goal]] — sprint goal + equivalence gate values.
2. Initialize `scratch/<sprint>/results.tsv` with the header row.
3. Register the [[perf-sprint-loop]] message verbatim.
4. Read [[perf-sprint-levers]] for ideas to seed the loop. Read [[perf-sprint-traps]] when something crashes.
5. Append a post-mortem to [[perf-sprint-history]] when the sprint ends.

## Scope

Calibrated for Burl-style mlx-lm inference on Apple Silicon (Gemma 4 E2B, M5 Max). The lever ladder, trap recipes, and `decode_tok_s` thresholds assume that workload. Other perf sprints can borrow the contract and the loop discipline; fork the levers and traps.

## Links

[[perf-sprint-goal]] [[perf-sprint-loop]] [[perf-sprint-levers]] [[perf-sprint-traps]] [[perf-sprint-history]] [[perf-on-the-table]]
