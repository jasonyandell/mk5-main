---
title: Light every core — cpu-slot by busy cores against the fixed three slots (#173, 2026-10-16)
kind: experiment
first_seen: 2026-10-16
last_updated: 2026-10-16
status: active
---

## Summary

Issue #173. `scripts/cpu-slot` let at most three runs share the 18-core Mac, because Walt thinks on
a wall clock and a crowded machine changes its moves. Every other player the arena seats is
deterministic in work units, so for them the cap only leaves cores idle. The new cpu-slot admits a
job when the cores busy now, plus the job's threads, fit in the core count
([[standing-desks]] § CPU). This page pre-registers a before/after on one fixed queue, and the
measurements behind the new default.

*Results follow once both arms have run.*

## What the jobs use (measured before the change)

Mean cores = the job's CPU seconds (user + sys, children included) over its wall seconds, on this
machine, through the old cpu-slot, 2026-10-16.

| Job | Mean cores | Peak (1 s) | Source |
|---|---:|---:|---|
| Arena batch, 1 game pair (seven study arms: count_lapse B0 Z20, no_plan K0 Z16, fallible Z40, crisis_reprice R, overbidders RS) | 0.96–1.09 | ≤ 1.6 | `measure.py` |
| Arena batch, 2 pairs (count_lapse B20) | 1.47 | 2.1 | `measure.py` |
| Arena batch, 12 pairs (the studies' registered size: count_lapse B0, Z20) | 4.85, 4.72 | 7.6, 6.8 | `measure.py` |
| `tournament work` (a live ladder chunk, Walt + Demon seats) | 2.09 | 7.0 (48 threads) | sampled holder tree |
| `lem.probe concepts` | 1.00 | 1.0 | sampled holder tree |
| A Zeb or Demon seat (native, one game) | ≈ 1 | | [[rematch-zeb-041-vs-demon-2026-10-15]]: CPU/wall 5,640/5,855 and 3,732/3,854 |

The default `--threads 2` is the median of those kinds rounded up. A Book-desk queue tells
cpu-slot its batch's measured cores (`scenario/book_batches.py`).

## Who thinks on a clock

- **Walt (`walt-plunge-26043bf9`).** Clocked. The pinned wasm stops each evaluation stage at a
  deadline read through the host's `now_us`; the partner-rollout review takes up to 64 samples in
  at most 500 ms; the live auction survey keeps the last round of 4/12/40/160 worlds finished
  inside 20 s. Its participant config says `"clock": "real"`.
- **Zeb** (npm 0.1.0/0.2.0, native 0.4.0/0.4.1): 160 worlds, seed 0. Deterministic.
- **The Demon** (every pin): fixed worlds, seeded by game, hand, seat and ply; reports `ms`, never
  reads a clock to decide. Deterministic.
- **E[Q] n=10** (fenced): ten samples, no clock.
- **The exploiter** (`scenario/exploiter.py`, a study instrument, not a seat): its best responses and
  coordinator gave up at wall-second caps. It now also takes a cap in search nodes.

## Pre-registered before/after

**Queue.** 60 arena batches, `python -u -m scenario.count_lapse arena --arm A --seed S --games 1`,
arms B0, Z20, B20 × seeds 3101–3120, offered by 18 queue workers (the shape of
`book_batches.queue`), each worker running its next batch through the arm's cpu-slot.

**Arms.** *Before*: origin/main `082dcf33`'s cpu-slot (sha256 `dc1127e6…`), three fixed slots.
*After*: this branch's cpu-slot, the load mode, `--threads 1` (a 1-pair batch measured 1.0
cores). Each arm has its own temporary slot directory; both run inside one real
`scripts/cpu-slot --exclusive` call (300 s cap), so no other session's run starts meanwhile, and
the before arm runs first.

**Measured.** Wall seconds to drain the queue; the machine's mean busy cores over the drain,
sampled every second from the Mach host CPU ticks (the quantity `top` reports); total batch CPU
seconds; the sha256 of every batch's `games.jsonl`.

**Pass rule.** The 60 `games.jsonl` files are byte-identical between the arms (any difference
fails: a move changed). Prediction: before about 60 × 10 s / 3 ≈ 200 s, after under 60 s, mean busy
cores well above before's. If the before arm cannot drain in 300 s, its rate (batches per second)
is reported instead. The runner is `scratch/t173/drain.py`, sha256 `45650dfd…`.
