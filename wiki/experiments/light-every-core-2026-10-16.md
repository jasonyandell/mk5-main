---
title: Light every core — cpu-slot by busy cores against the fixed three slots (#173, 2026-10-16)
kind: experiment
first_seen: 2026-10-16
last_updated: 2026-10-16
status: complete
---

## Summary

Issue #173. `scripts/cpu-slot` let at most three runs share the 18-core Mac, because Walt thinks on
a wall clock and a crowded machine changes its moves. Every other player the arena seats is
deterministic in work units, so for them the cap only leaves cores idle. The new cpu-slot admits a
job when the cores busy now, plus the job's threads, fit in the core count
([[standing-desks]] § CPU). This page pre-registered a before/after on one fixed queue, and the
measurements behind the new default.

**Result.** The same 60 batches drained in 65.9 s instead of 196.1 s (3.0× faster), with the
machine's busy cores at 14.1 of 18 on average instead of 3.9, and all 60 `games.jsonl` files
byte-identical between the arms: no move changed. Instructions retired were the same to 0.03%
while CPU seconds rose 36%, so instructions are the load-proof cost to quote.

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

## Results

Both arms ran 2026-10-16 inside `cpu-slot --exclusive`, before then after, every batch exit 0.

| | Before (3 fixed slots) | After (busy cores, `--threads 1`) |
|---|---:|---:|
| Wall to drain 60 batches | 196.1 s | **65.9 s** (0.34×) |
| Mean busy cores, of 18 (1 s samples) | 3.94 | **14.12** (peaks 17.9) |
| Batch CPU seconds, total | 624.7 | 847.1 (+36%) |
| Batch cycles, total | 2.641 × 10¹² | 3.267 × 10¹² (+24%) |
| Batch instructions retired, total | 1.2509 × 10¹³ | 1.2513 × 10¹³ (+0.03%) |
| Mean batch wall | 9.0 s | 13.1 s |
| `games.jsonl`, sha256 over the 60 | `1989718e…` | `1989718e…` (60/60 identical) |

**Pass.** The outputs are byte-identical, as the pass rule required.

**Reading.** The queue is three times faster, not 4.6× (14.1 / 3.9 busy cores): a batch costs more
CPU when the machine is full. The likely reasons (not separated here): with six performance and
twelve efficiency cores, a full machine runs most batches on efficiency cores, and every batch
shares caches and memory bandwidth. The
work itself did not change: instructions agree to 0.03%, where CPU seconds moved 36% and
cycles 24%. Wall time and CPU time both depend on the load; instructions retired do not. The idle
cores left in the after arm are the ramp (the first second), the waves (18 batches that start
together end together, and the 1 s measuring window plus the 3 s young-job allowance lag the
refill), and the tail (the last batches run alone).

**Exploratory, not registered.** A 1 s young-job allowance in place of 3 s drained the same queue
in 64.5 s at 14.9 busy cores, identical outputs again. One run each cannot tell a 2% difference
from noise, and a job that ramps slowly (a tournament worker spawning its seats) would be
over-admitted with it, so the default stays 3 s.

**Limits.** One queue of light (1-core) batches, one machine, one run per arm; the order was
fixed (before first). Other sessions' unslotted work ran in both arms. Not measured: a queue of
12-pair batches (about 5 cores each) should gain less, since three of them already kept about 15
cores busy under the old cap; it hurt most where jobs are light (single-threaded seats, probes,
1-pair batches).
