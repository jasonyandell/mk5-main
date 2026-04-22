# Gus — Evening Status 1 (2026-04-21 ~20:15 CDT)

## TL;DR

The 10k-games corpus gen (started previous morning) completed late
afternoon. First attempt to train on it instantly OOM'd the M5 Max
because `JointWorldFullDataset` eagerly loaded 110 GB. That forced the
infrastructure fix we'd been planning anyway: a new lazy
`JointWorldFullIterable` class that streams chunks through a shuffle
buffer. Memory is now bounded at ~3 GB regardless of corpus size.

With the new dataset, v3 consistency trained on the full 10k corpus:

- **v3-10k regret: 0.551 Q-points** (down from 1.346 at 3k — **−59%**)
- **v3-10k bot-match: 76.07%** (up from 66.4% at 3k)
- **Near-ties: 85.5%** (up from 77.5%)

Regret is below 1.0 for the first time in gus. The v2-10k-big baseline
is training now (to disambiguate "data scaling" vs "consistency loss");
results expected ~22:00 CDT.

Secondary line of work: we traced how the oracle actually picks its
"best move" and landed a clean utility/dynamics decomposition that
preserves the cliff-shaped "mark is mark" objective while acknowledging
that margin-as-safety-buffer belongs in LAMIR rollouts, not in the
single-decision utility. Pre-Vast-launch fix list captured in
`GEN_FLEET.md`.

## Adapter ladder — updated

Mean regret on held-out 560 decisions. Lower = better.

| adapter | bot-match | mean regret | near-ties |
|---|---|---|---|
| v1_full (100g, d=128/3L, 0.4M) | 59.3% | 2.48 | 68.9% |
| v1_full (1000g, d=192/4L, 1.2M) | 65.4% | 2.16 | 73.4% |
| v2_voids_big (1000g, 3.4M) | 62.7% | 2.10 | 70.5% |
| v2_voids_big (2000g, 3.4M) | 67.3% | 1.60 | 75.4% |
| v2_voids_big (3000g, 3.4M) | 67.9% | 1.39 | 77.3% |
| v3_consistency_3000g (3k, 3.4M) | 66.4% | 1.346 | 77.5% |
| **v2_voids_10000g_big (10k, 3.4M)** | 73.21% | 0.818 | 81.8% |
| **v3_consistency_10000g (10k, 3.4M)** | **76.07%** | **0.551** | **85.5%** |

*Bot-match chance ≈ 25% (4 legal moves avg). Q range is [-42, +42].*

## The OOM → lazy loading story

**Setback**: v3-after-10k watchdog kicked off training at 15:41 CDT.
Python got `Killed: 9` (macOS OOM SIGKILL) before the first forward
pass. `JointWorldFullDataset.__init__` held all 100 chunks' games in
memory — 110 GB of unified memory, which the OS wouldn't tolerate
alongside the Python runtime. `set -euo pipefail` propagated the
failure; the watchdog died too. v2-10k-big retrain never fired.

**Fix (the real one)**: `JointWorldFullIterable(IterableDataset)` in
`gus/model/dataset_seq_world.py`.

- Streams chunks in random order; each chunk loaded, items shuffled
  within, buffered and yielded, chunk released
- Shuffle buffer of 8192 items mixes across trailing ~1-2 chunks
- Length scan at init is ~0.5s per chunk one-time; cached to disk via
  `--length-cache` so subsequent runs skip it
- Training scripts gained `--lazy`, `--buffer-size`, `--length-cache`
  (additive, backward compatible)

**Smoke-test results** (5 chunks × 2 epochs, CPU):
- Peak RSS: **3.4 GB**, flat across epochs (no growth leaks)
- Throughput: ~4500 items/s (CPU-bound on item-building)

**Ran full 10k v3 training on real GPU** (MPS):
- RSS ceiling: 12 GB peak (vs 110 GB OOM with eager)
- Wall: ~1h 51m for 60 epochs (110s/epoch, 4.6× 3k's 24s/epoch —
  consistent with 10k/3k decision count ratio + chunk-load overhead)
- No instability, no memory drift

This unblocks the Vast-fleet plan (100k+ games, multi-TB corpora)
without further infrastructure surgery. Shipped as commit f138069.

**Tradeoffs accepted**: approximate shuffle (chunk+buffer rather than
global permutation — WebDataset uses the same pattern at scale), no
multi-worker support yet, ~2 min one-time length scan per new corpus.
Captured as PRACTICALITIES.md §16.

## v3-consistency at 10k — the headline result

| epoch | train pi | eval pi | eval belief | eval qMAE | best score |
|:-:|:-:|:-:|:-:|:-:|:-:|
|  7 | 61.8% | 62.7% | 38.9% | 10.61 | 0.2805 |
| 14 | 63.8% | 66.6% | 39.2% |  9.98 | 0.3177 |
| 22 | 65.5% | 66.4% | 39.3% |  9.40 | 0.3240 |
| 31 | 67.0% | 69.6% | 38.8% |  8.75 | 0.3594 |
| 40 | 68.1% | 71.9% | 38.5% |  8.69 | 0.3734 |
| 56 | 69.7% | 74.1% | 39.2% |  8.29 | **0.3965** |
| 60 | 70.0% | 72.7% | 39.6% |  8.25 | (held best at 56) |

Best was epoch 56. Final regret eval on CPU:

```
=== Summary over 560 decisions ===
  Bot-match rate:           76.071%
  Mean regret (Q-points):   0.551
  Decisions with regret<0.5: 479/560 = 85.5% (near-ties)
```

Near-perfect on end-game (decisions 24-27: all 100% bot-match, 0
regret). Mid-hand decisions (10-20) still carry most of the regret
mass, consistent with MORNING3 analysis.

**The single biggest correlate**: data scaling. v2_voids_big at 3k
had regret 1.39; v3_consistency at 10k has 0.55. Over 2.5× improvement.
How much of this is consistency loss vs data alone? — that's
v2-10k-big's job to tell us.

## Oracle bias forensics

Traced `forge/eq/generate/actions.py` to answer "how does the oracle
actually pick its best move?" The user's prior was that the oracle
had been patched to prefer minimum-make over maximum-margin.
Confirmed — and sharpened.

Key finding: the oracle maximizes **p_make** (probability of making the
contract) with a 1e-6 E[Q] tie-breaker that's below float32 noise for
real p_make gaps. So oracle is effectively pure p_make argmax.

This is the **correct utility** for 42 scoring (mark is mark, margin is
decorative). Confirmed with the user: they want the cliff preserved
("29 is never 30") and they don't want to bias toward bigger margins
above threshold — a mark is a mark.

**Two real issues surfaced**:

1. **Hardcoded bid=30 threshold** in `select_actions`:
   ```python
   p_make_offense = e_q_pdf[:, :, 60:].sum(dim=2)  # P(Q >= 18)
   p_make_defense = e_q_pdf[:, :, 25:].sum(dim=2)  # P(Q >= -17)
   ```
   Bin offsets 60 / 25 correspond to bid=30. `decl_id` encodes trump
   only, not bid amount. For higher bids the actual threshold is
   stricter. Teacher is systematically overconfident on >30 bids.
   **Pre-Vast-launch fix.**

2. **Utility vs dynamics** was briefly confused on my end. Proposed
   `U = E[Q] + C·p_make` as a "cliff + margin" formula. User correctly
   pushed back with a concrete counterexample:
   - A: p_make=1e-8, E[Q]=-42 (sliver of hope, otherwise blowout)
   - B: p_make=0, E[Q]=+16 (guaranteed minor loss)
   - `U = E[Q] + C·p_make` picks B — opposite of user's preference.

   Resolution: user's utility is literally `U = p_make`. Argmax handles
   indifference in ties (above and below threshold) correctly. The
   "going for it wins more" intuition is dynamics — margin at trick 3
   compounding into safety at trick 7 — and that belongs in LAMIR,
   not in the utility shape.

Both captured in PRACTICALITIES.md §17 and pre-Vast-launch fix list in
GEN_FLEET.md (commit a14200f).

## What's running / staged / open

**Running**: `scratch/train_full_10k_lazy.sh` — v3-10k training done,
regret eval done, v2-10k-big training just started (~2h expected).
PID 7556 at launch, may have rotated into a new PID for v2-big.

**Not running, waiting on decisions**:
- LAMIR-1 prototype. User's instinct (and mine): rotation-equivariance
  lets us query `π_me` from rotated views to serve as π_opp for free.
  No π_opp head needed. Cheap proof-of-concept on the CURRENT corpus.
  Decision point: start after v2-10k-big lands.
- Schema v2 re-gen (per-seat oracle softmax + bid_value). Needed
  before the Vast fleet runs on the diverse-seed corpus. Order:
  LAMIR-1 first (on current schema), then schema v2 (once LAMIR-1's
  empirical win motivates the data re-gen).

**Not pursued today**:
- MPS multi-world variance regularization of Q_head (receipt 15's
  follow-up). Waiting for the 10k baseline to settle before
  re-hypothesizing.
- Filing beads for the pre-launch fixes. `bd` is currently broken in
  this environment; captured as GEN_FLEET.md sections instead.

## Consistency-loss verdict — v3 clearly wins, and the gap widens with data

Final v3-10k-vs-v2-10k-big head-to-head on 560 held-out decisions:

|              | v2-big-3k | v3-cons-3k | v2-big-10k | **v3-cons-10k** |
|--------------|----------:|-----------:|-----------:|----------------:|
| bot-match    |    67.86% |     66.43% |     73.21% |      **76.07%** |
| mean regret  |    1.391  |      1.346 |     0.818  |       **0.551** |
| near-ties    |    77.3%  |      77.5% |     81.8%  |       **85.5%** |

**Regret decomposition:**
- Pure data scaling (v2 3k → 10k): **−41% regret** (1.391 → 0.818)
- Consistency loss at 10k (v2-10k → v3-10k): **additional −33% regret** (0.818 → 0.551)
- Stacked: v2-3k → v3-10k = **−60% regret** (1.391 → 0.551)

At 3k games, v3 vs v2 was roughly a wash (1.346 vs 1.391 — could have been
noise). **At 10k, v3 pulls decisively ahead** (0.551 vs 0.818). The
consistency regularizer scales BETTER than plain distillation — probably
because V_head becomes a more trusted anchor as data grows, so the
"force π to pick actions V thinks are good" loss has a stronger target.

**Decision: consistency loss rides forward** into LAMIR-1 and schema v2.

## Probe findings (tonight's exploration)

Three probes on v3-10k captured in `scratch/probe_*.{py,log}`:

- **Domino embedding similarity** — doubles cluster (+0.047 vs non-double
  −0.022), counts cluster (+0.037 vs non-count −0.021), high-pip families
  tighter than low-pip. Structural learning confirmed; relational
  concepts ("6-6 protects 6-4") are NOT in the raw embedding — they're
  contextual.
- **Attention patterns** on game 0 decision 0: CLS attention evolves
  across layers (DECL-anchored → scan MINE → focus on eventual choice).
  Final layer concentrates on the chosen action (0.26 on MINE[0] where
  π_me = 0.91).
- **Counterfactual hand swaps** (leave-one-out, consistent swap variant):
  - Swapping any domino → 0-0 (top trump in blanks) jumps V by +11 to
    +19 — model understands trump structure
  - Swapping 1-1 → 6-6 (user's intuition-check) DROPS V by −5.3 —
    suspicious. Investigation: V_head doesn't use world_assignment so
    the swap is valid input. The model's claim is real. Most likely
    explanation: **strategy-fusion leakage from PIMC training** —
    partner-trumps-partner in the oracle's per-seat-independent rollouts
    over-penalizes non-trump doubles that partner might trump. LAMIR's
    shared-π fixes this exact bug.

**Implication**: the probes confirm Gus has learned structural game
features but also inherits the oracle's strategic biases. LAMIR is the
escape from those biases, not just a performance optimization.

## Next session

1. **Write a PRACTICALITIES receipt on the probes** — the strategy-
   fusion leakage showing up in counterfactual V is worth preserving.
2. **LAMIR-1 prototype.** With consistency loss validated as a win,
   v3_consistency_10000g is the baseline. Use rotation-equivariance to
   query π_me on rotated views for opp seats (no π_opp head needed).
   Start with 1-ply look-ahead (roll to end-of-trick), V_head at leaf.
3. **Schema v2** only after LAMIR-1 shows life. Per-seat softmax,
   bid_value plumbing, all the fix list in GEN_FLEET.md.

All deferred to tomorrow; no new big runs tonight per user direction.
