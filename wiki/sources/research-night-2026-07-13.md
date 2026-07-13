---
title: Source — research night 2026-07-13
kind: source
first_seen: 2026-07-13
last_updated: 2026-07-13
status: active
---

One autonomous overnight session on branch `worktree-research-night-2026-07-13`
(b28fb55a → close), executing the continuation-frontier research adopted at
[[research-lane-selection]]. Every experiment registered predictions before
its runs; every result landed the same night. Commit trail is the branch
history; curated evidence under `champion/evidence/{stage0_2026-07-13,lane_b_target_granularity}/`
and `w42/world_sampler_audit/`; raw runs and corpora are ephemeral per
[[run-artifacts-policy]].

## What the night established

1. **Stage 0 closed** ([[stage-0-closure]]): C0 reproduces on the repaired
   sampler on both reserved blocks (`+0.385`/`+0.486`); symmetry clean;
   `judsearch(r4)` deficit reproduces; CUDA correctness passes with the
   sampler revealed as kernel-launch bound (throughput prediction MISS,
   recorded); legacy exposure quantified (2.51% distributional; 20/200-worst
   argmax flips, regret ≤ `7.37 Q`; not load-bearing). Repaired sampler costs
   ~2× C0 wall time on MPS.
2. **Lane A instrument validated** ([[auction-decoder-v0]]): bid decoding
   shows the clean causal signature (hand features help exactly and only for
   hand-dependent bidder populations); population-conditioning +0.24 nats;
   book fixtures untestable on the bid-thin corpus — enriched-bid corpus is
   the named next step.
3. **Lane B graded — a comprehensive negative** ([[jud-target-granularity]],
   two rounds): per-move targets at v1 capacity are marks-null in both the
   parent-side dense-aux and child-state value forms; 3× volume moves
   calibration only; ranking-label agreement does not order play strength;
   the residual narrows to capacity interaction, on-policy loop data, and
   opponents-in-rollout. Five of six round-1 predictions and three of four
   round-2 predictions missed — all registered before their runs.
4. **New durable capabilities**: `--teacher-forced` E[Q] labeling (decision k
   = recorded play step k; 100% label-join coverage, proven), `judauxplay`
   diagnostic consumer, `bid_decoder` scaffold, CUDA sampler bench, exposure
   scan machinery, verified literature map ([[search-literature-transfer]]).

## Corrections carried

The [[belief-weighted-jud-mcts]] backup semantics (no partner-max/opponent-min
outside the two legal forms), the [[dense-q-supervision]] boundary, the
superseded no-argmax-flip bounded observation, and the narrowed jud-v2 open
question.

## Links

[[research-lane-selection]] [[stage-0-closure]] [[auction-decoder-v0]]
[[jud-target-granularity]] [[search-literature-transfer]] [[auction-decoder]]
[[partnership-wall-research]] [[the-wall]] [[jud]] [[champion]]
[[consumption-ledger]] [[world-sampler-mrv-audit]]
