---
title: "Gus lazy continuations and success-reward integration smoke"
kind: experiment
first_seen: 2026-10-18
last_updated: 2026-10-18
status: complete
---
<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

## Question and result

Can the frozen panel grow lazily without changing existing evidence, and can
the existing trainer consume verified success probabilities with rehearsal?
The bounded integration smoke passed. This is engineering evidence, not a
method comparison or a playing-strength claim. The scientific next step is
[[gus-success-probability-planning]].

Source: [`ba31124a`](https://github.com/jasonyandell/mk5-main/tree/ba31124a87d3df07234996aefad672cefa30e751).
Generated evidence: [immutable archive](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/b0f2fbd1f0e6a112ebd5e8c575aa0784cca0ee10/lem/gus-lazy-rlvr-smoke-2026-10-18).
The archive contains commands, row selection, every continuation/recipe/snapshot,
reconstruction copies, training formation checkpoints and receipts. It references
the original immutable input archives rather than duplicating them.

## Frozen replay proof

Two unsettled penultimate-trick positions from [[gus-frozen-benchmark-2026-10-18]]
received three sampled hidden deals each. The pinned Zeb viewer baseline and
previously selected [[gus-retention-iterations-2026-10-18]] candidate used
identical root draws. Only the viewer changed; the other seats preserved their
original anchor G0/Zeb policies and all budgets. Every completed hand passed
independent legality, scoring and terminal-reward replay.

- Extending one to three samples preserved the initial records and snapshot.
- Reconstructing with reversed row order and disjoint sample-index slices
  produced identical bytes for all six baseline records.
- A stop-file invocation committed no incomplete unit; resuming reproduced the
  same six records byte for byte.
- Candidate and baseline recipes differ only in learner identity; each paired
  root shares seed, hidden deal, auction and seating.

This is a same-runtime replay check. It does not establish cross-build numerical
parity. Six correlated samples on two positions cannot establish playing
strength. The full outcomes are retained, without promoting their win count
into a scientific finding.

## Trainer and checks

An eight-round, two-updates-per-round run used `--mode exact --reward success`,
the original 185,652-parameter splat parent, original exact curriculum,
fixed-parent KL and supervised rehearsal. It completed and remained within the
existing retention bounds. No method or checkpoint was selected for promotion;
no fresh confirmation was read. Initialization, rounds four/eight, best, final
and complete optimizer/sampler/RNG state are archived.

The repository gate passed all 116 Lem tests plus wiki, license and conflict
checks. New tests cover probability-gap gradients, action-independent baseline
invariance, zero-gradient ties, rejection of unsuitable loss combinations,
actual SIGTERM/resume equality with probability reward and rehearsal, immutable
recomputation, corrupt evidence, recipe drift and concurrent duplicate work.
Operational instructions are in the `gus-training` skill, including the
`clean-neuron` handoff boundary.
