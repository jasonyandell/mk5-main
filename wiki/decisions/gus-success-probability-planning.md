---
title: "Gus: success-probability RLVR and lazy frozen evaluation"
kind: decision
first_seen: 2026-10-18
last_updated: 2026-10-18
status: active
---
<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

## Selected direction

The next step for [[gus-tiny-gpt]] is textbook reward learning on **verified
hand success**, with the existing architecture, clean inputs and other seats
fixed. Jason explicitly selected this direction and lazy frozen-set expansion
on 2026-10-18. External search and future-splat auxiliary training are deferred.
[Issue #220](https://github.com/jasonyandell/mk5-main/issues/220) records the completed
matched probability-reward experiment in [[gus-probability-reward-v3-2026-10-18]].
[Issue #222](https://github.com/jasonyandell/mk5-main/issues/222) continues the
broader autonomous goal through full-hand terminal-outcome training.

This is post-training: the pretrained policy is improved using a verifier.
The objective is making or setting the bid30 hand, not unspecified match win
probability. Exactness is conditional on the named finite game model. The
three-style training field and the G0/Zeb evaluation field remain distinct and
fixed within their respective comparisons.

## Textbook choice for the current subproblem

[[gus-rlvr-silas-2026-10-18]] supplies an exact rational value `Q(I,a)` for
**every** legal root action: uniform legal-consistent hidden deals and the
fixed lowest/highest/uniform continuation mixture. The viewer has two tiles
left; its final move is forced. This is a one-decision contextual bandit.
There is no hidden-world maximization or new search inside Gus.

`lem.observer.rlvr.train --mode exact --reward success` now minimizes:

`L = -mean_I sum_a pi_theta(a|I) Q(I,a)`
`    + beta KL(pi_theta || frozen_parent) + lambda L_original_tasks`.

The first term directly differentiates finite expected return, the standard
policy-gradient objective. Enumerating two known rewards avoids unnecessary
action-sampling variance and a learned critic. The remaining terms are the
existing fixed-parent KL and supervised rehearsal, justified by the forgetting
observed in [[gus-retention-iterations-2026-10-18]]. Teacher splats are rehearsal
targets, not the verifier. Rational values become float32 only at training.
Source: [policy-gradient formulation](https://papers.neurips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html).

For two action logits, the reward-loss derivative is
`dL/dz_a = -pi_a (Q_a - sum_b pi_b Q_b)` before batch averaging.
At equal probabilities, an 80% versus 40% decision has forty times the reward
gradient of an 80% versus 79% decision. Ties have zero reward gradient. Adding
an action-independent baseline changes the loss value, not this gradient.

The existing `--reward best` remains the explicit binary-correctness control.
GRPO standardizes within a question; with two distinct reward levels this
removes the magnitude of the gap. Merely substituting probabilities into that
standardized surrogate does not test this hypothesis. Success rewards therefore
require exact mode. GRPO remains available for the original experiment; it is
not renamed to cover a different objective. Source: [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300).

## Completed bounded comparison and next stage

[[gus-probability-reward-v3-2026-10-18]] completed the matched three-seed
comparison from the original splat parent, keeping architecture, question/replay
streams, update budget and KL/rehearsal coefficients fixed. Probability reward
did not pass the registered improvement criterion. Both predeclared seed42
representatives improved fresh finite-model regret with retention; brief game
checks did not establish transfer. The experiment page owns the measurements.

The next stage broadens training to full-hand sampled trajectories and verified
terminal outcomes, with a matched rehearsal-only control. One learner seat,
fixed G0 partner and pinned Zeb opponents define the declared full-game stage.
Its actual outcome verifier remains distinct from the exact three-style
endgame approximation. Grouped samples justify a GRPO estimator when complete
continuation values are no longer cheaply available. Register that protocol
before training; do not silently substitute it for the completed experiment.

The frozen panel remains development evidence. Reserve new confirmation before
selection and extend actual continuations only where needed. Existing tests,
model sizes and qualitative hypotheses are not grounds for a general-player
promotion. The pinned Demon is a separate evaluation competitor; its internal
forecasts never become learning targets.

## Lazy immutable continuation evidence

`lem.observer.benchmark.continue_rows` appends supplements beside
[[gus-frozen-benchmark-2026-10-18]], leaving its manifest, rows and assets intact.
The initial supported scope is unsettled, two-legal-action penultimate-trick
rows in the two G0/Zeb anchor pairs. Silas enumerates the small legal root fiber;
a named seed samples a hidden deal uniformly. This law deliberately excludes
pre-root behavior likelihood and is separate from Zeb's internal belief model.

The baseline puts pinned Zeb in the viewer seat. A candidate replaces that
seat alone. Each of the other three seats keeps its original anchor G0 or Zeb
policy, including the partner. Their actions are recomputed after changed
history. Original dealer, bids, weights, deterministic planner budgets and
seating are retained. The network still receives only its own original seven
and public contract/history. Full hands and outcomes are audit sidecars.

Each unit is keyed by recipe, frozen row ID, question hash and sample index.
The root seed is derived from the common pairing recipe, row/question, sample
index and named stream; candidate weights are excluded from that seed. It is
independent of request order, process restarts and work slicing. Candidate
weights remain part of the separate result identity. Complete trajectories and
terminal success are checked by an independent rules replay before commit.

The recipe pins base manifest/runtime hashes, policy and weight identities,
additional source files, native binary bytes, platform, Torch build and CPU
float32/single-example execution. Seeds alone do not guarantee replay across
software or hardware changes; a changed runtime uses a new recipe and parity
check. [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html).

Writes are atomic under per-unit locks. Failed/incomplete units have no completed
record; a later invocation recomputes them. Busy units are left for a subsequent
invocation. Existing valid records are skipped; `--recheck` recomputes them and
rejects any changed bytes. Corrupt records and recipe drift fail closed. Higher
`--samples`, added row IDs or independent `--start` slices append work. Sorted,
content-addressed snapshots preserve every earlier published view. Timings and
queue state are excluded from evidence bytes. Stop signals, a stop file and a
cooperative wall budget stop between complete hands; no timeout becomes a loss.

These are sampled actual-policy outcomes, not exact Zeb success probabilities.
Compare paired candidates at the same row/sample keys and report sample counts;
cluster broader uncertainty by original physical deal or complete anchor pair.
Retain sampled full-hand identities as future training exclusions alongside the
base panel's reserved deals. Add new provider scopes under explicit identities;
never rewrite existing evidence to make a larger panel look untouched.

The operational commands live in the `gus-training` skill's
`references/benchmark.md`; representation and unit claims follow `clean-neuron`.

Implementation and same-runtime replay evidence: [[gus-lazy-rlvr-smoke-2026-10-18]].
The matched multi-seed comparison is complete in [[gus-probability-reward-v3-2026-10-18]];
full-hand outcome post-training is next, with a matched rehearsal-only control
and the same clean inputs. Existing exact labels and measured terminal game
outcomes retain their distinct authority.
