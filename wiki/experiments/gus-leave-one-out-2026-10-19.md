---
title: Gus leave-one-out outcome baseline
kind: experiment
first_seen: 2026-10-19
last_updated: 2026-10-19
status: active
---
<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

## Question and motivation

Prepared comparison for [[gus-tiny-gpt]]: does a leave-one-out success baseline
improve learning over the current within-root standardized GRPO advantages?
Public registration must be verified before training. No result exists yet.
The independently tested sampled-policy lead failed in
[[gus-sampled-confirmation-2026-10-19]]. The earlier online critic did not
forecast new batches well ([[gus-ppo-value-2026-10-19]]). This factor tests a
parameter-free estimator without repairing a critic or increasing exposure.

For eight independently sampled complete hands at a fixed physical deal,
let R_i be verified team success (0 or 1). The new advantage is
A_i = R_i - sum(R_j for j != i)/7. GRPO instead centers and divides by the
within-group standard deviation. Both give zero when all eight outcomes agree.
The new baseline cannot discover a successful continuation absent from those
samples and is not direct move-level credit assignment.

The baseline excludes its own trajectory. Conditional on a fixed root and fixed
other policies, the other independent draws are independent of that trajectory's
score-function term, so subtracting their average preserves the on-policy
expected success gradient. The learner receives only its original own-seven
and public history; conditioning a training baseline on the physical root does
not expose other hands as actor inputs. All seven learner decisions share the
hand advantage; their log-probability gradients sum to its trajectory score.
The implementation averages seven decisions, a common fixed scale in both arms.

This justification follows the estimator in
[Ahmadian et al., section 2.3](https://arxiv.org/abs/2402.14740), whose empirical
results concern language-model preference learning, not 42. The experiment
retains per-decision probability-ratio clipping and two inner updates. It tests
a **leave-one-out advantage within the existing clipped actor**, not a complete
reproduction of that paper. Unbiasedness at the sampling policy is not a claim
about later clipped updates, AdamW or finite-run success.

## Fixed protocol

Three matched 32-round arms: leave-one-out, unanchored GRPO, and rehearsal-only.
The two reward arms have no imitation or frozen-reference KL loss. Rehearsal
remains an explicit control. All select by development greedy success, then
sampled success, earliest ties; initialization is eligible. Imitation metrics
are diagnostics, not vetoes.

Same 185652 parameters, initialization, original inputs, fixed G0 partner,
pinned Zeb opponents, bid30/pip, seed42, 256 fresh training roots, eight complete
hands per root, 64 updates, learning rate3e-5, clip.2, AdamW matrix decay.1,
gradient clip1, and full seven-turn credit. No search inside the player.
The change includes the natural advantage scale: raw 0/1 leave-one-out returns
versus standardized returns, with no rescaling or learning-rate sweep. It does
not isolate baseline exclusion from normalization and scale effects.

128 development deals, evaluation every eight rounds; freeze all selections
before 512 untouched confirmation deals. Primary endpoint remains greedy team
success. Promotion requires leave-one-out to beat initial parent by at least
2 percentage points and both controls by at least1 point, with each paired95
lower bound above zero. Development method criterion is at least2/128 above
both controls. Sampled play is secondary; no confirmation reselection.
Bootstrap 20000 physical-deal draws, seed91942. All newly reserved roots and
all512 previous confirmation roots become future training exclusions.

Each arm has at most16 segments, 45 seconds cooperative and90 seconds outer,
at most two live arms. Confirmation allows12 such segments per unique state,
at most two live states. These are engineering caps, not player think budgets.
Audit, native capture and integration budgets are frozen in the registration.
Failures and exhausted caps require inspection, never automatic budget growth.
Recovery root: `scratch/lem/gus-rloo-v1`.

## Capabilities and verification

The19:00UTC refresh found no new Lem#207 or Book#121 replies; SilasPR166 remains
open. Existing replay verifiers suffice. [[gus-book-silas-tools]] separates these
available tools from the still-missing transformer causal adapter. No repeat
request or opponent/input change follows from the refresh.

Trainer v6 adds the explicit `rloo` mode. Exact enumeration tests check the
expected Bernoulli success gradient against its analytic derivative. Matched
collection checks retain identical episodes and independently reconstruct
advantages. Actual SIGTERM and simulated mid-update crash tests compare resumed
CPU model/optimizer/RNG state with uninterrupted training. MPS has no byte-level
resume promise. Full formation and native captures remain available to
`clean-neuron`; prediction drift is not improved forecast calibration.
