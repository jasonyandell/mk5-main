---
title: Gus leave-one-out outcome baseline
kind: experiment
first_seen: 2026-10-19
last_updated: 2026-10-19
status: complete
---
<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

## Question and motivation

Registered comparison for [[gus-tiny-gpt]]: does a leave-one-out success baseline
improve learning over the current within-root standardized GRPO advantages?
The registered comparison is complete, audited and publicly verified. Neither
registered criterion passed; no stronger player or method advantage was established.
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

## Publication and execution

[Immutable preregistration](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/b4e31df1a504a84fc043bd5e96cb1d749a8845d9/lem/gus-leave-one-out-2026-10-19/registration) contains all13 executable study scripts, plans, exclusions and thresholds:
22 files,11840313 bytes, each freshly downloaded anonymously and SHA256 verified.
Registration SHA256 `f691cd67313dfffc7baf07b09a949026d3eb6922053800b692c089fcb004d838`;
source `66af9b26`. All604707 prior physical deals are excluded.
The controller performs training, independent selection reconstruction, native
capture, fresh evaluation, replay audit and summary, then stops for publication
and interpretation. It does not select another study. Runtime gates:165 Lem tests,
changed-suite GREEN4. The original failed schema assertions and wiki checks were
repaired before merge and registration; no experiment outcomes existed then.

## Completed result

| Label | Selected round | Greedy wins /512 | Sampled wins /512 |
|---|---:|---:|---:|
| Parent | initialization |224|215|
| Leave-one-out |32|227|216|
| GRPO |32|228|217|
| Rehearsal |24|226|223|

Leave-one-out minus parent greedy: **+0.586 percentage points**, paired95
**[-1.367,+2.539]**. Versus GRPO: -0.195pp[-1.172,+0.586]; versus rehearsal:
+0.195pp[-1.953,+2.344]. Both development and fresh promotion criteria failed.
No endpoint switch, confirmation reselection or main model replacement follows.
The sampled leave-one-out difference from parent is +0.195pp[-1.367,+1.953].
Rehearsal has a secondary sampled-parent difference +1.563pp[+0.195,+3.125];
this is descriptive and does not establish reward learning or authorize selecting
that decoder/checkpoint from confirmation. Prior sampled-policy replication failed.

73.44% of leave-one-out training groups and72.66% of GRPO groups had the same
outcome in all eight draws, hence zero within-root contrast. This is evidence
about these sampled attempts, not proof that their positions have no better
continuation. Both arms had mean training success36.52%. Altering the estimator
at this exposure did not materially separate the resulting player outcomes;
more varied training trajectories is a candidate question, not a selected next
recipe or proven explanation. Moving individual action credit earlier in the
hand remains a distinct comparison requiring its own information-state design.

## Audit and publication

All6144 training and4096 confirmation hands replayed independently, with original
inputs, legal supports and verified outcomes. NumPy reconstructs each advantage;
43008 common-reference decision probabilities and all96 formation outputs checked.
Maximum CPU/MPS log-probability errors were3.15e-5 and3.53e-5, below the frozen
5e-5 tolerance. Sixteen lazy CPU units passed reverse-order byte checks;4096
paired training prefixes agree until learner divergence. All896 roots regenerated.
Three final native captures each contain11116 frozen rows. None is a unit-level
causal explanation. All32 rounds completed in8/16segments per arm, all four
confirmation states in5/12segments each. Every pipeline receipt exited0.

Selected and final inference exports for all three arms preserve exact tensors
and both-head CPU outputs. Full optimizer/RNG recovery and all formation snapshots
remain in the archive. Inputs, other seats and verifier authority stayed fixed.

[Verified research archive](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/bbad1223099d5b42cdb3fda844bc2f331f551b63/lem/gus-leave-one-out-2026-10-19): 16 files,
537732982 bytes, 4943 members; every fresh anonymous file and
bundle member passed SHA256 verification. [Canonical integration records](https://huggingface.co/datasets/jasonyandell/texas-42-tournament/tree/c19bd828351d463f03da0d263b366fda007562c9)
were delivered in one batch: 15 records. 9 bounded duplicate
faceoffs, 18 games, 195 hands and 5460 plays were replayed;
maximum faceoff wall time 5.25s, below60s. These are
integration checks, not a strength estimate or promotion evidence.

Research exports: [rloo](https://huggingface.co/jasonyandell/gus/tree/0dad0795969c6eef2939f69ab5e35f84a94de3d7), [grpo](https://huggingface.co/jasonyandell/gus/tree/5e1251ae692a16f8f4048eb9a6b716bd0899b92f), [rehearsal](https://huggingface.co/jasonyandell/gus/tree/aeea8b632da9941edbde857da4c9ae2b6a69d69b).

Completion capability refresh: Lem#207 and Book#121 still have no new replies;
SilasPR166 remains open. No repeat request. Existing verifiers sufficed, and
the transformer causal adapter remains separate. The next experiment is unselected
and must wait for the actual readout steering window in readout-window.json.
