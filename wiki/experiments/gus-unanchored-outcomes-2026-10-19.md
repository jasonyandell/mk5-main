---
title: "Gus: outcome learning without teacher anchoring"
kind: experiment
first_seen: 2026-10-19
last_updated: 2026-10-19
status: active
---
<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

## Question

Does removing the teacher-anchoring loss package improve verified full-hand
success over anchored GRPO and imitation alone? Jason's steering in
[[gus-success-probability-planning]] prioritizes winning and expects behavior to
drift after pretraining. [[gus-ppo-value-2026-10-19]] established neither stronger
play nor useful online critic forecasts. This comparison keeps the existing
GRPO estimator, separating the teacher-constraint question from critic repair.

## Registered comparison

All arms initialize the same 185,652-parameter [[gus-tiny-gpt]] splat actor.

| Arm | Verified-outcome gradient | Imitation weight | Frozen-parent KL weight |
|---|---|---:|---:|
| Unanchored | GRPO | 0 | 0 |
| Anchored | GRPO | 1 | .02 |
| Rehearsal | None | 1 | None |

The changed factor is a **package**, not the individual effect of either loss.
All arms select by greedy then sampled development success, including
initialization and keeping earliest ties. Original tile NLL and splat CE are
recorded without an eligibility veto. This common selection rule differs from
prior studies; confirmation cannot revise the selections.

Original own-seven/public-history tokens, bid30/pip, fixed G0 partner and pinned
Zeb opponents stay unchanged. Each arm receives the same 256 fresh physical
roots, eight complete 28-play draws per root, 32 rounds and 64 actor updates.
Learning rate3e-5, eight roots/batch, two inner updates, batch112, gradient clip1,
AdamW matrix decay.1, full seven-turn credit and evaluation every8 rounds remain.
Group-standardized terminal returns supply credit. No learned critic is fitted.

The clipped probability ratio uses the policy that generated the current batch;
it remains at .2 even when the frozen-parent penalty is zero. Clipping controls
the surrogate objective, not a guaranteed hard bound on policy change.
Sources: [PPO](https://arxiv.org/abs/1707.06347),
[GRPO](https://arxiv.org/abs/2402.03300). Rehearsal/fixed-parent penalties are the
experimental controls, not universal requirements of policy-gradient learning.

Zero-weight replay still consumes identical diagnostic draws under no-grad.
Unused splat-head parameters receive neither gradients nor AdamW decay. Their
outputs may change with the shared backbone; this does not train calibrated
current-Gus histograms. Hidden simulator state and outcome sidecars never enter
actor inputs. No player search, new opponent, or Demon label is introduced.

## Selection, confirmation and budget

The common 128-root development panel selects each arm. The registered method
criterion is unanchored greedy success at least 2/128 above both controls.
After all selections freeze, 512 untouched roots evaluate the parent and three
selected labels, greedy and one coupled sampled rollout. Identical exact
inference functions may share records under the pinned runtime and stream.

Promotion requires unanchored greedy gains of at least 2pp over parent and
1pp over both controls, each paired 95% lower bound above zero. Imitation losses
are not promotion gates. Intervals resample physical roots:20,000 draws,
seed91942. Sampled results are secondary; one seed/recipe does not rank methods
generally or establish unrestricted auction strength.

Each arm has at most16 cooperative45-second segments with90-second outer caps,
max2 live arms. Confirmation has at most12 such segments/state, max3 live.
Identity, nonfinite, replay and budget failures halt with recovery preserved.
No automatic budget increase. Initial/per-round/final checkpoints preserve
AdamW and RNG. CPU signal/crash recovery is tested; MPS bitwise identity is not
promised. Frozen CPU evaluation retains strict lazy-record byte checks.
Final native captures preserve the existing 11,116-row panel for `clean-neuron`;
unit explanations remain separate work. Registered fixed G0/Zeb/fenced-Demon
smokes are one duplicate pair per distinct selected function and opponent,
at most60 seconds each, and do not establish playing strength.

## Public registration and capability refresh

[Preregistration](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/0618d5979298b8a044a5fa98bca39969166ab159/lem/gus-unanchored-outcomes-2026-10-19/registration)
precedes training:12 files,11,785,970 bytes, all freshly anonymously downloaded
and SHA256-verified. Registration SHA256:
`aaefde365f4e27667db96689fb2cd30084ab14d0bcf1ae3fb095765cf3c8efbb`.
Source is `b9a44691`; local recovery root is `scratch/lem/gus-unanchored-v1`.
All896 reserved roots exclude603,260 prior physical deals and become future
training exclusions. `run.py` owns bounded child segments; inspect its process
and completion receipts before resuming. No new result is established yet.

The refresh in [[gus-book-silas-tools]] finds no new Lem/Book replies; requests
remain pending. Silas PR166 remains open, and existing main outcome verification
suffices. Native transformer captures exist; the causal audit bridge remains
separate work. These boundaries do not change the comparison or its controls.
