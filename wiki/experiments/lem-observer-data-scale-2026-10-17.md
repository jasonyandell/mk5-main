---
title: "Lem observer: more deals at the same model size"
kind: experiment
first_seen: 2026-10-17
last_updated: 2026-10-17
status: complete
---

## Question and registration

After the worked-hand lesson, Jason chose to hand the existing checkpoints to
the interp work and build the next training experiment. The handoff is in
[[lem-observer-overbake-2026-10-17]]. This experiment asks whether more distinct
recorded deals improve prediction where repetition of the small corpus stopped
helping. It adds no population identities, auction, reward learning or new
architecture; those remain later choices in [[lem-nanogpt-foundations]].

Train the same 107,392-parameter observer from seed 42 initialization on the
first **50,000** Pour A deals by seed, versus the prior 5,000. Width 64, two
blocks, two heads, batch 64, AdamW learning rate 0.001, constant schedule and
20,000 total updates. Evaluation remains every 250 updates; retain formation
checkpoints every 1,000. No early stopping. Use local MPS, two CPU threads,
`scripts/cpu-slot` with its default 300-second cap. If the cap is reached,
resume the same experiment and sampler, not a fresh restart. No paid compute.

Preflight found 39,963 training deals, 5,098 validation and 4,939 test. The
original 5,000 recordings and split labels are retained exactly. The expanded
record fingerprint is
`0496869bbb9cd0e1dd273ddf22feee9117da45d1eaa389bbbe958d6b9abe6902`.
The larger test contains **4,436 deals outside the original 5,000**; none
trained either model. This new cohort is the primary comparison set. The
original 503 test deals remain an exploratory secondary cohort.

Primary comparison: **step 20,000 against step 20,000**, on the same 4,436 new
test deals, with the two models' losses paired by deal. Each run consumed
1,280,000 sampled viewer sequences by then. That is about 8.0 equivalent
epochs for the larger corpus versus 80.2 for the original. Equivalent epochs
are not equal compute; update counts and batch size define this comparison.
Validation-best against validation-best is a separate descriptive comparison,
because the validation sets and selection opportunities differ.

Predictions registered before training / inspecting the new test results:

1. At step 20,000, the larger-data model has lower next-tile NLL on the new
   shared test cohort than the old model's step-20,000 checkpoint.
2. On that cohort, the larger-data model at step 20,000 assigns less total
   probability to visibly impossible tiles than the old step-20,000 model.

Report paired whole-deal 95% bootstrap intervals (2,000 draws, seed 42),
keeping all views and decisions of a deal together; also report top-1
accuracy, untrained and visible-uniform baselines, own / other-seat and
discretionary groups. Grade the signs and whether the intervals exclude zero.
One training seed supports a result for these fitted models and these records,
not a population or training-seed guarantee. The corpus is a nested seed prefix,
so declaration/source composition can change with size; this is a data-prefix
experiment, not a proof that size alone explains a difference.

Verify the initial parameter tensors are identical across runs, and refuse
the comparison if a test deal is present in either training split. Preserve
best, final, optimizer/sampler state, exact recordings, hashes and W&B receipts.
The first model and both lessons stay as they were.

## Sources and launch

- Training implementation: `lem/observer/train.py`; the same Pour A `pip` and
  `dtnt` source manifests as [[lem-observer-2026-10-17]].
- New output: `scratch/lem/observer-data-scale`.
- Reference at matched budget:
  `scratch/lem/observer-overbake/step-20000.pt` in the frozen handoff archive.
- Source declaration counts: 0: 2018; 1: 2413; 2: 2475; 3: 2447; 4: 15612;
  5: 14057; 6: 4535; doubles: 3346; no trump: 3097. Declaration 8 is absent.

The registration was committed as `a2ec2d63` before training. The comparison
implementation and its tests are `3770c0f8`.

## Result

Completed the fixed 20,000-update budget; validation selected **step 18,500**.
The initial parameter tensors matched the original run exactly. Both registered
predictions held on the 4,436 fresh test deals, with paired intervals excluding
zero. At the same update budget:

| Fresh-test measurement | 5,000-deal run, step 20,000 | 50,000-deal run, step 20,000 |
|---|---:|---:|
| Next-tile accuracy | 39.08% [38.88, 39.28] | 49.00% [48.80, 49.20] |
| NLL | 2.210 [2.198, 2.223] | 1.370 [1.365, 1.375] |
| Visibly impossible probability | 9.75% [9.66, 9.84] | 4.70% [4.64, 4.76] |

The paired differences, new minus old, are **+9.92 percentage points
[9.72, 10.11]** accuracy, **−0.840 [−0.850, −0.830]** NLL, and **−5.05 points
[−5.11, −4.99]** impossible mass. On the same fresh cohort, saved initialization
gets 3.32% [3.27, 3.37] accuracy and 3.348 [3.347, 3.348] NLL;
visible-uniform gets 29.28% [29.24, 29.32] expected accuracy and
1.750 [1.749, 1.751] NLL, with zero impossible mass by construction.

This does not merely beat an overfitted reference: in the separate
validation-best comparison, the new specimen gets **49.21% [49.00, 49.41]**
against **42.31% [42.11, 42.51]** for the original best on the same fresh deals.
The paired gain is 6.89 points [6.72, 7.07]. Discretionary-play accuracy is
36.54% [36.25, 36.81] versus 29.92% [29.66, 30.17], with a visible-uniform
baseline of 16.77% [16.74, 16.80]. Best-model update counts differ (18,500 vs
4,000), so that comparison is descriptive rather than the primary fixed-budget
test. The reused 503-deal cohort is reported separately in the artifact.

The result supports using more recorded experience for this architecture and
seed. It establishes neither population modeling, autonomous playing strength
nor neuron meanings. Data-prefix composition, one initialization and the
fixed-contract teacher remain the experiment's scope.

## Evidence and verification

[Frozen archive, commit 93ea96c](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/93ea96ca04d305ee215f693b5f1ee0e492dbf89a/lem/observer-data-scale-2026-10-17)
contains the exact 50,000 recordings and splits, 23 model files (initial,
best, final and twenty 1,000-update snapshots), hashes, preflight, report,
intervals, the paired comparison and the best-model worked hand. `intervals.json`
describes the best checkpoint on all 4,939 test deals; `comparison.json`
separates fresh and reused cohorts and fixed-budget / best comparisons.
The fresh download of `best.pt` matched SHA256
`5d4823891bd8690fd99e517c9b75f649e304a9957bdedb04217059cde04cad99`.

[W&B run 6l2c7jji](https://forge.coreweave.com/wandb/jasonyandell-forge42/lem-observer/runs/6l2c7jji)
is finished and includes the fresh-test comparison in its summary. The full
training job used 1.909e12 CPU instructions, 123.6 CPU seconds and 148.6 wall
seconds; MPS did the neural computation, so CPU instructions do not measure its
GPU cost. The registered cap was respected.

Comparison tests check the nested recording contract, rejected split relabeling
and duplicates, identical initialization, correct fresh/reused cohorts, exact
CPU continuation, decision-weighted paired intervals and invariance to duplicating
correlated views within a deal. The full observer test module passes nine tests.
The shared handoff for subsequent unit reading is
[#207](https://github.com/jasonyandell/mk5-main/issues/207).
