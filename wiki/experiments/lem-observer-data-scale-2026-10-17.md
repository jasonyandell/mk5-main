---
title: "Lem observer: more deals at the same model size"
kind: experiment
first_seen: 2026-10-17
last_updated: 2026-10-17
status: active
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

This page is the registration. Results and a frozen artifact link follow the
run; no training or interpretability success is claimed in advance.
