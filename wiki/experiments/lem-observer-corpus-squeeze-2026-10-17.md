---
title: "Lem observer: half a million games for the same tiny model"
kind: experiment
first_seen: 2026-10-17
last_updated: 2026-10-17
status: active
---

## Question

Jason selected **too many distinct games for a little model, trained a normal
number of times**. Can the unchanged observer learn useful shared distinctions
from the full existing lawful pours, and what do its native mechanisms become?
This is the next specimen for [[clean-neurons-42]], not another repetition of
the small-corpus overbake in [[lem-observer-overbake-2026-10-17]].

Cammarata's [essay](https://already-clean.goodfire.com/), including its appendices,
was reread from the author-provided text. Its pressure experiments reduce
architectural capacity and find broader distinctions; this experiment instead
increases the range of experience at fixed capacity. That is a related empirical
question, not an exact replication or a proof that this corpus exceeds capacity.
The leading interpretation to test is coherent reuse and generalization. An
unnamed unit is not evidence of arbitrary packing. Training curves establish
learning; subsequent formation and causal studies must establish neuron meanings.

## Registered before training

- **Corpus:** all 250,000 Pour A games plus the earlier 250,000-game Zeb pour.
  Sources are [Pour A](https://huggingface.co/datasets/jasonyandell/texas-42-lem-pour-a)
  and [Zeb 250k](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-pour-250k).
  Duplicate physical deals and overlapping seed IDs are rejected. Smoke and
  alternative-utility recordings of the same deals are excluded.
- **Information:** original own seven tiles, fixed bid-30 contract, relative
  seats and public actor/tile history. Exactly the existing 46-token encoding
  and 28-way next-tile output. No added inputs, splat head, Nello examples,
  player identities or reward optimization in this run. The selected splat
  design remains the subsequent extension in [[lem-nanogpt-foundations]].
- **Model:** 107,392 parameters; width 64, two blocks, two attention heads,
  256 GELU units per block. Fresh seed-42 initialization; verify tensors match
  the preceding specimens. Native pre/post activations remain readable.
- **Training:** batch 64, AdamW 0.001 constant, betas (0.9, 0.95), matrix weight
  decay 0.1, gradient clip 1.0, **200,000 updates**, no early stopping. This is
  8.005 equivalent passes over 1,599,008 training views, versus 802.4 in the
  5,000-deal overbake. Sampling remains with replacement, so these are exposure
  equivalents rather than eight complete shuffled passes. Budget is fixed
  before observing training or new test metrics.
- **Formation:** retain initialization, steps 250/500/1,000, every 2,500 updates,
  and separate validation-best and final checkpoints. Preserve optimizer,
  sampler/RNG state, recordings, split hash and W&B receipt. Full validation
  and a fixed training probe at each saved formation point.
- **Compute:** local MPS, two CPU threads, $0 rental. `scripts/cpu-slot --threads 2
  --timeout 2400` bounds the job. Exception reason: one fixed 200k-update MPS
  trajectory with full 50k-deal validation; repeatedly rebuilding and replaying
  half a million games every five minutes would dominate work. It remains
  niced and shares the machine; no exclusive reservation. Expected tens of
  minutes, not a wall-clock training budget; snapshots permit recovery.

Preflight: **399,752 train / 50,210 validation / 50,038 test deals**, split by
the existing physical-deal hash before four views. All prior 50,000 recordings
and assignments are exactly retained. **45,099 test deals** are outside that
prior corpus; 4,939 are reused. Fingerprint:
`aeb1e173da23bb3414c394e0ba05e22137d3b62206abbe28851c5b9c1b632067`.

The corpus is 100 times the first observer's and ten times
[[lem-observer-data-scale-2026-10-17]]. It combines teacher generations: the
earlier 50k field, its following 200k field, and Pour A's pip field plus the
doubles/no-trump rung. Their exact specs, source commits and manifest hashes
are retained in the evidence. Teacher mixture and declaration balance change;
these records do not establish a behaviorally diverse player population.

## Predictions and grading

1. Validation-selected half-million-game model has lower next-tile NLL than
   the validation-selected 50k model on the 45,099 fresh shared test deals.
2. It places less probability on visibly impossible tiles on that cohort.
3. The interp hypothesis is that useful native units learn shared distinctions
   whose development can be followed across checkpoints, including broader
   patterns than the current concept list names. This stays **ungraded until
   native-unit evidence exists**; a loss decrease cannot grade it.

For predictions 1–2 report paired whole-deal 95% bootstrap intervals, 2,000
draws, seed 42; count all views and plays of one deal together. Include raw
top-1 accuracy, own/other/discretionary groups, saved untrained twin and
visible-uniform baselines. The fresh cohort is primary; reused deals are
separate and exploratory. Also compare both runs at step 20,000, which asks
about the same update budget and is distinct from training each for roughly
eight equivalent passes. The best-vs-best comparison has unequal updates,
validation sets and selection schedules; it is a comparison of specimens,
not a causal estimate of dataset size alone.

Any later clean-share result needs its own vocabulary-matched untrained floor,
uncertainty and learning-quality measurement. The field-net scope sentence and
its numeric corrections cannot be copied to a sequence model. Here the scope
is this observer, these original-own-hand/public-history inputs, this teacher
mixture and whichever vocabulary the interp study actually tests.

## Execution and evidence

Output directory: `scratch/lem/observer-corpus-squeeze`.
Preflight and source list: `scratch/lem/corpus-squeeze/`.
The experiment is registered; training results and archive receipt will be
added when available. The original lesson remains tied to its original model.
