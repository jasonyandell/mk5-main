---
title: "Lem observer: half a million games for the same tiny model"
kind: experiment
first_seen: 2026-10-17
last_updated: 2026-10-17
status: complete
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
Registered and implemented at `655bc7de` ([PR #210](https://github.com/jasonyandell/mk5-main/pull/210),
merged). Training started in
[W&B run 5lvllnsj](https://forge.coreweave.com/wandb/jasonyandell-forge42/lem-observer/runs/5lvllnsj).
The [registration archive](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/24c9731450d2bb9a130d34cc3c208f5da4a8cdd6/lem/observer-corpus-squeeze-2026-10-17)
preserves the untrained twin, source hashes, split counts and W&B identity;
its downloaded initialization matched the local SHA256. The initial tensors
also match the 50k specimen exactly.

Training and the subsequent comparison, coverage and archive jobs completed.
The original lesson remains tied to its original model.

Validation before launch: `scripts/check --fast` GREEN (33 Lem tests, wiki,
licenses, conflict markers). Every prepared array matched the preceding
implementation across 90 replayed deals, all nine declarations and four views.
The training probe uses the first 256 training deals, all from the older field
generation; it is a local diagnostic rather than a mixture-balanced loss estimate.

## Result

Completed 200,000 updates; validation selected the final step. The sampled
exposure audit exactly reproduced the saved sampler state: all **399,752
training deals** were encountered, covering 1,598,492 of 1,599,008 viewer
sequences. There were 12.8 million sampled sequences and 358.4 million tile
loss targets, or 8.005 equivalent passes. The unobserved 516 individual views
are consistent with replacement sampling; no whole training deal was missed.

On the **45,099 fresh test deals**, with whole-deal 95% bootstrap intervals:

| Measurement | 50k corpus, validation best | 500k corpus, validation best |
|---|---:|---:|
| Next-tile accuracy | 49.36% [49.30, 49.43] | **52.93% [52.86, 53.00]** |
| NLL | 1.368 [1.366, 1.369] | **1.261 [1.260, 1.263]** |
| Visibly impossible probability | 4.664% [4.645, 4.682] | **1.421% [1.413, 1.428]** |

Paired changes, large minus small: accuracy **+3.567 percentage points
[3.519, 3.618]**, NLL **−0.1063 [−0.1071, −0.1055]**, and impossible mass
**−3.243 points [−3.258, −3.226]**. Predictions 1–2 held with intervals
excluding zero. Prediction 3 remains ungraded: no unit interpretation or
causal test was performed by the training job.

Fresh-test saved initialization gets 3.345% [3.327, 3.363] accuracy and
3.3473 [3.3471, 3.3475] NLL. The stronger visible-uniform baseline gets
29.270% [29.257, 29.282] expected accuracy and 1.7501 [1.7498, 1.7503] NLL;
it has zero visibly impossible mass by construction.

| Prediction group | Trained accuracy | Visible-uniform expected accuracy |
|---|---:|---:|
| Viewer owns the acting hand | 79.52% [79.43, 79.61] | 58.73% [58.68, 58.78] |
| Another seat acts | 44.07% [44.00, 44.13] | 19.45% [19.44, 19.46] |
| More than one legal move exists | 40.23% [40.14, 40.33] | 16.77% [16.76, 16.78] |

Three quarters of the scored targets concern other seats. These are predictions
from the recorded prefix with hidden hands unavailable, not autonomous playing
strength. The interpretation of NLL beside accuracy is in
[[lem-nanogpt-foundations]].

**The equal-update comparison is different.** At 20,000 updates the large
corpus's accuracy is 49.05% [48.99, 49.12] versus 49.15% [49.08, 49.21]; paired
change **−0.091 points [−0.139, −0.048]**. NLL improves slightly,
**−0.00194 [−0.00254, −0.00136]**, as does impossible mass,
**−0.131 points [−0.138, −0.125]**. This gives a concrete case where loss
improves while top-1 accuracy does not. The larger best specimen consumed
more updates; the result does not isolate corpus size from compute or teacher
mixture. One initialization does not estimate seed-to-seed variation.

## Completed evidence

[Frozen archive, commit db83ef2](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/db83ef2ea26e95cba36340e9dab87c3c055f7057/lem/observer-corpus-squeeze-2026-10-17)
contains 307,533,405 bytes: all recordings and provenance, the untrained twin,
83 formation checkpoints, best/final checkpoints, report, paired fresh/reused
comparisons, verified sampling coverage, W&B receipt and checksums.
The fresh download of `best.pt` matched SHA256
`99fd106a15ae468dca9c02426541816f1121a2cdef0cc7b96fd37f2122f31d8d`.

The run finished within its registered cap: 1.526e13 CPU instructions,
1,098.2 CPU seconds and 1,304.2 wall seconds. MPS did the neural computation;
CPU instructions do not measure GPU work. The interpretability handoff remains
[#207](https://github.com/jasonyandell/mk5-main/issues/207).
