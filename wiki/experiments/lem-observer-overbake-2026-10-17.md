---
title: "Lem observer overbake: a long formation trajectory"
kind: experiment
first_seen: 2026-10-17
last_updated: 2026-10-17
status: active
---

## Registration

Jason requested deliberate extended training of the hatchling from
[[lem-observer-2026-10-17]], with live Weights & Biases charts while working
through the lesson. The clean-neuron skill was reread before this continuation.
This experiment preserves the same model and data rather than enlarging either.

The run branches from step 2,000, retaining model, AdamW optimizer, sampler and
RNG state. It continues to **200,000 total updates**, with the original constant
learning rate 0.001, batch 64, width 64, two blocks and two heads. The 3,988/509/503
deal split remains fixed. No early stopping: best and final are separate specimens.

An equivalent epoch is `step × 64 / (3988 × 4)`, because each deal has four
viewer sequences and sampling is with replacement. The target is 802.4 equivalent
epochs, about 794.4 more than the first run, not 802 complete shuffled passes.

Predictions registered before continuation:

1. The fixed training-probe loss ends below its step-2,000 value.
2. The best validation checkpoint precedes the final checkpoint; deliberately
   continuing can expose overfitting or optimizer fluctuations.
3. The final validation probability on visibly impossible tiles falls below
   the step-2,000 value (0.1735).

These are learning predictions, not claims about neuron meanings. Later native-unit
study uses both poles, held-out examples, the saved untrained twin, appropriate
vocabulary and causal tests. This net sees an original own hand and within-hand
public history; it has no named concept vocabulary yet. The field-net untrained
floors and interval correction are not assumed to apply to this transformer.

## Monitoring and preservation

W&B logs train-batch loss, a fixed 256-deal training probe, validation losses,
accuracies and impossible mass every 250 updates, using equivalent epochs on the
horizontal axis. Validation includes all 509 deals. The sampler is untouched by
evaluation. Test data is not a live curve; best and final are evaluated after the
run. Since the original test result has already been inspected, later test results
are exploratory rather than a fresh confirmation.

Latest state is saved every 250 updates, formation snapshots every 5,000, and the
best checkpoint whenever validation loss improves. W&B receives snapshots and a
final evidence artifact; local files retain exact recordings, source hashes and
the initial twin. The original worked-hand lesson and first-run files remain
available unchanged.

The local MPS job uses `scripts/cpu-slot --threads 2 --timeout 1800`, nice 10, in
a detached tmux session. The explicit 30-minute cap is an exception to the default
five-minute cap: Jason requested sustained training with a continuously live W&B
run, and the small model is expected to reach the fixed update budget within it.
There is no paid cloud compute and no exclusive machine lock. A timeout preserves
the most recent checkpoint for an explicit resume.

## Sources

- `lem/observer/train.py`, `tracking.py`, `requirements-wandb.txt`
- [[lem-nanogpt-foundations]]; [[clean-neurons-42]]
