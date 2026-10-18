---
title: "Lem observer overbake: a long formation trajectory"
kind: experiment
first_seen: 2026-10-17
last_updated: 2026-10-17
status: complete
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

Live run: [hatchling-overbake-200k](https://forge.coreweave.com/wandb/jasonyandell-forge42/lem-observer/runs/q61r1lm5),
entity `jasonyandell-forge42`, project `lem-observer`. The server confirmed the
run finished at step 200,000. The local run directory is
`scratch/lem/observer-overbake`. Source/registration commit: `47797a4a`.

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

The completion helper computed whole-deal intervals for the best checkpoint,
exported its worked hand, and published an allowlisted evidence directory to the
existing monthly HF repo. Its receipt is
`scratch/lem/observer-overbake/archive-status.json`: `published_and_verified`.
A fresh download of the final checkpoint matched its SHA256.

The local MPS job uses `scripts/cpu-slot --threads 2 --timeout 1800`, nice 10, in
a detached tmux session. The explicit 30-minute cap is an exception to the default
five-minute cap: Jason requested sustained training with a continuously live W&B
run, and the small model is expected to reach the fixed update budget within it.
There is no paid cloud compute and no exclusive machine lock. A timeout preserves
the most recent checkpoint for an explicit resume.

## Result and prediction grades

The run completed all 200,000 updates (802.4 equivalent epochs). Validation
selected step **4,000**, at 16.0 equivalent epochs. Continuing training improved
the fixed training probe while degrading prediction on held-out deals.

| Fixed-set measurement | Step 2,000 | Best validation: 4,000 | Final: 200,000 |
|---|---:|---:|---:|
| Training-probe NLL | 1.681 | 1.413 | 0.868 |
| Validation NLL | 1.756 | 1.617 | 2.739 |
| Validation next-tile accuracy | 39.73% | 42.26% | 36.95% |
| Validation visibly impossible probability | 17.35% | 9.51% | 11.49% |

All three registered predictions held: training-probe loss fell; the best
validation checkpoint preceded the final one; final impossible mass was below
its step-2,000 value. This is one deliberately overtrained seed, not a finding
about the benefits of overtraining for interpretability.

The validation-selected model's exploratory test accuracy is **42.32%
[41.74, 42.89]**, versus 3.34% [3.19, 3.52] for its initial twin and 29.32%
[29.19, 29.45] expected for visible-support-uniform guessing. NLL is **1.613
[1.594, 1.631]**, versus 1.749 [1.746, 1.751] for that stronger baseline;
the paired difference is −0.136 [−0.155, −0.118]. Its visibly impossible
probability is 9.63% [9.41, 9.85], versus zero for the masked baseline.
Intervals use 2,000 whole-deal bootstrap draws, keeping the four views and
all moves together; they condition on this model and this pour. They do not
cover seed or population uncertainty. The reused test set is exploratory.

## Interpretability handoff

Jason selected handoff to the interp work on 2026-10-17, posted as
[#207](https://github.com/jasonyandell/mk5-main/issues/207), while this workstream
continues with [[lem-observer-data-scale-2026-10-17]]. The specimens are ready;
no unit has been named or causally explained here.

**Start with the frozen evidence:**
[overbake archive, commit 035af17](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/035af17da11b3749c68a7e959ac2a6301a3cf667/lem/observer-overbake-2026-10-17).
It contains 43 model files: initialization, validation-best, final, and forty
5,000-update snapshots. The
[original archive](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/242f07beededcd8d1750c6cdda54e72d5449cee9/lem/observer-2026-10-17)
adds the early formation snapshots every 250 updates through step 2,000.
No snapshots exist at every evaluation between 2,000 and 5,000: the retained
best at 4,000 is the additional early specimen.

| File | Identity / use |
|---|---|
| `initial.pt` | Saved untrained twin, step 0; not a new random initialization |
| `best.pt` | Step 4,000, SHA256 `6ab6adae31ace79f2a2d1de4b063b9b6adcb663c0501b74b9665f65bc8f72991` |
| `latest.pt` | Step 200,000, SHA256 `7d3d778eefd7bb256eee9218e5fdc8d835a0bb581811b43a074ad2310dbaf626` |
| `deals.json`, `data.json` | Exact recordings, whole-deal splits, source and tokenizer hashes |
| `report.json` | `test` describes best; `final_test` describes final; `history` gives validation and training-probe trajectories |
| `intervals.json`, `worked-hand-best.html` | Both describe **best**, not the final overbaked specimen |
| `SHA256SUMS` | Checksums for the archived payload |

Code: `lem/observer/{model,data,inspect,train}.py`, source `47797a4a` or its
descendants. `torch.load(path, map_location="cpu", weights_only=True)` yields
`config`, `model`, and checkpoint metadata. Instantiate
`Observer(Config(**saved["config"]))`, load `saved["model"]`, and call
`model.eval()`. `prepare(records)` creates four viewer sequences per deal;
feed **only `x`** into the model. With `inspect=True`, the forward returns
logits plus per-block `attention`, `pre`, and `post` tensors. Read prediction
sites `QUERIES = 13,15,…,67`; each has its corresponding recorded tile target.
The native MLP has 256 units per block, two blocks, hence 512 units / 1,024
signed poles. Attention weights are not causal attributions.

**Scope and guardrails for the receiving study:** reread the clean-neuron skill
and [[lem-nanogpt-foundations]]. This is a next-play observer with an initial
own hand and within-hand public history, fixed seat-0 bid 30, no auction or
cross-hand identity, and no tested concept vocabulary yet. `legal` is a
referee-only diagnostic that uses the actor's real hand; it must not leak into
other-seat concepts, inputs or prediction masks. `possible` is viewer-visible
support. Interpretability examples and concept labels must read the visible
prefix only, even though the archive contains full deals for replay.

Register predictions before running probes. Use held-out deals, both poles,
the same vocabulary on the saved untrained twin, and a separate discovery /
confirmation split by whole deal for any newly proposed names. Report prediction
competence beside interpretations. Recalibrate uncertainty for repeated views
and time steps; the field-net 1.17 correction and numerical untrained floors
do not automatically transfer. Match causal doses and row difficulty. An
unnamed unit stays unnamed in the tested vocabulary. The useful first question
is whether a candidate mechanism is present before, at, and after the best
predictor; cleaner units after overbaking remain a hypothesis to test.

## Sources

- `lem/observer/train.py`, `tracking.py`, `requirements-wandb.txt`
- [[lem-nanogpt-foundations]]; [[clean-neurons-42]]
