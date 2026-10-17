---
title: "The squeeze: narrow the field net and ask whether its neurons get coarser or mixed (Lem, the paper's G2 width pilot)"
kind: experiment
first_seen: 2026-10-16
last_updated: 2026-10-16
status: active
---

Jason's idea, 2026-10-16: train undersized nets on the data we have, as a pilot, and look for
coarse features, "stronger, lower-resolution connections, like a grainy photograph". It is the
squeeze of [[clean-neurons-42]] (A3), taken from the essay that underpins this program: Nick
Cammarata, ["Neural Networks Are Already Clean"](https://already-clean.goodfire.com/) (Goodfire),
re-read for this page. Its squeeze cut InceptionV1's mixed3b from 480 channels to 10, and a medium
ConvNeXt's last layer from 1,024 to 13, both retrained from scratch with no bypass. The neurons
broadened instead of mixing: "one 'curves & loops' neuron instead of ten curve detectors". Its
author holds that small models give broad neurons and big ones narrow neurons. Superposition
predicts the opposite: unrelated concepts sharing a unit. G2 of the paper's work order
([`lem/paper/README.md`](../../lem/paper/README.md); [[lem-the-teacher]]). Registered before any
net was trained. Code: `lem/probe/squeeze.py` (CC0).

## Question

As the field net narrows, do its neurons cover coarser families of the teacher's concepts (the
essay), or unrelated mixtures (superposition)? And what does each width cost in fidelity? This is
the feature-count curve built from the other side: a net trained at width k, not a big net
pruned to k.

## Pre-registration

- **Data.** The five pip pours behind the shipped net (`pour_make_netfield_s100001_d50000` and the
  four `zeb200k` pours, 250,000 deals, about 4.3M positions), one dataset build for every net.
  The trainer holds out deal % 10 == 0; those rows are the audit's sample (434,535 rows, the same
  48 Silas labels, [[lem-field-net-audit-2026-10-15]]), so no net is audited on rows it trained on.
- **Nets.** `scenario.field_train.train`, margin target, MLX, the trainer's defaults except
  **60 epochs** (twice the shipped net's 30, so a small net is not undertrained; the validation
  loss curve is reported, and a curve still falling at epoch 60 is flagged). Widths (h1 × h2):
  **128×64** (the control, retrained by the same recipe), **32×16**, **16×8**, **8×4**, and
  **128×8** (the essay's design: one layer squeezed, the other kept). Seed 0 for each, plus seeds
  1 and 2 at 16×8. Seven nets. The trainer builds `field-v3`; on pip deals the pip-kind bit is
  always on, so each net is folded exactly into the `field-v2` net it equals there (its first-layer
  weights join the bias, integers; tested) and audited by the unchanged probe kit.
- **Measures per net.** The sealed audit (top 1%, clean ≥ 0.9 at base ≤ 0.5, partial, mixed,
  dead, 20 shuffles); validation regret (`regret_q`, make probability) and agreement from the
  trainer; and the **family test**:
  - The 48 concepts in 11 families fixed here: position, time, side, trump held, trump state,
    mastery, trick, count, score, doubles, forced (membership in `squeeze.FAMILIES`).
  - A live pole is **broadened** when the best union of two concepts (union base ≤ 0.5) labels its
    top set at least 0.05 better than the best single concept.
  - The statistic is the share of broadened poles whose best union lies within one family, against
    200 random reassignments of concepts to families of the same sizes (a null that keeps the
    unequal number of within-family pairs).
  - Coarse neurons put that share above the null; mixed neurons put it at the null.
- **Predictions, before training.**
  1. The single-label clean share falls with width: 32×16 below 128×64, and 8×4 the lowest.
     Both readings predict this, because a broad neuron scores partial under a fine label.
  2. **The essay's test.** The within-family share of broadened poles sits above the null's 95%
     band at every width, and it is higher at 16×8 and 8×4 than at 128×64.
  3. The share of live poles that are broadened rises as width falls (more coarse units).
  4. Regret rises as width falls; the retrained 128×64 is within 0.001 of the shipped net's
     0.0084; 8×4 is above 0.03.
  5. Dead poles: fewer in narrow nets (8×4 has none).
  6. The three 16×8 seeds have clean shares within 0.10 of each other, and their broadened poles
     fall in the same families: at least half of the families broadened in one seed are broadened
     in the other two.
  7. 128×8: its 16 layer-2 poles are broadened more often than the 128×64 control's layer-2 poles,
     and its layer-1 clean share is within 0.05 of the control's.
- **What would be a surprise.** A within-family share at the null in the narrow nets: the
  squeezed field net mixes rather than coarsens, against the essay. That would send G3's causal
  tests to the narrow nets first.
- **Cost.** One slot call trains all seven nets from one dataset build: the dataset takes about
  four minutes (42 s per 50,000 deals) and a net about four (26 s per 30 epochs over 50,000
  deals, scaled), so about half an hour in one process. That is over the 300 s house rule, with the
  reason written in the runner: one dataset build serves every width, and splitting it would rebuild
  it seven times. The audits, families and regret are one slot call per net, minutes. CPU and the
  Mac's GPU; no Haiku readers in the pilot.

## Results

Not yet run.

## Links

[[clean-neurons-42]] · [[lem-the-teacher]] · [[lem-field-net-audit-2026-10-15]] ·
[[lem-fidelity-by-concept-2026-10-16]] · [[scenario-player]] · [[strategy-fusion-fence]]
