---
title: "The squeeze: narrow the field net and ask whether its neurons get coarser or mixed (Lem, the paper's G2 width pilot)"
kind: experiment
first_seen: 2026-10-16
last_updated: 2026-10-16
status: complete
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

Run 2026-10-16/17 as registered: seven nets, 60 epochs each (the validation loss moved less than
0.0013 over the last ten epochs in every net, so none was stopped while still learning), the sealed
audit and the family test on each. Evidence (nets in both layouts, metrics, audits, families,
runner, logs, `SHA256SUMS`): `lem/squeeze-2026-10-16/` in
[texas-42-zeb-research-2026-10](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10).

| net (h1×h2) | neurons | regret (make prob.) | gap to random closed | agree | live poles | clean share | L1 / L2 clean | dead | broadened | within-family share | random families [95%] |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|
| 128×64 (control) | 192 | 0.0081 | 0.804 | 0.758 | 374 | 0.610 | 157/255 · 71/119 | 10 | 224 | 0.335 | 0.084 [0.022, 0.300] |
| 128×8 | 136 | 0.0103 | | 0.735 | 270 | 0.685 | 178/255 · 7/15 | 2 | 160 | 0.425 | 0.085 [0.019, 0.388] |
| 32×16 | 48 | 0.0112 | | 0.729 | 91 | 0.714 | 46/62 · 19/29 | 5 | 60 | 0.450 | 0.080 [0.000, 0.484] |
| 16×8, seed 0 | 24 | 0.0130 | 0.684 | 0.714 | 48 | **0.833** | 25/32 · 15/16 | 0 | 23 | 0.435 | 0.080 [0.000, 0.392] |
| 16×8, seed 1 | 24 | 0.0134 | | 0.712 | 46 | 0.652 | 21/31 · 9/15 | 2 | 25 | 0.600 | 0.085 [0.000, 0.560] |
| 16×8, seed 2 | 24 | 0.0130 | | 0.715 | 48 | 0.792 | 24/32 · 14/16 | 0 | 24 | 0.500 | 0.088 [0.000, 0.542] |
| 8×4 | 12 | 0.0159 | 0.615 | 0.698 | 23 | 0.652 | 10/15 · 5/8 | 1 | 16 | 0.312 | 0.083 [0.000, 0.375] |

Uniform random legal play has regret 0.0413. Shuffled-label clean share is 0.000 for every net.
The shipped net scores 0.597 under the same audit. The random-family bands are wide because many
poles share one best union, so the null is lumpy.

**What the narrow nets keep.** At 16×8 (seed 0) the clean labels are `winner-cannot-be-beaten` 8,
`partner-winning` 4, `last` 3, `lead` 2 and `trump-led` 2. There are also `count`-block declaration
detectors (8 poles), `no-trump-held` 2, `trumps-3plus`, `count-on-trick` and `count-out-20plus`.
At 8×4 the clean labels are `partner-winning` 3, `lead`, `last` and `winner-cannot-be-beaten` 2
each, plus one each of `trump-led`, `holds-boss-trump`, `can-take-it` and `holds-master`.
Mastery, voids and the score mostly drop out of named neurons at 16×8. The within-family
broadening is almost all one coarse unit, `last` ∪ `third`, "late in the trick": 4 of 5
within-family pairs at 8×4, 8 of 10 at 16×8 and 25 of 27 at 32×16.

**The predictions, graded.**

| | predicted | measured | |
|---|---|---|---|
| 1. clean share falls with width | 32×16 below 128×64; 8×4 lowest | 0.610 → 0.685 → 0.714 → 0.65–0.83 → 0.652 | **wrong, backwards**: narrow nets are cleaner |
| 2. within-family share above the random band at every width, higher at 16×8 and 8×4 than at 128×64 | | above at 4 of 7 nets, inside at 32×16, 16×8 s2 and 8×4; always 3.7–7× the null mean; higher at 16×8 in all seeds, lower at 8×4; carried by one union, `last` ∪ `third` | partly holds |
| 3. broadened share rises as width falls | | 0.60, 0.59, 0.66, 0.48–0.54, 0.70 | wrong (no trend) |
| 4. regret rises as width falls; control within 0.001 of 0.0084; 8×4 above 0.03 | | monotone; 0.0081; 8×4 **0.0159** | two of three; narrow nets play far better than I expected |
| 5. fewer dead poles in narrow nets; none at 8×4 | | 10, 2, 5, 0/2/0, 1 | wrong |
| 6. 16×8 seeds within 0.10 of clean share; ≥ half of families shared | | 0.652–0.833 (spread 0.18); exactly half shared | first wrong, second holds |
| 7. 128×8: layer 2 broadened more than the control's; layer 1 clean within 0.05 | | 10/15 vs 70/119; 0.698 vs 0.616 | first holds (small n), second wrong: layer 1 got cleaner too |

**Reading.** Jason's grainy photograph appears, with sharp grains. Squeezed, the field net keeps
fewer neurons, and each is *more* often one exact concept of the teacher's: 0.83 at 24 neurons
against 0.61 at 192. This is the essay's direction, and stronger than its "coarser" in one way:
the units the narrow net keeps are clean under fine labels, not only under unions. Where they do
broaden, they broaden within a family, chiefly one coarse unit, "late in the trick". The narrow
net's vocabulary is the trick in front of it: is the winner beaten, is partner winning, where am I
in the trick, was trump led, which pip is trump. That vocabulary costs less play than I expected:
24 neurons close 0.684 of the gap from random play to the teacher, against 0.804 for 192 neurons,
and 12 neurons close 0.615. Squeezing only the second layer to 8 costs 0.0022 of regret, about
half of what squeezing both layers to 16×8 costs. For the north star this is the first point on
the feature-count curve built from below: a small set of clean features carries most of the play,
and the rest of the fidelity (0.684 → 0.804 and beyond) is where the many-feature claim lives.
Seed scatter at 16×8 is large (clean share 0.65–0.83), so every narrow-net number above carries
at least that much spread.

Caveats: one seed except at 16×8; the 48-concept v1 list (the declaration enters only as raw
`count` features); the family test's power is weak below 32×16 (fewer than 30 broadened poles);
pip trumps, seat 0 bids 30. Next: checkpoints of the 16×8 and 128×64 nets to watch the units form
(the essay's formation test), and concept list v2 on these nets once #185's labels run.

## Correction, 2026-10-20

The family test's null (random reassignment of concepts to families) sets the bar too low.
Untrained nets beat it too ([[lem-family-statistic-2026-10-20]]), so "units broaden within one
family" is not yet shown against untrained twins. The re-check is #229. The clean shares, and
narrower-is-cleaner under conditioning ([[lem-squeeze-conditioned-2026-10-20]]), are unaffected.

## Links

[[clean-neurons-42]] · [[lem-the-teacher]] · [[lem-field-net-audit-2026-10-15]] ·
[[lem-fidelity-by-concept-2026-10-16]] · [[scenario-player]] · [[strategy-fusion-fence]]
