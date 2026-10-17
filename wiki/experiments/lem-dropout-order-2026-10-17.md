---
title: "Already-clean test 3: the drop-out order — which of the teacher's concepts survive as the field net narrows (Lem, #196)"
kind: experiment
first_seen: 2026-10-17
last_updated: 2026-10-17
status: complete
---

Part of THE ALREADY-CLEAN TEST (#193; [[lem-the-teacher]]). The squeeze ([[lem-squeeze-2026-10-16]])
ran one seed at most widths, and its 16×8 seeds scattered from 0.65 to 0.83. Formation
([[lem-formation-2026-10-17]]) found an untrained floor of 0.15–0.28 for the clean share. This
test puts three seeds at every width, adds two narrower widths, and measures the floor at each.
The prize is the order in which the teacher's concepts drop out as the net narrows. The last
concepts standing are what makes 42 tick: the game's vocabulary, ranked by how much play each
word carries. Registered before any new net was trained.

## Pre-registration

- **Nets.** The squeeze's recipe (five pip pours, margin, MLX, 60 epochs) at **128×64, 128×8,
  32×16, 16×8, 8×4, 4×2 and 2×1**, seeds **0, 1, 2**. The squeeze's seed-0 nets and its 16×8
  seeds 1 and 2 are reused, being the same recipe; the rest are trained new.
- **The untrained floor.** The same seven widths and three seeds at zero epochs: the initial net,
  quantized by the trainer.
- **Measures per net.** The sealed audit (48 v1 labels, 20 shuffles); validation regret and the
  share of the gap from random play to the teacher closed; for the trained nets at 32×16 and wider,
  the #191 split-half on the clean share.
- **The vocabulary of a width.** The Silas concepts that label at least one clean pole in at
  least two of the three seeds. A concept's **survival width** is the narrowest width whose
  vocabulary still holds it. The drop-out order is the concepts sorted by survival width.
- **Predictions, before training.**
  1. **Nesting.** For at least 0.8 of the concepts in any width's vocabulary, the concept is also
     in the vocabulary of every wider width.
  2. **The last words.** The 2×1 vocabulary, if it has any word, includes at least one of
     `lead`, `last`, `partner-winning` and `winner-cannot-be-beaten`.
  3. **Two neurons still play.** At 2×1 the net closes at least 0.40 of the gap from random play
     to the teacher (mean over seeds).
  4. **Above the floor.** Every trained net's clean share exceeds its width's untrained floor
     (mean over that width's three untrained seeds) by at least 0.20.
  5. **Seed scatter.** At 32×16 and wider, the three seeds' clean shares span less than 0.15.
- **Cost.** 15 new trainings at about 9 minutes each, about 2.3 hours on the Mac's GPU, from one
  dataset build per call. The 21 untrained nets take seconds. About 36 audits at 1–3 minutes each.

## Results

Run 2026-10-17 as registered, with one change: every trained net was trained fresh, not reused
from the squeeze. The recipe is deterministic, so the reused nets came back identical: 128×64 seed 0
at 0.610 and 16×8 seed 0 at 0.833, as in the squeeze. Evidence is `lem/dropout-order-2026-10-17/`
in [texas-42-zeb-research-2026-10](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/da59db8f64548dce335b484befa8eabab0d82e4a/lem/dropout-order-2026-10-17):
42 nets with their training records, 42 audits, 9 split-halves, the grade and `grade.py`.

**Scope.** This net, these inputs and this vocabulary:
- the field net's recipe on five pip pours;
- one position, against uniform beliefs, with no player-behaviour history;
- the 48 Silas concepts of list v1.

Every share below was 0.000 on all 20 label shuffles.

| width | neurons | clean share, seeds 0 / 1 / 2 | untrained floor (mean) | gap to the teacher closed (mean) | split-half, A / B |
|---|---:|---|---:|---:|---|
| 128×64 | 192 | 0.610 / 0.654 / 0.636 | 0.168 | 0.801 | 0.61/0.62, 0.65/0.66, 0.64/0.64 |
| 128×8 | 136 | 0.685 / 0.674 / 0.675 | 0.174 | 0.753 | 0.69/0.67, 0.67/0.67, 0.68/0.66 |
| 32×16 | 48 | 0.714 / 0.677 / 0.656 | 0.181 | 0.730 | 0.70/0.75, 0.67/0.67, 0.66/0.66 |
| 16×8 | 24 | 0.833 / 0.652 / 0.792 | 0.226 | 0.682 | |
| 8×4 | 12 | 0.652 / 0.727 / 0.826 | 0.167 | 0.611 | |
| 4×2 | 6 | 0.778 / **0.125** / 0.333 | 0.167 | 0.284 (0.456 / 0.207 / 0.188) | |
| 2×1 | 3 | 1.000 / 0.667 / 0.600 (3–5 live poles) | 0.111 | 0.281 (0.218 / 0.207 / 0.417) | |

**The drop-out order.** A width's vocabulary is the concepts that label a clean pole in at least
two of its three seeds. Concepts are listed by the narrowest width that still holds them:

| survives to | concepts |
|---|---|
| 2×1 | `lead`, `late` |
| 8×4 | `last`, `third`, `partner-winning`, `no-trump-held`, `holds-boss-trump`, `trumps-3plus` |
| 16×8 | `count-on-trick`, `trump-led` |
| 32×16 | `winner-cannot-be-beaten` |
| 128×8 | `holds-master`, `bidders-lead`, `partner-out-of-trump`, `setter-out-of-trump`, `holds-five` |
| 128×64 only | `second`, `doubles-2plus` |

**The predictions, graded.**

| | measured | |
|---|---|---|
| 1. nesting ≥ 0.8 | 57 of 62 = 0.919 | holds |
| 2. the 2×1 words include `lead`, `last`, `partner-winning` or `winner-cannot-be-beaten` | `lead`, `late` | holds |
| 3. 2×1 closes ≥ 0.40 of the gap | 0.281 | **wrong** |
| 4. every trained net ≥ its floor + 0.20 | 19 of 21 (4×2 seeds 1 and 2: −0.04, +0.17) | **wrong** |
| 5. seed span < 0.15 at 32×16 and wider | 0.044, 0.011, 0.059 | holds |

**Reading.**

- **The squeeze replicates with seeds.** Narrower nets are cleaner, not mixed, down to 12 neurons:
  - means 0.633 at 192, 0.678 at 136, 0.682 at 48, 0.759 at 24 and 0.735 at 12;
  - each 0.44–0.66 above its untrained floor;
  - every split-half within 0.05.

  The wide seeds agree to within 0.06. Below 48 neurons the seeds scatter, 0.65–0.83 at 24 and
  at 12.
- **What makes this net tick, in the order it gives the words up.** The last two words are where
  the mover sits in the trick: `lead` and the coarse `late`. Next come the trick in front of the
  mover (`last`, `third`, `partner-winning`) and the mover's trump holding (`no-trump-held`,
  `holds-boss-trump`, `trumps-3plus`). Then count and trump on the table, then whether the
  winner can be beaten. Mastery outside trump, the bidder's lead and the setters' voids need
  136 neurons, and `second` and `doubles-2plus` appear only at full width.

  This matches §4 of the audit, where every master pole is a trump master. The general master is
  a late word, and the boss trump an early one.
- **The nesting misses are one word.** All five misses come from `late` and `lead` at 4×2 and 2×1.
  `late`, "late in the trick", is in the three widest vocabularies. It is missing at 16×8 and 8×4,
  where both of its facets, `last` and `third`, have their own units. It returns at 4×2 and 2×1,
  where they do not fit. That is rung 4 of the ladder: one concept and its facets, coarse when the
  room is short. This reading came after the grade; #200's family statistic is where to test it.
- **Prediction 4 broke on nets that did not learn to play.** 4×2 seeds 1 and 2 close 0.21 and 0.19
  of the gap, no more than two neurons do. Seed 0 closes 0.46 and is clean at 0.778. The ladder
  stops at rung 3, bad learning: in two of three tries, six neurons fell into a poor solution.
  With 8 or 9 live poles, a single pole moves the share by 0.11–0.13. Nothing here needs
  superposition. A net that did not learn the play did not learn its words either.
- **Two neurons do not play.** At 2×1 the mean is 0.28, and the best seed reaches 0.42. Prediction
  3 was set too high. The play needs about a dozen neurons to close 0.6 of the gap, and 24 to
  close 0.68.

Caveats:
- concept list v1 only (v2 is #197's);
- the vocabulary rule needs two of three seeds;
- for 4–5 live poles, the shares at 2×1 are near noise;
- correlation on top activations, not use (#198).

## Links

[[lem-squeeze-2026-10-16]] · [[lem-formation-2026-10-17]] · [[lem-width-tree-2026-10-17]] ·
[[clean-neurons-42]] · [[lem-the-teacher]]
