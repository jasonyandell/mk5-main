---
title: "Already-clean test 4: name every unit of the narrow nets with concept list v2 and the readers (Lem, #197)"
kind: experiment
first_seen: 2026-10-17
last_updated: 2026-10-17
status: complete
---

Part of THE ALREADY-CLEAN TEST (#193; [[clean-neurons-42]], [[lem-the-teacher]]). The
drop-out order ([[lem-dropout-order-2026-10-17]]) read 21 trained nets with concept list v1
(48 Silas concepts). Under v1, a quarter of the live poles at 16×8 and 8×4 are partial, meaning
their best label has precision in [0.7, 0.9), and 0.22–0.30 more are clean only by a raw input.
The essay judges a unit by its top examples, so this test reads those units two more ways:
- with concept list v2 (#185: 70 concepts, adding the declaration, the readers' conjunctions and
  off-suit voids);
- with Haiku readers, who see a pole's top cards and say what they share.

The target is every live unit of the 16×8 and 8×4 nets either named or explicitly not named in
list v2 and by the readers, so each unit has a name or a recorded gap.
Scope: this net, these inputs (one position, uniform beliefs, no player-behaviour history), and
these vocabularies. A unit nothing names is "not named in list v2 or by the readers". It goes to
the #202 inventory as a vocabulary or data gap, never as mixed.

## Pre-registration

- **Nets.** The 21 trained nets of #196, three seeds at each of 128×64, 128×8, 32×16, 16×8, 8×4,
  4×2 and 2×1, and the 21 untrained nets as the floor.
- **Sample and labels.** The audit sample of 2026-10-15: 434,535 held-out rows. Labels are list
  v2 (`labels_shipped_v2.npz`, 70 concepts, identity `2f9a3ddc…`). The audit is the sealed audit:
  top 1%, clean at precision ≥ 0.9, partial at ≥ 0.7, 20 shuffles.
- **Readers.** `lem.probe cards` then `readers`, with Haiku, 12 explain cards and 20 score cards,
  on every live pole of the 16×8 and 8×4 nets (all three seeds; about 210 poles). A pole is
  detected when the Wilson lower bound of its balanced accuracy exceeds 0.5. The null is the
  shifted description, scored by the same rule.
- **Measures, per width (pooled over seeds).**
  - the clean share under v2;
  - the share of live poles clean by a Silas concept;
  - for 16×8 and 8×4: the readers' detected share, its shifted null, the detected share among
    v2-partial and v2-unnamed poles, and the share of live poles named by neither (not clean
    under v2 and not detected).
- **Predictions, before any v2 audit or reader runs on these nets.** The v1 baselines are from
  the #196 audits.
  1. **Renamed, not added.** At every width, the clean share under v2 is within 0.05 of v1.
     G4 found that v2 renamed rather than added.
  2. **Into Silas words.** At 16×8 and 8×4, the Silas-clean share under v2 is at least
     v1 + 0.10. The v1 shares are 0.458 and 0.515; the raw-input units, mostly declaration
     detectors, should take Silas names.
  3. **The readers see them.** At 16×8 and 8×4, readers detect at least 0.70 of live poles, with
     the shifted null at or below 0.20.
  4. **The readers see what the list misses.** Among the poles that are not clean under v2 at
     16×8 and 8×4, readers detect at least 0.60.
  5. **The target.** The share of live poles at 16×8 and 8×4 named by neither is at most 0.10.
- **Cost.** 42 audits at 1–3 minutes each on the Mac, through `cpu-slot`, and about 420 Haiku
  calls.

## Results

Run 2026-10-17 as registered. Evidence: `lem/naming-2026-10-17/` in
[texas-42-zeb-research-2026-10](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/119af39d6ae2307eabf9322128902d045115b38a/lem/naming-2026-10-17).
It holds 42 v2 audits, the cards and every reader's description and marks, `grade.py` and
`grade.json`.

**Scope.** These are the field net's recipe on pip pours, read on one position, against uniform
beliefs, with no player-behaviour history. The vocabularies are list v2 (70 concepts) and Haiku
readers working from 12 cards. Every v2 share was 0.000 on all 20 label shuffles.

| width | live poles (3 seeds) | clean, v1 → v2 | clean by a Silas concept, v1 → v2 | untrained floor, v1 → v2 |
|---|---:|---|---|---|
| 128×64 | 1,114 | 0.633 → 0.638 | 0.360 → 0.446 | 0.168 → 0.312 |
| 128×8 | 805 | 0.678 → 0.684 | 0.327 → 0.422 | 0.174 → 0.321 |
| 32×16 | 274 | 0.682 → 0.697 | 0.365 → 0.489 | 0.181 → 0.354 |
| 16×8 | 142 | 0.761 → 0.761 | 0.458 → 0.606 | 0.226 → 0.389 |
| 8×4 | 68 | 0.735 → 0.779 | 0.515 → 0.603 | 0.167 → 0.292 |
| 4×2 | 26 | 0.423 → 0.577 | 0.423 → 0.577 | 0.167 → 0.333 |
| 2×1 | 11 | 0.727 → 0.818 | 0.636 → 0.818 | 0.111 → 0.278 |

The readers on the 16×8 and 8×4 nets, all three seeds:

| | 16×8 | 8×4 |
|---|---:|---:|
| live poles read | 142 | 68 |
| detected (Wilson lower bound > 0.5) | 0.796 | 0.853 |
| shifted-description null | 0.134 | 0.132 |
| poles not clean under v2 | 34 | 15 |
| of those, detected by the readers | 0.794 | 0.867 |
| **named by neither** | **0.049 (7)** | **0.029 (2)** |

**The predictions, graded.**

| | measured | |
|---|---|---|
| 1. clean share under v2 within 0.05 of v1 at every width | +0.005 to +0.044 at 12 neurons and wider; +0.154 at 4×2 and +0.091 at 2×1 (26 and 11 poles) | **wrong** at the two narrowest |
| 2. Silas-clean share +0.10 at 16×8 and 8×4 | +0.148, +0.088 | **wrong** at 8×4 |
| 3. readers detect ≥ 0.70, shifted null ≤ 0.20 | 0.796 / 0.853; 0.134 / 0.132 | holds |
| 4. readers detect ≥ 0.60 of the poles v2 does not name | 0.794 / 0.867 | holds |
| 5. named by neither ≤ 0.10 | 0.049 / 0.029 | holds |

**Reading.**

- **The target is met.** Of 210 live units in the 12- and 24-neuron nets, 201 have a name: a v2
  concept at precision ≥ 0.9, or a reader's description that detects the unit's top cards on
  held-out cards. Readers detect 0.80–0.87 of the units the list cannot name, against a 0.13
  null. Read the essay's way, by their top examples, nearly every unit of the narrow nets has one
  job.
- **The nine units named by neither.** All nine are listed in `grade.json`. In every case the
  list's best label is partial (precision 0.72–0.88), and in eight the reader offered a
  description. Seven of those eight describe the trick and the mover's seat together: "partner is
  winning the trick and I play last", "opponent is winning the trick late; you can't take it
  back", "third to play, with the trick already won". Each is a conjunction of who is winning
  with where the mover sits. The ladder stops at rung 2, the vocabulary. List v3 needs `winning
  seat × my seat`, and the readers' scoring with one 20-card sample is too coarse to confirm a
  fine conjunction. None of the nine points at the bidding, earlier tricks or the score. For
  #202 that means no history gap in these nets, and a vocabulary gap instead.
- **The floor rises with the vocabulary. This is the finding for the tools.** Under v2 an
  untrained net scores 0.28–0.39 clean, against 0.11–0.23 under v1. Trained nets move by 0.00 to
  0.04 at 12 neurons and wider. A list of 70 concepts finds more chance matches among random
  units, but adds almost nothing to trained ones (G4 found the same: renamed, not added). So the
  trained margin over the floor shrinks from 0.44–0.57 to 0.33–0.49, and every clean share must
  quote the floor for its own list. The skill's single floor figure (0.15–0.28) is a v1 number.
- **Prediction 1 failed only where a pole is a large share.** At 4×2 and 2×1, one pole is 0.04
  and 0.09 of the share. v2 named 4 more poles at 4×2 and 1 more at 2×1.
- **Prediction 2 failed narrowly.** At 8×4, v2 moved six poles from raw-input names to Silas
  names, +0.088 against the registered +0.10. At 16×8 the gain was +0.148.

Caveats:
- the readers use a single 20-card scoring sample per pole;
- the v2 list was written partly from the readers' earlier descriptions, so their agreement is not
  independent;
- correlation on top activations, not use (#198).

## Links

[[lem-dropout-order-2026-10-17]] · [[lem-squeeze-2026-10-16]] · [[lem-field-v3-audit-2026-10-16]] ·
[[lem-field-net-audit-2026-10-15]] · [[clean-neurons-42]]
