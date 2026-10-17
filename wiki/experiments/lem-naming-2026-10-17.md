---
title: "Already-clean test 4: name every unit of the narrow nets with concept list v2 and the readers (Lem, #197)"
kind: experiment
first_seen: 2026-10-17
last_updated: 2026-10-17
status: active
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

Not yet run.

## Links

[[lem-dropout-order-2026-10-17]] · [[lem-squeeze-2026-10-16]] · [[lem-field-v3-audit-2026-10-16]] ·
[[lem-field-net-audit-2026-10-15]] · [[clean-neurons-42]]
