---
title: "Already-clean test 6: the squeeze on pour A — do narrow nets keep the declaration, and does each declaration buy its own units? (Lem, #199)"
kind: experiment
first_seen: 2026-10-20
last_updated: 2026-10-20
status: active
---

Part of THE ALREADY-CLEAN TEST (#193; [[clean-neurons-42]]). The squeeze
([[lem-squeeze-2026-10-16]], [[lem-squeeze-conditioned-2026-10-20]]) trained on pip deals only.
Pour A ([[lem-pour-a-2026-10-16]]) has every playable declaration: the seven pip trumps, doubles
trump and no trump. The book declares, with a forced share for doubles and no trump. G4
([[lem-field-v3-audit-2026-10-16]]) found 32 doubles-trump units in the wide field-v3 net. This
test asks what the declaration costs as the net narrows. Do the narrow nets keep a declaration
unit? Is the narrow vocabulary the same across declarations, or does each declaration buy its own
units?

Scope: the field-v3 recipe on pour A, one position against uniform beliefs, no history, list v2.
The audit reads both ways: conditioned on the play count (the honest absolute, per
[[lem-conditioned-audit-2026-10-20]]) and unconditioned, for continuity.

## Pre-registration

- **Nets.**
  - `lem.probe squeeze-train` on pour A's two halves (`merged/pip`, `merged/dtnt`). The recipe is
    margin, MLX and 60 epochs, the same as the squeeze.
  - Widths 128×64, 32×16, 16×8 and 8×4, three seeds each (0, 1, 2).
  - Untrained twins at zero epochs, one per width and seed.
  - The field-v3 nets (`.v3.npz`) are audited.
- **Rows and labels.** The G4 sample: 434,804 held-out rows of pour A (the trainer's
  `(deal + offset) % 10 == 0`), labelled with `labels_v3_v2`.
- **Audits.** The sealed audit with `--condition n_plays` (primary) and without (secondary), with
  20 shuffles on 16×8 seed 2 conditioned. #196's vocabulary rule: a concept clean in at least 2 of
  3 seeds.
- **Predictions (conditioned unless stated).**
  1. **Narrower is still cleaner with every declaration.** The 16×8 mean clean share is above the
     128×64 mean.
  2. **The narrow nets keep the declaration.** The 8×4 vocabulary holds at least one
     `trump-is-*` concept.
  3. **Doubles and no trump buy their own units, and lose them first.** At 128×64 the vocabulary
     holds `trump-is-doubles` or `no-trump-game`, and the narrowest width that still holds either
     is no narrower than the narrowest that holds a pip `trump-is-*`.
  4. **Above the floor.** At every width the trained mean is at least the width's untrained mean
     + 0.15.
- **Cost.** 12 trainings at about 9 minutes each on the Mac's GPU (about 1.8 hours) and 48 audits
  of 434,804 rows (about 1–2 minutes each), through `scripts/cpu-slot`.

## Results

Not yet run.

## Links

[[lem-squeeze-conditioned-2026-10-20]] · [[lem-field-v3-audit-2026-10-16]] · [[lem-pour-a-2026-10-16]] ·
[[lem-squeeze-2026-10-16]] · [[clean-neurons-42]]
