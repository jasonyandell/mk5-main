---
title: "Already-clean test 6: the squeeze on pour A — do narrow nets keep the declaration, and does each declaration buy its own units? (Lem, #199)"
kind: experiment
first_seen: 2026-10-20
last_updated: 2026-10-20
status: complete
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

Run 2026-10-20 as registered. Evidence is `lem/squeeze-pour-a-2026-10-20/` in
[texas-42-zeb-research-2026-10](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/de135b959d0933ddd5f62b9353c6354602858d10/lem/squeeze-pour-a-2026-10-20):
- the 12 trained and 12 untrained field-v3 nets with their training records;
- 48 audits, conditioned and plain;
- `grade.py`, `grade.txt`, `grade.json` and the runner.

Shuffled labels gave 0.000 on all 20 shuffles of 16×8 seed 2.

| width | neurons | validation regret (3 seeds) | **conditioned** clean, seeds 0 / 1 / 2 | mean | floor | plain mean (floor) |
|---|---:|---|---|---:|---:|---|
| 128×64 | 192 | 0.0085–0.0087 | 0.469 / 0.532 / 0.509 | **0.504** | 0.132 | 0.648 (0.214) |
| 32×16 | 48 | 0.0120–0.0123 | 0.633 / 0.681 / 0.593 | 0.636 | 0.083 | 0.772 (0.201) |
| 16×8 | 24 | 0.0143–0.0147 | 0.660 / 0.696 / 0.522 | **0.626** | 0.042 | 0.762 (0.236) |
| 8×4 | 12 | 0.0169–0.0176 | 0.739 / 0.542 / 0.750 | 0.677 | 0.139 | 0.844 (0.208) |

The declaration words in each width's vocabulary (a concept clean in at least 2 of 3 seeds,
conditioned; the plain read gives the same sets):

| width | pip trump | doubles trump / no trump |
|---|---|---|
| 128×64 | `trump-is-blanks`, `trump-is-fives`, `trump-is-fours` | `trump-is-doubles` |
| 32×16 | `trump-is-fives`, `trump-is-fours` | `trump-is-doubles` |
| 16×8 | `trump-is-fives`, `trump-is-fours` | `trump-is-doubles` |
| 8×4 | none | **`trump-is-doubles`** |

**The predictions, graded.**

| | measured | |
|---|---|---|
| 1. 24 neurons cleaner than 192, conditioned | 0.626 against 0.504 | holds |
| 2. the 8×4 vocabulary keeps a declaration word | `trump-is-doubles` | holds |
| 3. doubles/no trump at full width, and lost no later than the pip trumps | doubles is at full width, but it outlasts every pip trump; `no-trump-game` never appears | **wrong** |
| 4. every width ≥ its floor + 0.15 | +0.372 to +0.584 | holds |

**Reading.**

- **The squeeze holds with every declaration.** On pour A, conditioned on the play count, narrower
  is cleaner: 0.504 at 192 neurons, 0.626 at 24 and 0.677 at 12, against floors of 0.04–0.14.
  The play costs what it costs: validation regret doubles from 0.0086 to 0.0172.
- **Doubles trump is the declaration a narrow net keeps.** I predicted the rarer declarations
  would buy their own units and lose them first. Instead `trump-is-doubles` is in every width's
  vocabulary down to 12 neurons, after the pip trumps are gone. Even the two common pip trumps,
  fours and fives (0.313 and 0.281 of pour A's 250,000 deals), are lost at 12 neurons, while doubles
  trump (0.068) is kept.
  - Doubles trump is the declaration that most changes the suit algebra: every double becomes a
    trump and leaves its suit. A unit for it changes the meaning of every other input, and a pip
    trump does not.
  - This reading is mine, after the grade. A causal test on the doubles units is how to check it.
  - The planner's own trouble at doubles ([[ch9-at-the-table-2026-10-17]], #201) is a separate
    measurement, and this result does not explain it.
- **No trump never gets a clean unit,** at any width, in either read. It is either spread across
  units the list names otherwise, or not named by `no-trump-game`. On the ladder that is rung 2,
  the vocabulary, before anything else.
- **Pour A's shares are higher than the pip-only squeeze's**, 0.504 against 0.406 at 192 neurons,
  conditioned. The rows and declarations differ, so the comparison is descriptive.

Scope: the field-v3 recipe on pour A (the book declares, with a forced share), one position
against uniform beliefs, no history, list v2. What we learned from this model, not all of 42, yet.

## Links

[[lem-squeeze-conditioned-2026-10-20]] · [[lem-field-v3-audit-2026-10-16]] · [[lem-pour-a-2026-10-16]] ·
[[lem-squeeze-2026-10-16]] · [[clean-neurons-42]]
