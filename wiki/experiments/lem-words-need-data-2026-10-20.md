---
title: "Already-clean test 1d: the words need data — the observer's units at 5k, 50k and 500k deals (Lem, #228)"
kind: experiment
first_seen: 2026-10-20
last_updated: 2026-10-20
status: active
---

Part of THE ALREADY-CLEAN TEST (#193; [[clean-neurons-42]]). Read conditioned on the play count,
two runs of the same architecture differ sharply:
- [[lem-overbake-units-2026-10-20]]: on 3,988 training deals the observer never formed units
  named above the floor;
- [[lem-conditioned-audit-2026-10-20]]: Gus `pretrained-v0` on 500,000 deals formed them after
  about 10,000 steps.

This test adds the middle point: the data-scale specimen of #207, the same architecture, init seed
and tokens, trained on 39,963 deals for 20,000 steps.

Scope: this architecture, these inputs (one seat's original hand and the public plays of this
hand), list v2, conditioned on the play count. An unnamed unit is "not named in list v2".

## Pre-registration

- **Checkpoints**, read-only from [`lem/observer-data-scale-2026-10-17` @ 93ea96ca](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/93ea96ca04d305ee215f693b5f1ee0e492dbf89a/lem/observer-data-scale-2026-10-17),
  local copy in Codex's scratch, hashes checked before any forward pass:
  - `initial.pt` (step 0);
  - steps 1,000, 2,000, 3,000, 4,000, 5,000, 7,000, 10,000, 13,000 and 16,000;
  - `best.pt` (step 18,500, validation best);
  - `latest.pt` (step 20,000).
- **Floor.** `initial.pt` and untrained twins under torch seeds 1 and 2.
- **Rows, labels and instrument.** As in [[lem-overbake-units-2026-10-20]]: 43,927 test-split rows
  of pour A, list v2, `gus-audit --condition n_plays`, with 20 shuffles at step 20,000.
- **The other two points, already measured on the same rows and instrument** (conditioned share):

  | step | 4k deals | 500k deals |
  |---:|---:|---:|
  | 5,000 | 0.039 | 0.069 |
  | 10,000 | 0.022 | 0.135 |
  | 20,000 | 0.019 | 0.237 |

- **Predictions.**
  1. **In between.** At step 20,000 the 50k share is strictly between 0.019 and 0.237.
  2. **Words, not just play.** At the validation best (18,500) the 50k share is at least the floor
     mean + 0.05.
  3. **Ordered at matched steps.** At each of steps 5,000, 10,000 and 20,000, the 50k share lies
     between the 4k and the 500k shares, for at least 2 of the 3.
  4. **Family kept.** If at least 20 units are clean with a Silas label at the end, family kept is
     at least 0.6 and at least twice the null. If fewer, the prediction is void.
- **Cost.** 14 audits of 43,927 rows on the Mac's CPU through `scripts/cpu-slot`, about 15 seconds
  each.

## Results

Not yet run.

## Links

[[lem-overbake-units-2026-10-20]] · [[lem-conditioned-audit-2026-10-20]] ·
[[lem-observer-data-scale-2026-10-17]] · [[gus-tiny-gpt]] · [[clean-neurons-42]]
