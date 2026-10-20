---
title: "Already-clean test 1d: the words need data — the observer's units at 5k, 50k and 500k deals (Lem, #228)"
kind: experiment
first_seen: 2026-10-20
last_updated: 2026-10-20
status: complete
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

Run 2026-10-20 as registered. All twelve checkpoints matched the published `SHA256SUMS` before
any forward pass. The three runs share one seed-42 initialization: their step-0 audits are
identical. Evidence is `lem/words-need-data-2026-10-20/` in
[texas-42-zeb-research-2026-10](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/450e3b14bec43dbc0bb07de48acd4ec5c1b0c2a2/lem/words-need-data-2026-10-20):
14 audits, the published sums, `grade.py`, `grade.txt` and `grade.json`. Shuffled labels gave
0.000 on all 20 shuffles at step 20,000.

**One initialization, three amounts of data** (conditioned clean share; untrained floor 0.014):

| step | 4k deals ([[lem-overbake-units-2026-10-20]]) | **50k deals** | 500k deals (`pretrained-v0`) |
|---:|---:|---:|---:|
| 1,000 | | 0.029 | 0.046 |
| 2,000 | | 0.061 | |
| 4,000 | 0.034 (validation best) | 0.067 | |
| 5,000 | 0.039 | **0.075** | 0.069 |
| 10,000 | 0.022 | **0.111** | 0.135 |
| 13,000 | | 0.135 | |
| 16,000 | | 0.144 | |
| 18,500 | | 0.167 (validation best) | |
| 20,000 | 0.019 | **0.173 [0.128, 0.238]** | 0.237 |
| 200,000 | 0.015 | | 0.275 |

The 50k run's top-1 agreement with the recorded move is 0.573 at step 5,000 and 0.635 at
step 20,000. At step 20,000 its 175 clean units are 93 by a Silas concept and 82 by a raw bit. The
largest groups are `x:count:14` 32, `trump-is-fives` 27, `late` 22 and `x:count:20` 11.

**The predictions, graded.**

| | measured | |
|---|---|---|
| 1. step 20,000 strictly between 0.019 and 0.237 | 0.173 | holds |
| 2. at the validation best, ≥ floor + 0.05 (0.064) | 0.167 | holds |
| 3. between the 4k and 500k shares at matched steps, ≥ 2 of 3 | 10,000 and 20,000 yes; 5,000 no (0.075 against 0.039 and 0.069) | holds (2 of 3) |
| 4. family kept ≥ 0.6 and ≥ 2× null (45 units ≥ 20) | **0.978** against 0.233 | holds |

**Reading.**

- **The words come with data.** Started from one initialization, the observer's nameable units
  at step 20,000 are:
  - 0.019 on 4k deals;
  - 0.173 on 50k deals;
  - 0.237 on 500k deals (0.275 after ten times more steps).

  The 4k run never left the floor. The 50k run starts forming words by step 2,000 and is still
  climbing at its end. The 500k run climbs fastest after step 5,000.
- **Early on, data size barely matters.** At step 5,000 the 50k run is slightly ahead of the 500k
  run (0.075 against 0.069), both well above the 4k run. The separation comes later, when the 50k
  run's growth slows and the 500k run's continues. With only 4k deals, the net turned to memorizing
  before words could form. With 50k it forms them, more slowly.
- **The words keep their family as they form,** 0.978 against 0.233, as in the 500k run (0.981).
  More data adds words without mixing them.
- **What this says on the ladder.** Narrow data (rung 3) is the difference between a net that
  plays without nameable units and one whose units carry the game's words. The same architecture
  and the same start, given more of the game, grows a vocabulary.

Scope: this architecture, these inputs (within-hand history, no cross-hand history), list v2,
conditioned on the play count. What we learned from this model, not all of 42, yet.

## Correction, 2026-10-20: family kept

"Family kept" is compared with independent draws from the final labels. That null misses
continuity: a unit's label persists between checkpoints for reasons that have nothing to do with
concepts, and most of family kept is the same label staying. When a label does change, a
co-occurring concept, often in the same family, is the likeliest replacement
([[lem-family-statistic-2026-10-20]]). Read the number as "units keep their label as they form",
which is expected, and not yet as growth along a concept's facets. The re-check against a
shuffled-label run and a co-occurrence-weighted null is #229.

## Links

[[lem-overbake-units-2026-10-20]] · [[lem-conditioned-audit-2026-10-20]] ·
[[lem-observer-data-scale-2026-10-17]] · [[gus-tiny-gpt]] · [[clean-neurons-42]]
