---
title: "Already-clean test 1c: what overfitting does to a tiny GPT's units — the overbake specimens (Lem, #207)"
kind: experiment
first_seen: 2026-10-20
last_updated: 2026-10-20
status: complete
---

Part of THE ALREADY-CLEAN TEST (#193; [[clean-neurons-42]]), and the interpretability handoff of
#207 ([[lem-observer-overbake-2026-10-17]]). The overbake observer is Gus's architecture (107,392
parameters), trained on only 3,988 of pour A's first 5,000 deals. It continued past its
validation best (step 4,000) to step 200,000. Training loss kept falling while validation loss
rose to 2.739, so it memorized. The essay's ladder puts narrow data on rung 3. The question is
what memorizing does to the units: do the general words keep forming past the validation best,
or does memorization wear them away?

Scope: this net, these inputs (one seat's original hand and the public plays of this hand, no
cross-hand history), list v2, read conditioned on the play count ([[lem-conditioned-audit-2026-10-20]]).
An unnamed unit is "not named in list v2".

## Pre-registration

- **Checkpoints, read-only from the handoff's frozen evidence** ([`lem/observer-overbake-2026-10-17` @ 035af17d](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/035af17da11b3749c68a7e959ac2a6301a3cf667/lem/observer-overbake-2026-10-17), local copy in Codex's scratch):
  - `initial.pt` (step 0, the saved untrained twin);
  - `best.pt` (step 4,000);
  - steps 5,000, 10,000, 20,000, 40,000, 80,000, 120,000 and 160,000;
  - `latest.pt` (step 200,000).
- **The floor.** `initial.pt` plus two untrained twins, `Observer(Config)` under torch seeds 1 and
  2.
- **Rows.** The same 43,927 rows as [[lem-gus-formation-2026-10-20]]: pour A's G4 sample on the
  deals that `lem.observer.data.partition` puts in **test**. The split is the same hash for every
  observer, so none of these deals trained this net. Labels are list v2.
- **Instrument.** `lem.probe gus-audit --condition n_plays`, with 20 shuffles at step 200,000.
- **Predictions.**
  1. **Learned by the validation best.** At step 4,000 the conditioned clean share is at least the
     floor mean + 0.03.
  2. **The words keep forming past the best.** At step 200,000 it is above step 4,000's.
  3. **Memorization wears some away.** At step 200,000 it is at least 0.03 below the series
     maximum over steps 4,000–200,000.
  4. **Less than the full corpus.** At step 200,000 it is below `pretrained-v0`'s conditioned
     0.275, which was trained on 500,000 deals.
  5. **Family kept.** Over the series, family kept is at least 0.6 and at least twice the null.
- **Cost.** 12 audits of 43,927 rows on the Mac's CPU through `scripts/cpu-slot`, about 15 seconds
  each.

## Results

Run 2026-10-20 as registered. The ten checkpoints were checked against the handoff's
`SHA256SUMS` before any forward pass, and all matched. Evidence is
`lem/overbake-units-2026-10-20/` in
[texas-42-zeb-research-2026-10](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/028b83fae7ef8e14fac61cfe75fc0b778ab4aee1/lem/overbake-units-2026-10-20):
12 audits, `grade.py`, `grade.txt` and `grade.json`. Shuffled labels gave 0.000 on all 20 shuffles
at step 200,000.

| step | conditioned clean share | Silas / raw | top-1 agreement with the recorded move |
|---:|---|---:|---:|
| untrained, seeds 42 / 1 / 2 | 0.026 / 0.009 / 0.008 | | 0.035 / 0.034 / 0.031 |
| 4,000 (validation best) | 0.034 [0.015, 0.075] | 9 / 26 | **0.527** |
| 5,000 | 0.039 [0.015, 0.069] | 9 / 31 | 0.533 |
| 10,000 | 0.022 [0.005, 0.043] | 1 / 22 | 0.500 |
| 20,000 | 0.019 [0.008, 0.042] | 0 / 19 | 0.475 |
| 40,000 | 0.013 [0.006, 0.036] | 0 / 13 | 0.446 |
| 80,000 | 0.019 [0.003, 0.037] | 2 / 17 | 0.445 |
| 120,000 | 0.019 [0.006, 0.033] | 0 / 19 | 0.427 |
| 160,000 | 0.019 [0.007, 0.036] | 0 / 19 | 0.436 |
| 200,000 (overbaked) | 0.015 [0.006, 0.036] | 1 / 14 | 0.428 |

**The predictions, graded.**

| | measured | |
|---|---|---|
| 1. step 4,000 ≥ floor + 0.03 (0.044) | 0.034 | **wrong** |
| 2. step 200,000 above step 4,000 | 0.015 against 0.034 | **wrong** |
| 3. step 200,000 at least 0.03 below the series maximum (0.039) | 0.024 below | **wrong**, narrowly |
| 4. step 200,000 below `pretrained-v0`'s 0.275 | 0.015 | holds |
| 5. family kept ≥ 0.6 and ≥ 2× null | one unit clean with a Silas label at the end | not gradeable (ill-posed) |

**Reading.**

- **The words need data.** On 3,988 deals the observer agrees with the recorded move on 0.53 of
  held-out decisions at its validation best. That is close to the 500,000-deal Gus at the same
  step (0.578 at step 5,000). But its units never rise much above the untrained floor: 0.034 to
  0.039 against 0.014, with 9 Silas-named units at most.
  - Gus on 500,000 deals formed its words after about 10,000 steps (0.135 at step 10,000, 0.237 at
    20,000).
  - This net's validation best comes at step 4,000, before any words have formed.
  - Training on keeps lowering the training loss and memorizes. Held-out agreement falls to 0.43,
    the conditioned share stays at the floor, and the few clean units are raw `gone` bits.
- **What this says on the ladder.** It is rung 3, narrow data. The small-data net's units are not
  named in list v2. Nothing here says they are mixed. A memorizing net could hold deal-specific
  features that no general vocabulary names, and that is the data gap the essay describes. My
  predictions assumed the general words would form first and erode later. They never formed.
- **The third point** is the 50,000-deal data-scale specimen of #207, filed as #228:
  [[lem-words-need-data-2026-10-20]].

Scope: this architecture, these inputs, list v2, conditioned on the play count.

## Links

[[lem-observer-overbake-2026-10-17]] · [[lem-gus-formation-2026-10-20]] · [[lem-conditioned-audit-2026-10-20]] ·
[[gus-tiny-gpt]] · [[clean-neurons-42]]
