---
title: "Already-clean test 1c: what overfitting does to a tiny GPT's units — the overbake specimens (Lem, #207)"
kind: experiment
first_seen: 2026-10-20
last_updated: 2026-10-20
status: active
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

Not yet run.

## Links

[[lem-observer-overbake-2026-10-17]] · [[lem-gus-formation-2026-10-20]] · [[lem-conditioned-audit-2026-10-20]] ·
[[gus-tiny-gpt]] · [[clean-neurons-42]]
