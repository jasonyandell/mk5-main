---
title: "Already-clean test 3b: is narrower still cleaner with position taken out? The squeeze and the drop-out order, conditioned (Lem, #227)"
kind: experiment
first_seen: 2026-10-20
last_updated: 2026-10-20
status: active
---

Part of THE ALREADY-CLEAN TEST (#193; [[clean-neurons-42]]). The program's headline, 24 neurons
cleaner than 192, was read unconditioned ([[lem-squeeze-2026-10-16]],
[[lem-dropout-order-2026-10-17]]). [[lem-conditioned-audit-2026-10-20]] found that the field net's
units carry position: conditioned on the play count, the 128×64 share falls from 0.638 to 0.406,
and its floor from 0.312 to 0.077. A narrow net could carry a different share of position than a
wide one, so the headline is open again. If narrower is not cleaner here, that is the finding, and
CLAUDE.md's sentence changes through the Mayor.

Scope: the field net's recipe on pip pours, one position against uniform beliefs, no history,
list v2.

## Pre-registration

- **Nets.** The 21 trained drop-out nets and their 21 untrained twins: 128×64, 128×8, 32×16, 16×8,
  8×4, 4×2 and 2×1, three seeds each.
- **Instrument.** `lem.probe audit --condition n_plays` on the 434,535-row audit sample, list v2
  (`labels_shipped_v2`). It is the sealed audit otherwise, with 20 shuffles on 16×8 seed 2. The
  three 128×64 trained and untrained reads are [[lem-conditioned-audit-2026-10-20]]'s, the same
  instrument.
- **Comparison.** The unconditioned v2 reads of [[lem-naming-2026-10-17]].
- **Predictions.**
  1. **Narrower is still cleaner.** The conditioned clean share's mean at 16×8 (24 neurons) is
     above its mean at 128×64 (192 neurons).
  2. **Above the floor.** At every width from 8×4 up, the trained conditioned mean is at least its
     width's conditioned untrained mean + 0.15.
  3. **The drop-out order nests.** With #196's vocabulary rule (a Silas concept clean in at least
     two of three seeds), at least 0.8 of the concepts in any width's conditioned vocabulary are
     in every wider width's.
  4. **The narrow words.** The 8×4 conditioned vocabulary contains at least one of
     `partner-winning`, `holds-boss-trump`, `no-trump-held`, `trumps-3plus` and
     `count-on-trick`, words that the play count does not fix.
- **Cost.** 36 new audits of 434,535 rows on the Mac's CPU through `scripts/cpu-slot`, about 1–3
  minutes each.

## Results

Not yet run.

## Links

[[lem-conditioned-audit-2026-10-20]] · [[lem-squeeze-2026-10-16]] · [[lem-dropout-order-2026-10-17]] ·
[[lem-naming-2026-10-17]] · [[clean-neurons-42]]
