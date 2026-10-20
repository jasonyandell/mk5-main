---
title: "Already-clean test 3b: is narrower still cleaner with position taken out? The squeeze and the drop-out order, conditioned (Lem, #227)"
kind: experiment
first_seen: 2026-10-20
last_updated: 2026-10-20
status: complete
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

Run 2026-10-20 as registered. Evidence is `lem/squeeze-conditioned-2026-10-20/` in
[texas-42-zeb-research-2026-10](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/590e8c4f2bb79ff6635007b97419cc741a5622d3/lem/squeeze-conditioned-2026-10-20):
36 audits, `grade.py`, `grade.txt`, `grade.json` and the runner. The 128×64 reads are those of
[[lem-conditioned-audit-2026-10-20]]. Shuffled labels gave 0.000 on all 20 shuffles of 16×8 seed 2.

| width | neurons | conditioned clean share, seeds 0 / 1 / 2 | mean | conditioned floor | margin | unconditioned mean (v2) |
|---|---:|---|---:|---:|---:|---:|
| 128×64 | 192 | 0.385 / 0.420 / 0.412 | **0.406** | 0.077 | +0.329 | 0.638 |
| 128×8 | 136 | 0.452 / 0.507 / 0.445 | 0.468 | 0.082 | +0.386 | 0.684 |
| 32×16 | 48 | 0.440 / 0.441 / 0.456 | 0.445 | 0.035 | +0.411 | 0.697 |
| 16×8 | 24 | 0.562 / 0.522 / 0.562 | **0.549** | 0.014 | +0.534 | 0.759 |
| 8×4 | 12 | 0.435 / 0.364 / 0.652 | 0.484 | 0.056 | +0.428 | 0.779 |
| 4×2 | 6 | 0.111 / 0.125 / 0.333 | 0.190 | 0.167 | +0.023 | 0.574 |
| 2×1 | 3 | 0.000 / 0.000 / 0.000 | 0.000 | 0.167 | −0.167 | 0.867 |

**The predictions, graded.**

| | measured | |
|---|---|---|
| 1. 24 neurons cleaner than 192, conditioned | 0.549 against 0.406 | **holds** |
| 2. margin ≥ 0.15 at every width from 8×4 up | +0.329 to +0.534 | holds |
| 3. the conditioned drop-out order nests ≥ 0.8 | 46 of 52 = 0.885 | holds |
| 4. 8×4's vocabulary has a word the play count does not fix | `holds-boss-trump`, `no-trump-held` | holds |

**Reading.**

- **The headline survives with position taken out.** Conditioned on the play count, the field net
  retrained at 24 neurons is cleaner than at 192: 0.549 against 0.406, three seeds each, with
  floors of 0.014 and 0.077. Every width from 12 neurons up is 0.33–0.53 above its floor. The
  shares are lower than the unconditioned ones because position no longer counts. The ordering is
  the one the unconditioned audit found.
- **Below 12 neurons the conditioned read has nothing to name, by construction.** At 4×2 and 2×1
  the conditioned share falls to the floor or to zero. The unconditioned read of the same nets was
  0.574 and 0.867, against floors of about 0.17 and 0.11.
  - Those units do the play count's own job: the mover's place in the trick, the `lead` and
    `late` of [[lem-dropout-order-2026-10-17]].
  - For the field net that is a learned reading of an input bit, not an architecture's freebie,
    and a z-score within the play count cannot see a unit whose whole job is the play count.
  - Both statements are true. #196's last words are real units (unconditioned, far above their
    floor), and they are position words (conditioned, nothing left).
- **The drop-out order without position.** The last words that name something beyond the play
  count are the mover's trump: `holds-boss-trump` and `no-trump-held`. Next are `holds-master` and
  `trump-is-fives` at 8×4. Then `trump-led`, two more declaration words, and the trick's state
  (`partner-winning`, `winner-cannot-be-beaten`, `count-on-trick`) at 24–48 neurons. Voids and
  `holds-ten` come only at full width.

Scope: the field net's recipe on pip pours, one position against uniform beliefs, no history,
list v2. What we learned from this model, not all of 42, yet.

## Links

[[lem-conditioned-audit-2026-10-20]] · [[lem-squeeze-2026-10-16]] · [[lem-dropout-order-2026-10-17]] ·
[[lem-naming-2026-10-17]] · [[clean-neurons-42]]
