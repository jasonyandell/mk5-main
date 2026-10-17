---
title: "Already-clean test 3: the drop-out order — which of the teacher's concepts survive as the field net narrows (Lem, #196)"
kind: experiment
first_seen: 2026-10-17
last_updated: 2026-10-17
status: active
---

Part of THE ALREADY-CLEAN TEST (#193; [[lem-the-teacher]]). The squeeze ([[lem-squeeze-2026-10-16]])
ran one seed at most widths, and its 16×8 seeds scattered from 0.65 to 0.83. Formation
([[lem-formation-2026-10-17]]) found an untrained floor of 0.15–0.28 for the clean share. This
test puts three seeds at every width, adds two narrower widths, and measures the floor at each.
The prize is the order in which the teacher's concepts drop out as the net narrows. The last
concepts standing are what makes 42 tick: the game's vocabulary, ranked by how much play each
word carries. Registered before any new net was trained.

## Pre-registration

- **Nets.** The squeeze's recipe (five pip pours, margin, MLX, 60 epochs) at **128×64, 128×8,
  32×16, 16×8, 8×4, 4×2 and 2×1**, seeds **0, 1, 2**. The squeeze's seed-0 nets and its 16×8
  seeds 1 and 2 are reused, being the same recipe; the rest are trained new.
- **The untrained floor.** The same seven widths and three seeds at zero epochs: the initial net,
  quantized by the trainer.
- **Measures per net.** The sealed audit (48 v1 labels, 20 shuffles); validation regret and the
  share of the gap from random play to the teacher closed; for the trained nets at 32×16 and wider,
  the #191 split-half on the clean share.
- **The vocabulary of a width.** The Silas concepts that label at least one clean pole in at
  least two of the three seeds. A concept's **survival width** is the narrowest width whose
  vocabulary still holds it. The drop-out order is the concepts sorted by survival width.
- **Predictions, before training.**
  1. **Nesting.** For at least 0.8 of the concepts in any width's vocabulary, the concept is also
     in the vocabulary of every wider width.
  2. **The last words.** The 2×1 vocabulary, if it has any word, includes at least one of
     `lead`, `last`, `partner-winning` and `winner-cannot-be-beaten`.
  3. **Two neurons still play.** At 2×1 the net closes at least 0.40 of the gap from random play
     to the teacher (mean over seeds).
  4. **Above the floor.** Every trained net's clean share exceeds its width's untrained floor
     (mean over that width's three untrained seeds) by at least 0.20.
  5. **Seed scatter.** At 32×16 and wider, the three seeds' clean shares span less than 0.15.
- **Cost.** 15 new trainings at about 9 minutes each, about 2.3 hours on the Mac's GPU, from one
  dataset build per call. The 21 untrained nets take seconds. About 36 audits at 1–3 minutes each.

## Results

Not yet run.

## Links

[[lem-squeeze-2026-10-16]] · [[lem-formation-2026-10-17]] · [[lem-width-tree-2026-10-17]] ·
[[clean-neurons-42]] · [[lem-the-teacher]]
