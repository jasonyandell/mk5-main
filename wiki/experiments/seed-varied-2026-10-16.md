---
title: The seed-varied field — #117's auction read with the planner seed in the sampling (#190, 2026-10-16)
kind: experiment
first_seen: 2026-10-16
last_updated: 2026-10-16
status: active
---

## Summary

Issue #190, a methods repair. A Zeb seat is one planner seed, so two seeds are two fixed,
slightly different players, and a study that seats the field at one seed and bootstraps over
deals does not see the difference. #117's A/A found it: plain Zeb at seed 1 took +0.12 [+0.02,
+0.23] marks/game from the seed-0 field ([[auction-read-2026-10-16]]), and #149 saw the same
spread ([[regulars-table-2026-10-14]]). Paired differences of two team-A arms against one field
cancel the field's seed. A single arm's margin against it does not.

`arena.seed_varied.SeedVaried` plays each hand at a planner seed its deal picks from a pool of 16.
The deal is keyed as a set of hands, so the pick is the same in every arm, in both halves of a
game pair, and at every decision. Two seats with one offset share the seed on every deal, and
two offsets give an honest A/A. At a pool of one it is the plain player, byte for byte. This study
re-runs #117's arms against a seed-varied field.

*Results follow once every registered batch has run.*

## Design (pre-registered before any measured batch; see `scenario/seed_varied.py`)

- **Players.** Every seat bids by the heuristic rule at caution 0 and plays deployed Zeb 0.3.0,
  seed-varied. Team B, the field: offset 0. Team A: **K** (the read by the known method, offset 0,
  so on every deal it differs from the field only by the read), **U** (the method unknown), **Za**
  (plain Zeb at offset 16, an independent seed per deal: the A/A, whose interval should cover 0).
- **Arms.** Marks: K, U, Za. The book's game (250 points): PK, PZa.
- **Primary.** K, read against the A/A: the read pays only if K excludes 0 above *and* Za covers 0.
  If Za excludes 0 again, K is reported, not claimed.
- **Secondaries.** U; K − U; K − Za; PK − PZa and PK − ½ on games won; each beside #117's value.
- **Size.** 576 game pairs per marks arm (seeds 3501–3596 × 6), 192 per points arm (3601–3632 ×
  6); 352 batches through `scripts/cpu-slot`. MDE on K about 0.16. Timing pilot seed 3490
  excluded (56–79 s per 4-pair batch, three times #117's: 16 seeds batch less).
- **Estimator, batch-drop rule and fence.** As [[book-small-n-2026-10-14]].

## Links

[[book-authority]] · [[book-assumptions]] · [[auction-read-2026-10-16]] ·
[[regulars-table-2026-10-14]] · [[strategy-fusion-fence]]
