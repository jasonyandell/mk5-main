---
title: The seed-varied field — #117's auction read with the planner seed in the sampling (#190, 2026-10-16)
kind: experiment
first_seen: 2026-10-16
last_updated: 2026-10-16
status: complete
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
two offsets give an A/A (on each deal a fixed pairing, seed 16 + k against seed k; see Results). At a pool of one it is the plain player, byte for byte. This study
re-runs #117's arms against a seed-varied field.

## Results (complete, 2026-10-16; 352 of 352 registered batches ran, none dropped)

**One line.** The A/A failed again, in the other direction. A plain-Zeb seat at independent pool
seeds reads −0.14 [−0.25, −0.03] marks/game against the seed-varied field, and so, by the
registered bar, K = +0.12 [+0.00, +0.23] is reported, not claimed. The diagnostic that was
meant to explain #117's A/A does not: no planner seed shows an effect of its own. So
planner-seed scatter is **not supported** as the reason A/A arms miss, and the estimator's
calibration is the open question (#191).

Data: [texas-42-zeb-research-2026-10](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10)
`book/seed-varied/` (per batch `meta.json` and `games.jsonl`, `summary.json`, the queue log, `CARD.md`).

### The registered contrasts (paired by game pair; MDE at 80% power), beside #117

| contrast | here [95%] | #117 (one-seed field) | pairs | SD/pair | MDE |
|---|---|---|---|---|---|
| **primary K, the read by a known method** | **+0.12 [+0.00, +0.23] marks/game** | +0.17 [+0.06, +0.29] | 576 | 1.40 | 0.16 |
| U, the read by an unknown method | +0.06 [−0.05, +0.17] | +0.07 [−0.04, +0.18] | 576 | 1.35 | 0.16 |
| K − U | +0.06 [−0.08, +0.20] | +0.10 [−0.03, +0.23] | 576 | 1.69 | 0.20 |
| Za, the A/A | **−0.14 [−0.25, −0.03]** | +0.12 [+0.02, +0.23] | 576 | 1.39 | 0.16 |
| K − Za | +0.26 [+0.10, +0.41] | +0.05 [−0.09, +0.19] | 576 | 1.89 | 0.22 |
| to 250: PK − PZa, share of games won | +0.5 pts [−4.2, +5.2] | +2.6 [−1.8, +7.0] | 192 | 0.34 | 6.8 pts |
| to 250: PK − 50% | +4.4 pts [+1.0, +7.8] | +3.4 [+0.3, +6.5] | 192 | 0.24 | 4.9 pts |

Games to 250 won by team A: PK 54.4% [51.0, 57.8], **PZa 53.9% [50.5, 57.3]**, where the A/A
should read 50%.

### What the A/A says, and what it does not (post hoc, not registered)

- **Two A/As, opposite signs.** #117's A/A read +0.12 and this one −0.14, each about 2.3 standard
  errors from 0. If the intervals were right, both would miss with a probability of about 0.001. The
  points A/A missed too.
- **Not planner-seed scatter.** Each hand is played at the seed its deal picks, so team A's
  hand-mark margin can be split by pool position (the script is in the data folder, `diagnostics/`). Across
  the 16 positions the per-position means vary *less* than noise alone predicts, in every arm
  (SD 0.022 marks/hand against about 0.035). A seed with an effect of its own would show as extra
  spread, and there is none. The design note above also overstated the A/A's independence: Za and
  the field read the same deal key, so on each deal it is seed 16 + k against seed k, sixteen
  fixed pairings, not independent draws. Sixteen pairings with no per-seed effect still cannot
  make −0.14.
- **Not correlated batches.** Clustering the bootstrap by batch rather than by game pair barely
  widens it ([−0.27, −0.01] against [−0.25, −0.02]).
- **So the estimator itself is suspect.** Either the percentile bootstrap over game pairs
  undercovers at this size (heavy tails from the race to 7, or the paired halves), or something
  systematic separates two copies of one player at one table. #191 measures it directly: many
  independent A/A studies, counting how many 95% intervals miss 0.

### Reading, under [[book-authority]]

- **The book's read.** K leans the book's way in both studies (+0.17, +0.12), and the known
  method leans over the unknown in both (K − U = +0.10, +0.06). Neither clears its bar once the
  A/A is honest, and the A/A cannot currently be trusted to be honest. The claim stays: reading the
  auction by its known method does not cost, and its gain is not established.
- **What changes for every desk.** The caveat added to single-arm margins (#190) named planner-seed
  scatter. That mechanism is not supported, so the caveat is restated as an estimator caveat: A/A
  arms have missed by ±0.12–0.14 marks/game and by 4 points of share, cause open (#191).

### Resolution (#191, 2026-10-17)

The calibration this page called for came back clean ([[estimator-calibration-2026-10-16]]).
Twenty fixed-pair A/A replicates scatter as their intervals say (R_F 1.06, 2 of 20 misses), and
σ_seed is 0. So this page's A/A miss (−0.14) was chance, at about 2.4 standard errors, and the
estimator is not suspect. K stays "reported, not claimed" here as registered. Pooled with #117's
K on disjoint deals, exploratory: +0.14 [+0.06, +0.22] marks/game.

### Receipts

Pre-registered at dae7ee3e (`scenario/seed_varied.py`, `arena/seed_varied.py`). 352 batches
through `scripts/cpu-slot`, two at a time, 53–173 s each including the wait for a slot. Every
batch directory holds its 12 games; none dropped. Timing pilot seed 3490 excluded. Run
2026-10-16 11:20–15:27.

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
