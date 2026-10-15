---
title: "Demon power: compact exact search and 1,600 distinct worlds"
kind: experiment
first_seen: 2026-10-14
last_updated: 2026-10-14
status: complete
---

**STRATEGY-FUSION FENCE.** The [[the-demon]] is a sanctioned opponent only.
No values, labels or decisions from this study enter Walt, Zeb or Silas policies.

## Question and design

After [[demon-search-tables-2026-10-14]], Jason asked for more strength through
transposition tables and dynamic programming. This pass separates two questions:

- Can exact search do less work on the same worlds, with identical values?
- Does a larger, distinct sample improve complete-game play against the compact
  100-world baseline?

The strength test fixed 128 duplicate pairs (256 games to seven marks) before the
first game. Both arms use the arena's heuristic bidder and margin tiebreak. The
challenger draws 1,600 distinct worlds; the baseline is the 100-world policy of
`18ac096d`, rebuilt from unchanged relevant source at `7f99fa24`. Pair seeds are
`0xD3666200 + i`, for `i=0..127`, passed through `FaceoffSpec.base_seed(0)`; both
seatings play, every game is kept, and interrupted games resume at hand boundaries.
The primary statistic is mean paired win fraction against 50%; the secondary is
paired marks margin per game. Intervals resample whole pairs, 20,000 bootstrap
replicates, seed 666. Execution later runs up to three independent pairs inside
each permitted `cpu-slot` batch; the frozen binaries and deals do not change.

This is a joint test of sample count and distinct sampling, with unequal compute.
It does not separate their effects, establish strength against Zeb or Walt, or
supply a matched-cost comparison. It uses fresh seeds, independent of the earlier
20-pair pilot. [Issue #165](https://github.com/jasonyandell/mk5-main/issues/165)
tracks the wider comparison.

## Exact search

Silas's double-dummy reference ([[silas]]) already caches **future** points, independent
of banked scores and the bid, and answers make-or-set with a null window. The pass
keeps that invariant and adds:

1. **A 31-bit transposition key.** With tile ownership fixed, the remaining pool
   determines all four hands. Its 28 bits, two leader bits and one declaring-side
   bit identify a trick-start state exactly. If a later call assigns a known tile
   to another hand, the table is cleared before that world is searched. Hash
   collisions still compare the full key.
2. **Interchangeable plays.** Two tiles of one hand may share a branch only if
   their count, led context and follow membership match and their rank comparisons
   against every tile outside that hand, including the open trick, agree in every
   context. Swapping their names preserves every legal continuation and trick
   result: their owner can play only one tile in a trick. This prunes a known
   world's actions; it does not merge hidden worlds or claim an information-state
   guarantee.
3. **Forced work.** Single legal moves bypass sorting; the last trick is resolved
   directly. Remaining-point bounds apply inside an open trick too.
4. **Shared rules.** Independent world solvers share one immutable set of rule and
   interchangeability tables, with separate search tables and node counters.
5. **Exact margins by threshold queries.** Bisect the integer interval from zero
   to the points remaining. A null-window search proves which side of the
   threshold contains the value; its fail-soft bound narrows the interval. Repeat
   with the same table until the bounds meet. No approximation or depth cutoff
   enters the answer.

The reused-table regression changes owners of the same pool, declaring side,
banked score, bid and open prefix, and checks every root value against unpruned
minimax. Existing exhaustive endings, a whole-tree reference, seat rotations and
the Demon's independent Zeb-rules referee remain gates.

## Distinct sampling through the count DP

The fiber's completion counts support un-ranking: world number `k` can be read
without listing its predecessors. Floyd's algorithm draws a uniform subset of
ranks; integer rejection sampling avoids modulo bias, and insertion order is
kept separately from hash-table iteration. At or below the requested budget,
the whole fiber is enumerated. An empty fiber is refused.

Every world has the same inclusion probability. For a fiber of size `T > 1` and
`n < T` draws, the variance of a fixed world's-value sample mean, relative to
independent sampling with replacement, is multiplied by `(T-n)/(T-1)`. This is
an estimation guarantee, not a guarantee of better PIMC play. The small-case test
enumerates every draw sequence for choosing three ranks from five and finds
all ten subsets equally often; further tests check distinct, legal, deterministic
worlds and exact enumeration.

## Measurements and receipts

**Search performance, identical worlds.** Against the compact `18ac096d` policy,
rebuilt at `7f99fa24`, the final solver with the original sampler preserves every
move, make count, margin and exactness flag on 840 positions at 400 worlds.
CPU is 29.839 → 25.914 s (13.2% less); peak RSS is 187.171 → 55.378 MB (70.4% less).
The paired elapsed reduction is 13.4%, with a deal-bootstrap 95% interval of
[9.7%, 17.1%]; search calls fall 18.7%. These are whole random-hand replays, with
all ten declarations and alternating execution order. CPU and RSS are single-run
resource counters; the elapsed interval measures variation across deals, not
repeated-run uncertainty on this shared Mac.

**Ablations.** Scoped keys alone save 2.6% CPU and 69.1% RSS on the 100-world corpus.
Removing interchangeable-play pruning makes the otherwise identical distinct-world
candidate 4.0% slower (paired elapsed 95%: 1.3–6.5% slower). Threshold queries save
another 11.0% CPU on that corpus, then 8.4% on 60 fresh deals / 1,680 positions
(paired elapsed saving 8.3% [5.7%, 10.8%]), with every move and value preserved.
The earlier forced-trick, no-bisection variant saves 8.1% CPU at 100 worlds and
9.1% at 400. The frozen strength binary uses that earlier exact solver; the final
threshold solver changes its cost, not its policy, and is checked separately.

**Strength: no demonstrated gain.** All 128 pairs finished: 256 games and 2,886
hands, no failed or dropped games. The 1,600-world challenger wins **124–132**,
48.44% [44.14%, 52.73%] against the 50% chance baseline. Its paired marks margin is
+0.0078 per game [−0.2266, +0.2461]. This does not establish improvement or
inferiority; it is also not an equivalence test.

**Cost of the frozen strength policy.** Across the complete games, native play CPU
is 5,712.725 s for distinct 1,600 versus 497.042 s for compact 100: 22.315 versus
1.942 s per game, **11.49×**. The processes answer 40,404 decisions each and solve
27,168,120 versus 2,113,195 worlds; forced moves and full-fiber endings make the
realized ratio smaller than the configured 16×. Peak per-process RSS is 273.465
versus 95.928 MB. These are direct resource counters for the frozen pre-bisection
strength binary, not a matched-cost test or a measurement of the final solver's
whole-game CPU.

The engineering gains are retained. The 1,600-world pin is an experimental
larger-budget opponent, not a proven promotion over 100 worlds. More uniform
worlds alone did not deliver the requested strength gain in this test. Bidding,
record-based beliefs, a fixed-world sampler comparison and a matched-cost test
remain open under #165; the current bidder was held fixed here.

**Validation.** Full checks before and after threshold search are GREEN (17 suites
passed; CUDA forge unavailable on this Mac; retired Burl not run). After rebasing
on main, all 10 affected suites pass; the roster/fence/wiki checks pass too. The
release build reproduces all 1,680 held-out decisions of the frozen strength
policy. Both new participant records are downloaded and byte-checked, and the
1,600-world binary is rebuilt successfully from its published source trees.

Complete raw games, paired replay records, build provenance, analysis scripts and
validation logs are [public on Hugging Face](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/d18d1700f78e89cba85c8dbaf7e8b673f47f2554/demon/2026-10-14-power)
(CC0, 688 files, 11.8 MB). Commit `d18d1700f78e89cba85c8dbaf7e8b673f47f2554` is immutable. A fresh download
of `report.json` matches SHA-256 `bfc42f32ab1c8ce9177203bd54a8c8ec793fa9471ed218ef93c179e573d5dd67`; a raw game is also
freshly downloaded and checked against the published `SHA256SUMS`.

The registered participants are
[`demon-pimc-n100-228812ff`](https://huggingface.co/datasets/jasonyandell/texas-42-tournament/blob/382c4233b1d0489d63474bd043cb707b9f93abe0/participants/demon-pimc-n100-228812ff.json)
and
[`demon-pimc-n1600-228812ff`](https://huggingface.co/datasets/jasonyandell/texas-42-tournament/blob/382c4233b1d0489d63474bd043cb707b9f93abe0/participants/demon-pimc-n1600-228812ff.json).
Their source commit is [[228812ff]]. Every earlier participant is unchanged.

## Links

[[the-demon]] · [[demon-search-tables-2026-10-14]] · [[silas]] ·
[[scenario-player]] · [[strategy-fusion-fence]] · [[tournament]] ·
[[3cd6ae53]] · [[228812ff]]
