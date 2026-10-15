---
title: "The Demon: smaller exact search tables, and the 400-world strength pilot"
kind: experiment
first_seen: 2026-10-14
last_updated: 2026-10-14
status: complete
---

**STRATEGY-FUSION FENCE.** [[the-demon]] is a sanctioned tournament opponent only.
Its per-world values never enter another player's policy or training data.

## Change and correctness

The Demon keeps a separate `ClairvoyantSolver` for each sampled world during the
make-count pass. It then completes all tied moves within one world and drops that
world's table before moving on. The solver, sampler, objectives, make-count ties,
margin ties and seeds are unchanged. Small tables improve locality; dropping each
world after its margin pass prevents large margin tables from accumulating.

The old shared table reused a few more states, but that saving did not pay for
its lookup and memory cost. Search nodes rise by 1.6% at 100 worlds and 3.7% at
400 worlds on this panel; CPU time falls. Computing every move's full margin up
front was also tried and rejected for excess search work.

The independent brute-force minimax test still checks all make counts and margins
on 30 enumerated fibers. It now also checks the no-margin mode against those same
counts and its lowest-tile tie rule. Whole-hand legality covers all ten declarations;
the Python seat tests exercise deterministic whole games and pinned source builds.

Source: [[18ac096d]]. Code: [`tournament/demon/src/lib.rs`](../../tournament/demon/src/lib.rs),
[`tournament/demon/tests/demon.rs`](../../tournament/demon/tests/demon.rs).

## Paired replay: speed and memory

Thirty random deals, seed 666, played by uniform legal moves yield 840 fixed requests
spanning declarations 0–9. Old and new binaries answer each request at the same sample
count, alternating which runs first. At **both 100 and 400 worlds**, the selected tile,
every make count, every reported margin, world count and exact flag agree on all
840 requests. Nodes and elapsed time are allowed to differ.

| Worlds | Baseline CPU | New CPU | CPU reduction | Baseline peak RSS | New peak RSS |
|---|---:|---:|---:|---:|---:|
| 100 | 11.214 s | 9.819 s | 12.4% | 284.7 MB | 75.2 MB |
| 400 | 40.237 s | 32.190 s | 20.0% | 604.9 MB | 222.6 MB |

CPU is total child-process user + system time for the 840 requests; RSS is the
process peak, decimal MB. On the same alternating replay, elapsed-time reduction
is 12.7% [10.3, 14.5] at 100 worlds and 24.8% [15.4, 31.0] at 400. These are 95%
percentile bootstrap intervals over the 30 hands (10,000 draws, seed 666), not
repeat-run uncertainty. Other sessions ran tests concurrently, especially during
the 400-world replay: these elapsed times are not isolated-machine latency claims.
The baseline for every speed comparison is the original implementation at the
**same** number of worlds. No matched-cost strength claim follows.

## Strength pilot

The fixed pilot compares 400 worlds against the original 100-world Demon over
20 duplicate pairs of games to seven marks, sides swapped on identical deals.
Both use the same arena `heuristic` bidder and margin tie break. Pair seeds are
`0xDE0666 + pair_index`. Every game saves its full auction, play and score record.
The chance baseline is 50%; the marks baseline is zero marks/game.

The 400-world prototype wins **21–19** of 40 games, versus a 50% chance
baseline. Its marks/game advantage is **+0.175 [-0.303, +0.653]**
(normal 95% interval over the 20 duplicate-pair means). All 40 games completed;
interruptions resumed at saved hand boundaries, with no discarded pair.

The larger sample is a challenger, not a demonstrated strength improvement.
The pilot is local research; its experimental participant ids and binary hashes
are preserved, and its games are not relabeled as production tournament games.

## Evidence and reproduction

[Data, requests, replies, games, scripts and source patch](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/71bd471fe34861a1c8a4bde6291b8edf186b8847/demon/2026-10-14-search-tables)
are CC0. `report.json` holds the aggregates; the paired replay summaries contain
binary hashes and resource usage; `manifest.json` names source revisions and the
machine. `compare.py` asserts parity before recording each paired reply.

The benchmark baseline is `a72e629f`; the sampler, rules and clairvoyant search are
unchanged from the first pinned Demon, `b88c5432`. The match uses the equivalent
compact prototype, with its own binary hash recorded; the final implementation's
paired replay follows it. The patch is sufficient to reproduce the optimization
from the named baseline. [[tournament]] retains immutable participants; new code
and a new world count receive new ids.

The full `scripts/check` gate is GREEN: 17 suites passed; CUDA-only forge is
unavailable on this Mac and retired Burl is not run by the gate. The two older
wiki routing warnings are outside this experiment; its hub route is present.
The powered, held-out strength experiment and bidder/play attribution remain
[#165](https://github.com/jasonyandell/mk5-main/issues/165).
