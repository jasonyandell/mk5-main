---
title: "Demon next ascent: answering Zeb 0.4.1 at comparable CPU"
kind: experiment
first_seen: 2026-10-16
last_updated: 2026-10-16
status: active
---

**STRATEGY-FUSION FENCE.** [[the-demon]] remains a sanctioned tournament
opponent. This study follows [[rematch-zeb-041-vs-demon-2026-10-15]]: the
shipped contested bidder beat the confirmed Demon while using more CPU.
The objective is to improve the Demon against that pinned opponent within
comparable total seat CPU. [Issue 184](https://github.com/jasonyandell/mk5-main/issues/184).

## Opponent and baseline

The opponent is `zeb-0.4.1-contested`, tag `zeb-v0.4.1` at `0b57a8f5`,
160 worlds, seed 0, shipped match-aware overcall bidder, pip declarations,
soft reading and exact margin ties. The baseline is the exact confirmed
`demon-pimc-n100-bidc102a137-straight-2e233d29` from
[[demon-overcalling-2026-10-15]]. The previous trial's exact native binaries
are reused and checked by SHA-256; npm 0.4.1's integrity and both shipped
JavaScript policy files match the tagged source. No opponent policy is edited.

## Fixed development, stage 1

Before any development game: seven arms, **32 duplicate pairs / 64 complete
games each**, straight 42 to seven marks. Fresh seed base `0xD4500000`, plus
pair index. Both seatings finish. Every arm uses 100 play worlds, 64 bidding
worlds, nine straight declarations, exact margin ties and distinct uniform
samples. Only the two cutoffs change; every arm keeps minimum point raises
and partner pass. Cutoffs are DD decision parameters, not calibrated chances.

| Arm | Opening cutoff | Opponent-held cutoff |
|---|---:|---:|
| Control | 0.875 | 0.5 |
| Open 0.50 | 0.5 | 0.5 |
| Open 0.625 | 0.625 | 0.5 |
| Open 0.75 | 0.75 | 0.5 |
| Over 0.25 | 0.875 | 0.25 |
| Over 0.625 | 0.875 | 0.625 |
| Over 0.75 | 0.875 | 0.75 |

All arms finish, without early stopping or dropped games. Infrastructure
failures preserve the outputs and are audited before deterministic rerun.
The reporter uses 95% whole-pair bootstrap intervals, 40,000 resamples, seed
666. Among arms whose total CPU-ratio upper bound is at most 1.20, select
the highest game-win fraction; ties go to marks/game, then lower CPU.
These comparisons are development only. Further development, if needed,
gets a separate protocol before its games.

The [public protocol and executable harness](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/25273a221ed808612f1275ba18826d35770623a3/demon/2026-10-16-ascent/development-protocol)
pin runtime source `a48b835f`, workspace and script hashes, participant
records, binary hashes and package parity. Plan SHA-256:
`d6ffaec97d8c29649bde0deed59feadae85b749218898be1edea31112522d23e`.
Fresh downloads of plan and manifest match local bytes.

## Stage 1 result and round 2

All **448 games / 4,777 hands** finished without replacement, and the source,
seat-visible request, bidding, scoring and cost audit passed. The
[complete evidence](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/64ead57317cf2ec67fcc450b4ffa1fcecd5bd029/demon/2026-10-16-ascent/development-results)
contains every game, trace, cost and arm report. Fresh downloads of the plan,
manifest, a raw game and all reports match local bytes.

| Arm | Wins / 64 | Win rate, 95% pair interval | Total CPU / native Zeb, 95% pair interval |
|---|---:|---:|---:|
| Control | 29 | 45.3% [35.9, 54.7] | 0.579 [0.567, 0.590] |
| Open 0.50 | 21 | 32.8% [21.9, 43.8] | 0.547 [0.535, 0.560] |
| Open 0.625 | 18 | 28.1% [18.8, 37.5] | 0.554 [0.541, 0.568] |
| Open 0.75 | 20 | 31.3% [21.9, 40.6] | 0.563 [0.551, 0.574] |
| Over 0.25 | 27 | 42.2% [29.7, 54.7] | 0.594 [0.581, 0.606] |
| Over 0.625 | 32 | 50.0% [39.1, 60.9] | 0.567 [0.555, 0.581] |
| Over 0.75 | 32 | 50.0% [39.1, 60.9] | 0.555 [0.543, 0.568] |

The fixed tiebreak selects over 0.625 (+0.047 marks/game, 95% [-0.688,
+0.813]) over 0.75 (-0.078, [-0.859, +0.703]). None establishes a win
against Zeb. This is development, not a promotion.

Round 2 fixes **nine arms × 32 fresh duplicate pairs / 576 games**, seed
base `0xD4510000`. Reference: opening 0.875 / opponent 0.625, 100 play
worlds, 64 bid worlds. The other arms change opponent cutoff to 0.875 or
1; opening cutoff to 0.9375 or 1; both cutoffs to 0.9375/0.875 or 1/1;
play worlds to 400; or bid worlds to 128. The engines and adapters remain
identical. Every arm finishes before the same CPU-qualified selection;
confirmation remains separate. The remaining CPU budget motivates the
larger-world arms, without assuming they improve play. The
[round 2 protocol](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/0fda7a1c52e6143f05c7dd78a3a9e36cc03cf35a/demon/2026-10-16-ascent/development-2-protocol)
was published and freshly verified before play (plan SHA-256
`da05bf7d926fa52bd3bcb8270cb6e6ec397e47a8673f421519c33f996ba1f2e3`).

## Exact declaration elimination, prototype

An isolated prototype searches declarations in the existing tie order:
heuristic preference first, then ascending id. The first is evaluated fully.
For a later declaration, successes plus unsearched worlds is an upper bound
on its final count. When that bound is at most the incumbent's count, it
cannot displace the incumbent, including on a tie. The winning declaration
and its count remain exact on the same sampled worlds.

All **1,934 forecasts** from stage 1's 64 control games were replayed without
outcome filtering. Every winner, winning count and sample size matches; the
full reference also reproduces every saved original node count. Three Rust
tests cover randomized/subset/reordered declaration sets, ties, invalid
inputs and the perfect preferred declaration. The replay alternates full
and bounded process order; intervals resample the 32 whole duplicate pairs,
40,000 times, seed 666.

| Native forecast cost | Bounded / full | 95% pair interval |
|---|---:|---:|
| CPU, including process startup and shutdown | 0.5210 | [0.5124, 0.5295] |
| Complete native self instructions | 0.5221 | [0.5135, 0.5306] |
| Search nodes | 0.5220 | [0.5134, 0.5306] |
| Per-world solves | 0.5520 | [0.5440, 0.5601] |

CPU falls from 424.873 to 221.368 seconds over the fixed corpus. This is
**forecast-kernel performance**, not a whole-seat runtime ratio or a strength
result. Wall time is descriptive under shared load. The prototype is isolated
from both frozen strength rounds and is not a registered player.

[Public source overlay, inputs, both outputs and report](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/f41144e612b1dae5f175ac30a1ed7cee367c096f/demon/2026-10-16-ascent/bid-pruning-prototype)
reconstruct the prototype from the original native source pin `2e233d29`.
Fresh downloads of its plan, manifest and report match local bytes.

## Independent confirmation rule

Freeze the selected policy and source before **512 fresh duplicate pairs**
against the pinned Zeb 0.4.1. Attempt r spends alpha `0.05/(r*(r+1))`,
with 40,000 whole-pair bootstrap resamples (seed 666). Success requires
game-win lower bound above 50% and total seat CPU-ratio upper bound at most
1.20. Every attempt remains public; no development deals enter confirmation.
There is no candidate promotion yet.

## Measurement and clocks

Each seat's startup, bidding, declaration, play, close, IPC, Python adapter
and waited native descendant CPU are included. Build/install and the common
referee are excluded. All players decide by deterministic worlds and seeds.
Each serial duplicate pair requests `scripts/cpu-slot --threads 1`; other
sessions share the machine. Wall time is reported with that scope and is
not a selection or confirmation criterion. Any separate wall-clock
confirmation requests an exclusive slot ([[standing-desks]]).

Available instruction counts are also retained: Python during seat calls,
the seat process, and native Zeb's child sampled just before closing it.
That last count excludes child shutdown, so it is not labeled a complete
descendant instruction total. CPU includes the waited child's full lifetime.

## Links

[[the-demon]] · [[tournament]] · [[rematch-zeb-041-vs-demon-2026-10-15]] ·
[[demon-overcalling-2026-10-15]] · [[strategy-fusion-fence]]
