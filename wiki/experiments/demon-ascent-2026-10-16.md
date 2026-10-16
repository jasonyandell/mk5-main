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
