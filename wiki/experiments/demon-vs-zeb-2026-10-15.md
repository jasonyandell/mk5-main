---
title: "Demon versus Zeb at comparable total runtime"
kind: experiment
first_seen: 2026-10-15
last_updated: 2026-10-15
status: complete
---

**STRATEGY-FUSION FENCE.** The [[the-demon]] remains a sanctioned tournament
opponent. The objective is a measured win against current pinned Zeb at
comparable total seat runtime, on complete games of straight 42.

## Declaration audit

The initial development run exposed a domain error in the PIMC bidder:
it could declare 8, doubles as their own suit, the physics used for nello.
The general arena accepted that physics under a points contract. Shipped
Zeb correctly refuses it in straight 42. The straight declarations are
`[0,1,2,3,4,5,6,7,9]`; nello is outside this tournament's rules.

All four initial development arms were stopped and invalidated. Completed
and partial records remain public; none enter selection. The corrected
arms restart on the same development seeds and retain their fixed sizes.
No confirmation had started. The new participant factory records the allowed
declarations explicitly and uses a new identity. Workers exclude nonstraight
PIMC records from scheduling and ratings, and preserve incompatible pending
claims under `unsupported/` without continuing them. Historical participant
records and raw game bytes remain unchanged. Source: [[1273bc97]].

The earlier [[demon-next-form-2026-10-14]] bidder confirmation used declaration
8 in **87 of 5,620 hands**. Its 324–188 result describes the ten-declaration
experimental arena; it is **not a valid straight-42 strength benchmark**.
Removing those hands or games would not repair the comparison, because
declaration eligibility also affects bids that do not win the auction.

[Public audit and invalid trial records](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/25320e15068b745a13959542aeea8167ada602ef/demon/2026-10-15-vs-zeb/declaration-audit)
include the declaration census, complete and partial initial games, original
and corrected protocols, adapter checks and source provenance. The census, one
raw game and the manifest were freshly downloaded and hash-verified.

## Frozen opponents and success criterion

The npm registry's latest release was 0.4.0 when checked. Its integrity is
`sha512-7DhOCStdZbGEkh+RBwdDWOYgptwS+OYEPoGLF3BVJwCs/A0rmMNdq+X3fBrgQ2C55WWRXb3dNVmFY9jucFDKDA==`.
Both opponents use the shipped play defaults at 160 worlds, seed 0, with
reading and exact margin ties enabled. Development runs the npm WASM package;
confirmation runs the identical Rust player natively, with the shipped npm
JavaScript bidding functions:

- Primary: opt-in match-aware bidding, pip declarations only as measured in
  [[match-aware-bidding-2026-10-14]], cutoff 0.8.
- Secondary: the standard [[tournament]] seat, hand-only bidding behind
  partner pass, nine default forecast declarations, cutoff 0.8.

The research host's standard mode is held to the existing tournament host;
its match-aware choices are held to direct calls to the pinned package.
The 27-case check covers seat orientation, bid holder, and declarations.
Native Zeb is built from release commit `29f519953467f6aacd78bd17700f3c67fda6b4eb`,
adding only an rlib output, the existing JSON dependency, and a JSON-lines CLI.
All 203 replay requests agree exactly with npm WASM, including probabilities;
four complete games (47 hands) reproduce their auctions, plays and scores.
Only own tiles and public history/auction/scores enter either seat.

Calibration and two budget-development arms ran 16 duplicate pairs each.
The budget arms separately raise play from 100 to 800 worlds or bidding
from 16 to 64 common worlds; the bid cutoff stays 0.875. Selection follows
completed paired marks and runtime measurements, never an interim win count.

The selected configuration is frozen for 256 fresh duplicate pairs per
opponent, with no early stopping or dropped games. Success requires a win
fraction interval wholly above the 50% chance baseline and runtime-ratio
upper bounds at most **1.20** for both total seat wall time and CPU.
Faster is acceptable. Intervals resample whole pairs 40,000 times; marks/game
is secondary. If confirmation is repeated after further development, attempt
`r` uses two-sided alpha `0.05/(r*(r+1))`; every attempt is retained.

Total seat time covers initialization, bidding, declaration, play, close,
IPC and Python adapter work. CPU is child-process user+system time plus the
in-process seat calls. Native Zeb's Node host waits for its native child;
a direct accounting check confirms both processes' CPU is included.
Downloads/builds and the shared referee are excluded.
Startup and per-operation latency distributions are reported separately.
Each CPU-slot batch runs one serial duplicate pair on the Apple M5 Max;
processor-specific tuning must be labeled and preserve exact search values.

The protocol is recorded before confirmation in
[issue #165](https://github.com/jasonyandell/mk5-main/issues/165).
The native runtime amendment and candidate freeze were recorded before the
first confirmation game. Attempt 1 is complete: the standard comparison
qualifies, but the match-aware win interval crosses 50%. The joint criterion
is not met and the candidate is not promoted.

## Completed development and frozen candidate

All four corrected arms completed, 32 games each. Intervals below are 95%
paired bootstrap intervals, 40,000 resamples; chance is 50%, marks baseline
zero. Runtime ratios here compare native Demon with npm WASM Zeb and are
development measurements, not the confirmation runtime criterion.

| Demon configuration / opponent | Wins | Win fraction | Marks/game | Total CPU ratio |
|---|---:|---:|---:|---:|
| Play 100, bid 16 / match-aware | 19–13 | 59.38% [40.63, 78.13] | +1.000 [−0.250, +2.281] | 0.236 [0.227, 0.244] |
| Play 100, bid 16 / standard | 24–8 | 75.00% [56.25, 90.63] | +1.375 [+0.344, +2.344] | 0.108 [0.104, 0.111] |
| Play 800, bid 16 / match-aware | 19–13 | 59.38% [40.63, 78.13] | +0.938 [−0.313, +2.125] | 1.077 [1.021, 1.138] |
| Play 100, bid 64 / match-aware | 20–12 | 62.50% [46.88, 78.13] | +1.313 [+0.031, +2.531] | 0.554 [0.537, 0.572] |

The selected configuration is **100 play worlds, 64 bidding worlds, cutoff
0.875**, uniform play worlds, exact margin ties, and the nine straight
declarations. It has the highest development marks estimate among candidates
against match-aware Zeb and fits the native runtime yardstick. Selection does
not establish that it is stronger than the other development arms.

The final source is [[7e00788a]], candidate
`demon-pimc-n100-bide7c1889b-straight-7e00788a`. Its frozen binary SHA-256 is
`1c2ccee87a02069ef4e14d957e97c6042007694abd79058bf8d3f1b73721494e`.
Attempt 1 ran 256 fresh duplicate pairs against each native Zeb policy,
seed base `0xD4100000`, with 97.5% two-sided intervals. All games finished
before evaluation; the candidate stayed fixed throughout. This candidate is
not promoted to the tournament roster.

## Complete confirmation attempt 1

Can this frozen Demon beat both pinned native Zeb policies while using at
most 1.20 times their total seat runtime? **The joint criterion is not met.**
All 1,024 scheduled games completed without drops or retries. Intervals are
97.5% two-sided paired bootstrap intervals, 40,000 resamples; chance is 50%,
marks baseline zero, and runtime ratios divide Demon by Zeb.

| Native Zeb 0.4.0 opponent | Wins–losses | Win percentage [interval] | Marks/game [interval] | Total CPU ratio [interval] | Total wall ratio [interval] |
|---|---:|---:|---:|---:|---:|
| Match-aware | 278–234 | 54.30% [49.80, 58.79] | +0.922 [+0.602, +1.242] | 0.663 [0.658, 0.669] | 0.664 [0.659, 0.669] |
| Standard | 294–218 | 57.42% [52.93, 61.91] | +0.688 [+0.420, +0.961] | 0.291 [0.289, 0.294] | 0.292 [0.289, 0.294] |

The standard comparison clears every bar. The match-aware comparison clears
both runtime bars but its win interval includes 50%. Its positive marks
interval does not substitute for the registered game-win criterion. The
observed lead is not a confirmed broad win against both policies.

Across the match-aware games, Demon used 3,271.5 CPU seconds and 3,274.2 seat
wall seconds; Zeb used 4,932.0 and 4,930.8. Against standard Zeb, the totals
were 3,609.6 / 3,614.2 for Demon and 12,387.5 / 12,397.0 for Zeb. These are
whole-seat sums over 512 games per arm on the recorded Apple M5 Max.

| Opponent / seat | Bid p50 / p95 / max (ms) | Play p50 / p95 / max (ms) | Startup p50 / p95 (ms) |
|---|---:|---:|---:|
| Match-aware / Demon | 166.4 / 262.7 / 448.6 | 0.26 / 78.9 / 879.5 | 6.15 / 7.23 |
| Match-aware / Zeb | 242.1 / 286.7 / 429.8 | 0.95 / 72.3 / 444.7 | 48.19 / 55.31 |
| Standard / Demon | 169.0 / 267.1 / 447.9 | 0.27 / 81.5 / 883.2 | 6.21 / 7.82 |
| Standard / Zeb | 707.3 / 844.0 / 1,122.8 | 0.91 / 76.3 / 640.6 | 48.61 / 57.53 |

The full audit reconciles all 1,024 games and 11,005 hands, frozen binary and
source hashes, seeds, legal straight declarations, seat-visible request
boundaries, metered call sums and report totals. An independent rebuild from
the public source pin matches all 185 retained replies and search-node counts.
[Complete attempt-1 evidence](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/d99f621677f6102fa0308922e7b21edf2d0f46f2/demon/2026-10-15-vs-zeb/confirmation-attempt-1)
contains every game, timing, trace, report, protocol and audit; both reports,
the verdict, audit, one raw game and the manifest were freshly downloaded and
verified. Provenance: [[demon-zeb-attempt-1]].

Further development is [[demon-overcalling-2026-10-15]], under issue #165. The completed attempt is
retained regardless of later outcomes. A new confirmation must freeze its
candidate, fresh seeds and size before play, with attempt-2 alpha 1/120;
the old games cannot be added to a new confirmation or selectively discarded.

## Profiling and retained optimization

Processor-specific compiler flags and cached solver templates produced no
useful measured gain and were rejected. The retained change asks the native
forecaster only for the seat's allowed declarations. Unrequested entries are
explicitly absent (`null`), and no search runs for them. Requested counts use
the same common worlds and remain exact. Source: [[7e00788a]].

On 185 fixed requests, five alternating replays reduce mean child CPU from
2.652 to 2.402 seconds, an observed 9.4% saving (individual paired savings
8.0–13.0%). This is repeated timing on one fixed corpus, not a strength result
or a population confidence interval. The final release reproduces every
retained reply and node count of the measured prototype. Tests cover reordered
subsets, missing results, invalid declarations, independent minimax and whole
games; `scripts/check --fast` is GREEN (four affected suites).

[Development evidence and frozen protocol](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/11cb9258fb345a0d385308ab2e07077955a7c101/demon/2026-10-15-vs-zeb/development)
include all corrected games, timings, traces, reports, rejected trials,
native parity, patches and the selection record. The selection, one raw game
and manifest were freshly downloaded and hash-verified. The completed
confirmation is reported above.

## Links

[[the-demon]] · [[tournament]] · [[scenario-player]] ·
[[demon-next-form-2026-10-14]] · [[strategy-fusion-fence]]
