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

## Round 2 result

All **576 games / 5,739 hands** finished without replacement; the full audit
passed. [All arms and raw evidence](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/cfe492814eaf383f03a793f35f102de369f9cfce/demon/2026-10-16-ascent/development-2-results)
are public, with fresh plan/manifest/raw-game/report downloads verified.

| Arm | Wins / 64 | Win rate, 95% pair interval | Total CPU / native Zeb, 95% pair interval |
|---|---:|---:|---:|
| Reference | 28 | 43.8% [32.8, 54.7] | 0.550 [0.541, 0.558] |
| Opponent cutoff 0.875 | 23 | 35.9% [25.0, 46.9] | 0.524 [0.516, 0.531] |
| Opponent cutoff 1 | 17 | 26.6% [15.6, 39.1] | 0.516 [0.507, 0.524] |
| Opening cutoff 0.9375 | 23 | 35.9% [28.1, 43.8] | 0.554 [0.543, 0.564] |
| Opening cutoff 1 | 11 | 17.2% [9.4, 25.0] | 0.554 [0.547, 0.560] |
| Both 0.9375 / 0.875 | 15 | 23.4% [15.6, 32.8] | 0.531 [0.521, 0.541] |
| Both 1 / 1 | 2 | 3.1% [0.0, 7.8] | 0.527 [0.519, 0.535] |
| 400 play worlds | 27 | 42.2% [31.3, 53.1] | 0.854 [0.835, 0.874] |
| 128 bid worlds | 29 | 45.3% [35.9, 54.7] | 0.981 [0.962, 0.999] |

The fixed selection is 128 bid worlds, but no arm establishes a win. Merely
passing more loses badly in this field; additional play/bid samples alone
also fail to establish a gain in this development set. There is still no
confirmation or promotion.

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

## Integrated search and preemptive bidder

`bounded_forecast=True` opts a new participant into the native `forecast_best`
operation. It returns the exact winning declaration and count; other counts
are explicitly absent. Existing participant records and their full-forecast
path remain unchanged. The integrated native protocol reproduces all 1,934
original winners/counts and the prototype's sampled-world, node and solve
counts ([complete integration replay](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/aab6fe30cd93560bcc33e83291952b0999b9f1a6/demon/2026-10-16-ascent/bid-pruning-integration)).

The experimental `pimc-jump:worlds:opening_cutoff:opponent_cutoff` bidder
searches for the **highest supported legal bid from 30 through 42**, instead
of the minimum raise. Each target uses the same hidden deals and exact DD
values. For each declaration, make count is nonincreasing in the point
target; its maximum across declarations is therefore nonincreasing too.
After checking the minimum legal bid, binary search needs at most four
more forecasts. The winning declaration is cached for the won contract.
Partner pass remains; a standing bid of 42 or more is not raised. These
cutoffs remain decision parameters, not calibrated make probabilities.

The hypothesis is that preemption makes it more costly for Zeb to take a
contract from a strong Demon hand. This bidder is implemented, not yet
strength-validated or rostered. Tests cover bounded/full whole-game parity,
repeatable legal jump-bidding games, monotonicity on common worlds, maximum
legal supported bids and partner pass. The five relevant check suites pass:
16 Rust, 66 tournament and 8 fence tests, plus wiki and license checks.

## Fixed development, round 3

The search/bidder capability is merged at `87951424`; no new participant is
rostered. A clean build from its public native trees has SHA-256
`e5cecac715b0f344daae58cdec3bb149163dfe9c8461c844def738bfb1ac0711`.
The full and bounded jump policies play identical games at both seatings
against native Zeb: 22 distinct hands, every auction/play/score identical,
and the source, maximal-bid, declaration and cost audit passes. The initial
smoke wrapper omitted the new `historical` keyword; both failures occurred
before seat startup, with zero game records. Their logs are retained.

Round 3 fixes **six arms × 32 fresh duplicate pairs / 384 games**, seed
base `0xD4520000`. All use 100 play worlds and the bounded forecast. The
reference retains minimum raises with 128 bid worlds and cutoffs
0.875/0.625. Jump arms use 32, 64 or 128 bid worlds at those cutoffs; two
more use 64 worlds with opponent cutoffs 0.5 and 0.75. Every arm finishes
before the same CPU-qualified selection. This is development only.

The [public protocol and preflight](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/121a18e5ec83d2ebe7c66a9f835b5de17c0013ae/demon/2026-10-16-ascent/development-3-protocol)
were frozen before development. Plan SHA-256:
`3977417585105e5e54894c3801698cd4cc80e186778a90dad0a0522ba9d23b25`.
Fresh plan, manifest and raw smoke downloads match local bytes.

## Round 3 result: preemption does not establish a gain

All **192 duplicate pairs / 384 games / 4,105 hands** completed. The
source, seat-visible trace, legal bid, maximal supported jump, declaration,
score and cost audits pass. Every arm stays within the development CPU
criterion, but none establishes a win above the 50% chance baseline.
Intervals below are 95% whole-pair bootstrap intervals.

| Arm | Demon wins | Win fraction [95% CI] | Total seat CPU / Zeb [95% CI] |
|---|---|---|---|
| reference | 29/64 | 45.31% [32.81, 57.81] | 0.581 [0.569, 0.592] |
| jump32 | 16/64 | 25.00% [15.62, 34.38] | 0.358 [0.343, 0.373] |
| jump64 | 26/64 | 40.62% [28.12, 53.12] | 0.574 [0.556, 0.592] |
| jump128 | 20/64 | 31.25% [20.31, 42.19] | 1.005 [0.975, 1.034] |
| jump50 | 16/64 | 25.00% [15.62, 34.38] | 0.615 [0.594, 0.635] |
| jump75 | 26/64 | 40.62% [29.69, 51.56] | 0.527 [0.510, 0.544] |

The fixed selection returns the minimum-raise reference, 29/64. No independent
confirmation starts from this result. The data reject promoting the tested
preemptive arms; they do not prove every preemptive policy inferior.
[All raw games, costs, traces, reports and audits](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/bc8fd1ee839e3abfdc781cb97e6cd1a679147087/demon/2026-10-16-ascent/development-3-results)
are public: 87,661,418 bytes, 1,197 files; the plan, manifest, one raw game
and every arm report were freshly downloaded and matched.

## Exact support predicates and the next auction hypothesis

A bid decision needs only `max(make_count) >= required`, not the winning
declaration. A declaration passes once enough sampled worlds make; it fails
when even every remaining world making cannot reach the required count. The
search returns on the first supported declaration. This is an exact predicate
on the same sample. A won auction still computes the best declaration.

The isolated prototype replays **all 1,934 original control forecasts**, with
required count 56 at bid 30 and 40 at higher bids, out of 64 worlds. Every
predicate agrees with the existing exact best forecast, and every stopping
certificate passes. Native lifetime CPU is 90.274 versus 189.892 seconds:
ratio **0.4754 [0.4652, 0.4853]** (95%, 32 whole duplicate pairs, 40,000
resamples, seed 666). Complete self-instruction ratio is
**0.4770 [0.4673, 0.4864]**. Process startup/shutdown are included;
shared-load wall is descriptive. This is predicate cost, not whole-seat
cost or strength; eventual declaration work remains.
[Prototype source overlay, inputs, outputs, tests and report](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/0f5bcde41da7889abf90ee79990b360ca0321e1a/demon/2026-10-16-ascent/bid-support-prototype)
are public and freshly verified.

The optional `support_forecast` mode uses that native operation for bidding
and the existing bounded forecast for declaration. Its count threshold is
found with the original division/comparison rule, avoiding multiplication
rounding at cutoffs such as `0.07` with 100 worlds. It has its own participant
ID suffix; existing records and native protocols stay unchanged.
The integrated JSON operation also matches all 1,934 prototype predicates,
world counts, search nodes, solve counts and stopping certificates
([production replay](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/364619a7c9cc66133f46e59f34317e530b3b6a81/demon/2026-10-16-ascent/bid-support-integration)).
Complete-game parity covers minimum, jump and early-opening bidders across
full, bounded-best and support-predicate modes. The relevant gates pass:
18 Rust, 81 tournament and 8 fence tests, plus wiki and licenses. All 13
existing roster records are byte-for-byte unchanged; no roster entry is added.

The experimental `pimc-bait:n:opening:opponent` bidder opens the minimum legal
point bid unconditionally when no bid stands and at least two seats remain
after it. Later openings and opponent-held auctions use their named cutoffs;
it still passes over partner and standing mark bids. The hypothesis is that
Zeb's aggressive overcalling can be induced to carry weaker contracts. This
is not established by the failed conservative or preemptive sweeps; a new
fixed development trial is required. No new participant is promoted.

## Fixed development, round 4

The support/bait capability is merged at `db239f39`; no participant is
rostered. A clean public-source build has SHA-256
`16d755fb3feb86e0952310f069516c2545826e931db201df67d5594fbd719847`.
Best-forecast and support-predicate early-opening policies play identical
whole games at both seatings against native Zeb: 21 distinct hands, with
all auctions, plays, scores, support certificates and cost audits checked.

The fixed sweep is **eight arms × 32 fresh duplicate pairs / 512 games**,
seed base `0xD4530000`. All use exact support bounds. The reference is
100 play worlds, 128 bid worlds, minimum raises and cutoffs 0.875/0.625.
Five early-opening arms use 100 play worlds, 64 bid worlds, later-opening
cutoff 0.875 and opponent cutoffs 0.625, 0.75, 0.875, 0.9375 or 1. Two more
use the 0.875 and 1 opponent cutoffs with 400 play worlds. All arms finish
before the existing CPU-qualified selection; this remains development.

The [public protocol and preflight](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/e7044f0dc95cf5f79ae1786e2059ee30c25f433e/demon/2026-10-16-ascent/development-4-protocol)
were frozen before development. Plan SHA-256:
`d934cdca2dd901067b2f726500d96bfa1e63b2e09e5f97cc1807ba740ad4b81a`.
Fresh plan, manifest and raw smoke downloads match local bytes.

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
