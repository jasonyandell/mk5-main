---
title: "The Demon: a fenced CPU PIMC player, seated in the tournament as strategy fusion's measuring stick"
kind: topic
first_seen: 2026-10-14
last_updated: 2026-10-16
status: active
---

**STRATEGY-FUSION FENCE.** The Demon is a fenced tournament opponent and a measuring stick
only. It is never a component of Zeb, Walt or Silas. The ruling is on
[[strategy-fusion-fence]].

## What it is

The Demon is the textbook strategy-fusion player, Perfect-Information Monte Carlo (PIMC), and
it runs on any CPU. Jason asked for it on 2026-10-14: "we definitely want to test v strategy
fusion." Until then the ladder's only fusion player was E[Q] n=10 ([[expected-q-value]]). E[Q]
needs a GPU, so the cloud worker could not seat it.

At each play the seat:

1. replays the public record with Walt's `Rec` (`scenario/core/src/planner.rs`): the plays,
   the banked points and every void a seat has shown;
2. draws the configured number of distinct worlds uniformly from its fiber, or takes
   the whole fiber when it is no larger. A completion-count dynamic program un-ranks
   each selected world without enumerating its predecessors;
3. solves each world double dummy with Silas's `ClairvoyantSolver` ([[silas]]): exact
   make-or-set of the contract, all four hands face up, every seat playing perfectly;
4. plays the tile that makes in the most worlds (sets, when defending). Exact ties go to the
   best summed double-dummy margin, then to the lowest tile id.

A seat with one legal tile plays it without solving anything. The original pins use the
arena's `heuristic` bidder (Roberson risk budget), the same bidder E[Q] n=10 uses in the
tournament. The experimental `4454f603` form uses `pimc:16:0.875`: forecasts on
16 common hidden deals choose a declaration and whether to bid; it passes over a
partner's standing bid. [[demon-next-form-2026-10-14]] defines the policy and its
ten-declaration evidence. That promotion is withdrawn: declaration 8 belongs to
nello, outside straight 42. New bidders explicitly restrict declarations to
0–7 and 9; [[demon-vs-zeb-2026-10-15]] holds the audit and corrected comparisons.
The 64-world bidder searches only the requested declarations. Its first native
Zeb comparison fails the joint promotion criterion. The next form lowers the
threshold when an opponent already holds a point bid, with the same forecast
and no added search work. [[demon-overcalling-2026-10-15]] confirms this form
beats both pinned native Zeb 0.4.0 bidders at lower total seat CPU and wall
time on Apple M5 Max; its exact participant is registered and rostered.

[[rematch-zeb-041-vs-demon-2026-10-15]] measures Zeb 0.4.1's reply: its
contested bidder wins against that Demon at greater CPU cost.
[[demon-ascent-2026-10-16]] starts the next development and independent
confirmation. Its first five development rounds do not establish a win.
The optional bounded forecast preserves declaration choices while pruning
losing searches; exact support predicates further reduce bidding work. The
preemptive jump bidder does not earn promotion. An early opening of 30 is
cheaper but does not establish a win. Its optional early point bid now allows
30–40. The sixth fixed sweep selects early 40, which targets pinned Zeb's
willingness to overcall. That exact early-40 form now passes the independent
win/CPU criterion and is registered and rostered. The study records the
opponent-specific rules, full evidence and limits.

Step 3 is the fusion. Each world is credited with the line a seat could choose only if it saw
every hand, and E[max] ≥ max E ([[strategy-fusion]]). That is why the Demon makes a useful
opponent and why it is fenced.

## Prior work it reuses (checked before building)

- **The solver.** Silas's `ClairvoyantSolver` (`silas/src/clairvoyant.rs`, behind the
  non-default `clairvoyant` feature): integer alpha-beta with a transposition table, and a
  null-window search for make-or-set. It is the fastest exact per-world solver in the projects.
  texas-42's Rust "scalar PI" (`walt/walt/src/strat/scalar.rs`) and its Python
  `exp5_core.Solver` compute the margin by tricks or points only. They have no make-or-set and
  no Python or Rust API in mk5. Forge's oracle and E[Q] are GPU-only.
- **The sampler and the rules.** Walt's fiber completion counts, record and rules tables, plus
  `ZebPhysics` (`scenario/core/src/physics.rs`), which maps Zeb's declaration ids onto Silas's
  physics.
- **The cross-game labs** (`~/code/plunge-walt-bridge/lab/`). Their PIMC bots are for bridge,
  euchre, skat and spades. None of them plays 42.
- **The bar.** "BR(BR(uniform)) beats PIMC" is Walt's +0.74 [+0.42, +1.05] marks/game against
  E[Q] n=10 ([[brainstorm-2026-10-14-ledger]], [[scenario-player]]). That PIMC sits on a learned
  Q. The Demon is the same idea with an exact solve.

## Code and the fence

| Where | What |
|---|---|
| `tournament/demon/` (Rust crate) | `src/lib.rs` `decide` (the four steps), `src/sampling.rs` (uniform subsets or posterior draws), `src/bidding.rs` (contract forecasts), `src/main.rs` (JSON protocol; refuses inconsistent public records), `tests/demon.rs` |
| `tournament/demon_fenced_seat.py` | `DemonSeat`: the bidder and one binary process per team per game; `build` from the pinned trees |
| `tournament/participants.py` | `demon(...)`, immutable pins, explicit straight declarations and scheduling eligibility |
| `tournament/tests/test_demon.py` | whole faceoff games, determinism, the pin's tree check |

How it stays fenced:

- Every file above carries the token and the line "The Demon: a fenced tournament opponent;
  measuring stick only". A Rust test checks this.
- The crate is the only one outside Silas's own tests that turns on Silas's `clairvoyant`
  feature.
- The guard test (`src/tests/architecture/strategy-fusion-fence.test.ts`, the `fence` suite)
  fails on any of the following:
  - a file that names the Demon's seat or crate without the token;
  - a different crate that enables the feature;
  - any file under `silas/`, `scenario/`, `zeb/`, `arena/`, `walt/` or `src/game/` that names
    the Demon.
- The Demon is recorded in the fence page's inventory as a sanctioned baseline.

Tests (`scripts/check demon`, `py-tournament`):

- **Agreement with an independent double-dummy solver.** A brute-force minimax over Zeb's own
  rules tables (not Silas's physics) reproduces the Demon's make counts and margin tiebreaks on
  every world of 30 enumerated fibers, about three tricks from the end, under randomly drawn
  declarations (any of the ten) and bids from 30 to 84.
- **Determinism.** The same seed gives the same decision bit for bit. The worlds depend only on
  the seed, and the solver's values are exact, so its table changes the work and never a value.
  The same faceoff spec plays the same game twice.
- **Legality.** Whole hands play legally at all four seats under all ten declarations. Whole
  faceoff games play through the arena referee.

## Search-table locality (2026-10-14)

The current implementation keeps one small search table per world, completes that
world's margin tie work together, and drops its table before the next world.
Moves and values are preserved. The paired replay, CPU and memory measurements,
and 400-versus-100-world pilot live in [[demon-search-tables-2026-10-14]].

## Compact keys and distinct worlds (2026-10-14)

The next pass uses 31-bit ownership-scoped transposition keys, exact pruning of
interchangeable plays, direct final-trick resolution and shared immutable rules.
The sampler draws distinct uniform worlds through the fiber's completion-count
DP. Its proof, exact-parity checks and fixed 1,600-versus-100-world strength trial
are in [[demon-power-2026-10-14]]. At 400 identical worlds, final search uses 13.2%
less CPU and 70.4% less peak RSS than the compact build (paired elapsed saving
13.4% [9.7%, 17.1%]). The 1,600-world trial finishes 124–132 over 128 duplicate
pairs: 48.4% [44.1%, 52.7%], no demonstrated gain, at 11.49× native play CPU for
the frozen pre-bisection policy. The larger budget stays experimental. The earlier
pins keep their original sampler.

## Is sample count the only remaining strength knob?

The declaration-reader and contract-forecast candidates are implemented and
graded in [[demon-next-form-2026-10-14]]. The bidding result used a ten-declaration
arena and is not a valid straight-42 gain. The reader and combined form do not
establish gains. Corrected trials against Zeb are in [[demon-vs-zeb-2026-10-15]];
the historical participant bytes remain unchanged and the incompatible pin is
excluded from straight-game scheduling and ratings.

No. The per-world solve is already exact through the end of the hand; deeper
search cannot improve that world's answer. More worlds reduce sampling error
in the chosen PIMC expectation. Transposition tables and exact pruning reduce
its cost. Neither establishes the best possible PIMC player for Texas 42.
[[demon-power-2026-10-14]] tests one joint change of budget and sampler against
one baseline with a shared bidder; its null result does not prove saturation.

Three unexhausted directions remain within a PIMC-based player:

- **Beliefs.** Uniform play still uses legality, hand sizes and proven
  voids. The opt-in declaration reader has positive support and a direct
  posterior sampler, but its heuristic-based likelihood did not establish a
  gain. Bids and discretionary plays are still unmodeled. [[silas]]'s bid/play
  reading supplies related model-conditioned inference machinery; a model
  calibrated to the actual opponent remains a candidate.
- **Bidding and declaration.** The overcalling form now meets the straight-42
  strength/runtime objective against pinned native Zeb in
  [[demon-overcalling-2026-10-15]]. The cutoff is a decision parameter; double-dummy
  make frequency is not a calibrated imperfect-information win probability.
  Bidding utility, score-aware risk and separating the contributions of
  partner pass, willingness and declaration remain open.
- **Budget allocation and decision rules.** The current fixed world budget and
  exact-make-tie margin rule are design choices. Adaptive sampling of close
  choices and alternative tie rules are testable; their strength is unmeasured.
  Allocation can improve strength per unit compute even if it leaves the
  infinite-sample policy unchanged.

The first direction has precedent: [Solinas, Rebstock and Buro (2019)](https://arxiv.org/abs/1903.09604)
improved PIMC Skat cardplay by sampling worlds informed by move history. That
supports an experiment here, not a claim of a measured Texas 42 gain.

There is also a structural limit. Even complete enumeration still credits each
world with continuations tailored to hidden hands. [[strategy-fusion]] remains
regardless of sample count ([Long et al., 2010](https://webdocs.cs.ualberta.ca/~nathanst/papers/pimc.pdf)).
Requiring one action across indistinguishable information states changes the
search beyond plain independent-world PIMC. [[demon-overcalling-2026-10-15]]
completes [#165](https://github.com/jasonyandell/mk5-main/issues/165)'s pinned
Zeb comparison; optimal PIMC and generalization to other opponents remain open.

## Original cost (measured 2026-10-14, Apple M-series, one core)

Pilot: whole hands with the Demon at all four seats, random deals, declarations 0–9, bids
30–42. Per decision:

| worlds | tiebreak | mean | trick 1 p50 / max | trick 2 mean | tricks 4–7 |
|---|---|---|---|---|---|
| 10 | margin | 4.6 ms | 3 ms / 0.32 s | 5.6 ms | < 0.5 ms |
| 100 | none | 9.1 ms | — | — | — |
| **100** | **margin (seated)** | **32 ms** | **42 ms / 2.9 s** | **49 ms** | **< 2 ms** |
| 400 | margin | 167 ms | 0.30 s / 6.3 s | 173 ms | < 10 ms |

The margin tiebreak costs about 3.5× because it runs a full margin solve on tied tiles. Ties
are common once a contract is decided in every sampled world. At 100 worlds the Demon takes
about 0.45 s of a hand's 14 plays. Demon vs E[Q] games ran 10.8 s each on average, both players
included (max 15.4 s). Games against Walt and Zeb ran about 72 s, which is mostly their time.
A one-pair faceoff fits the worker's `--budget-s 200` under `timeout 300` with room to spare.
The first build from the pinned trees takes about 40 s per machine.

## Pinned

`demon-pimc-n100-b88c5432`: mk5 commit `b88c5432`, with the git trees of `silas/`
(`75b67c11…`), `scenario/core/` (`11735eb0…`) and `tournament/demon/` (`40738d38…`). The worker
archives those trees (`git archive <tree>`) and builds them with `cargo build --release
--locked`. It never builds the checkout's copy, so later edits to Silas or Walt cannot move the
Demon. Changing a knob or the code means a new participant id. In a shallow clone where a tree
is missing, the worker fetches the pinned commit first.

The compact-table builds are `demon-pimc-n100-18ac096d` and
`demon-pimc-n400-18ac096d`, pinned to commit `18ac096d` and trees
`silas/` `c10cb8b3…`, `scenario/core/` `11735eb0…`, `tournament/demon/` `2431c1f1…`.
The original participant remains immutable. The new 100-world build preserves
its play for a given request and seed; 400 worlds is an experimental challenger.
The roster includes both; their evidence is in [[demon-search-tables-2026-10-14]].

The distinct-world builds are `demon-pimc-n100-228812ff` and
`demon-pimc-n1600-228812ff`, with source pin `228812ff` (trees `silas/` `9dff8ca1…`,
`scenario/core/` `66a68a2a…`, `tournament/demon/` `35495a96…`). Both participant records
are published and freshly verified at tournament-store revision `382c4233`.
The 1,600-world pin builds from its archived public trees and answers the protocol.
[[demon-power-2026-10-14]] reports their search and strength evidence.

The withdrawn bidding form is `demon-pimc-n100-bid7ff83929-4454f603`, pinned to
[[4454f603]] (trees `silas/` `2330f0f2…`, `scenario/core/` `61cd6a52…`,
`tournament/demon/` `1873a3de…`). Its published immutable record, fresh download
verification, held-out trial and cost are in [[demon-next-form-2026-10-14]]. It
uses 100 uniform play worlds and `pimc:16:0.875` bidding over all ten physics.
It is historical research only after [[demon-vs-zeb-2026-10-15]]'s domain audit;
the declaration reader also stays experimental.

The confirmed overcalling form is
`demon-pimc-n100-bidc102a137-straight-2e233d29`, source [[2e233d29]], with
trees `silas/` `2330f0f2…`, `scenario/core/` `61cd6a52…` and
`tournament/demon/` `9ddb6a58…`. It uses 100 uniform play worlds and
`pimc-over:64:0.875:0.5` bidding over the nine straight declarations. Its
immutable registry record, source rebuild proof and independent confirmation
are in [[demon-overcalling-2026-10-15]]. Existing participant bytes are unchanged.

## First results: a smoke test, not a ranking (2026-10-14)

Ladder `marks7`, one duplicate pair per faceoff, run on this Mac. The E[Q] games ran on MPS.
The Bradley–Terry table had 31 faceoffs in all, and other workers were adding records at the
same time. The chance baseline for every pair is 50%.

| Demon vs | duplicate pairs | games | Demon wins | marks/game (Demon − foe) | 95% CI |
|---|---|---|---|---|---|
| `eq-n10-fenced` | 10 | 20 | 11–9 | +0.35 | [−0.98, +1.68] |
| `walt-plunge-26043bf9` | 3 | 6 | 4–2 | +0.33 | too few pairs |
| `zeb-0.3.0` | 3 | 6 | 1–5 | −3.33 | too few pairs |

Bradley–Terry after these games: the Demon at 1539 [1391, 1687], anchored on production Walt
at 1500 (32 games, 16 wins).

How to read it:

- **Against E[Q].** The Demon and E[Q] n=10 look like the close cousins they should be: 11–9,
  with a marks interval that straddles zero by more than a mark either way. That is a sanity
  check, not a ranking.
- **Against Walt and Zeb 0.3.0.** Three pairs each tell us nothing yet. Zeb 0.3.0's 5–1 is the
  first number to watch, and the bar to beat is Walt's +0.74 marks/game against E[Q].

## Links

[[strategy-fusion-fence]] (the ruling; the Demon is in its inventory) · [[tournament]] (where
it sits) · [[scenario-player]] (the hub) · [[strategy-fusion]] · [[silas]] (the solver) ·
[[expected-q-value]] (its GPU cousin) · [[brainstorm-2026-10-14-ledger]] ("BR²u beats PIMC")
