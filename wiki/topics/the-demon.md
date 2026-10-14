---
title: "The Demon: a fenced CPU PIMC player, seated in the tournament as strategy fusion's measuring stick"
kind: topic
first_seen: 2026-10-14
last_updated: 2026-10-14
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
2. draws 100 worlds from its fiber with Walt's exact-uniform sampler
   (`scenario_core::fiber::sample_worlds`), or takes the whole fiber when it has 100 members
   or fewer;
3. solves each world double dummy with Silas's `ClairvoyantSolver` ([[silas]]): exact
   make-or-set of the contract, all four hands face up, every seat playing perfectly;
4. plays the tile that makes in the most worlds (sets, when defending). Exact ties go to the
   best summed double-dummy margin, then to the lowest tile id.

A seat with one legal tile plays it without solving anything. In bidding the Demon uses the
arena's `heuristic` bidder (Roberson risk budget), the same bidder E[Q] n=10 uses in the
tournament. The two fusion players therefore differ only in how they play.

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
- **The sampler and the rules.** Walt's fiber sampler, record and rules tables, plus
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
| `tournament/demon/` (Rust crate) | `src/lib.rs` `decide` (the four steps), `src/main.rs` (one JSON request per line; refuses a record that disagrees with the table's leader or points), `tests/demon.rs` |
| `tournament/demon_fenced_seat.py` | `DemonSeat`: the bidder and one binary process per team per game; `build` from the pinned trees |
| `tournament/participants.py` | `demon(...)`, `DEMON_N100` |
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

## Cost (measured 2026-10-14, Apple M-series, one core)

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
