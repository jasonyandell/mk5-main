---
title: "The strategy-fusion fence: clairvoyant solvers fenced, information-state solvers welcome"
kind: topic
first_seen: 2026-10-13
last_updated: 2026-10-13
status: active
---

**STRATEGY-FUSION FENCE.** This page is where the token points. Grep for `solver`,
`double dummy`, `dd`, `perfect information`, `minimax`, `oracle` or `clairvoyant` and
every usable API answers with the token and a link here.

## The ruling (Jason, 2026-10-13)

- **Allowed:** a "perfect solver" that **enumerates the information states**: an exact
  imperfect-information solve that chooses one action per information set and never one
  per world. Perf work on it is always on the table.
- **Fenced:** strategy fusion, meaning a clairvoyant, perfect-information or double-dummy
  value used per world and then averaged. In Jason's words: "Demonic. Strong but wrong.
  The church sanctions demon summoning only in specific circumstances. A showcase where
  it is clearly more appropriate is fine if fenced. Better if replaced, but the fence is
  good."
- **Findable by grep.** "Agent will find it by grep, not chunk. 'Oh there's a DD solver,
  I'll just use that' is entirely fair play by a future model." So the fence carries one
  fixed token, `STRATEGY-FUSION FENCE`, wherever a clairvoyant API is defined or called.

Why: E[max] ≥ max E. A per-world best line credits each world with a play no seat that
cannot see the hands could choose ([[strategy-fusion]]). The oracle taught every student
to play as if it could see all four hands, which is [[the-wall]]; `CLAUDE.md` calls it
*algorithmia non grata*.

## What to use instead

An information-state solve: one choice per information set, the same in every world the
seat cannot tell apart.

- **Silas** ([[silas]]): `(lay-down)`, `(certain-make)` and `(certain-set)`, verified from
  `silas/src/outcome.rs`. `(lay-down)` is one fixed sequence of the viewer's tiles that wins
  every trick in every world against every reply. `(certain-make)`/`(certain-set)` are an
  and-or search over the viewer's frames, memoized on the public record. Because every
  play is public, the viewer's information state is exactly the frame, so the viewer
  picks one tile per frame and that tile serves the whole fiber. The other three seats,
  his partner included, range over every tile some world lets them play. That gives
  them more power than any real seat has, so the guarantee is conservative, never fused.
- **Zeb / Walt** ([[scenario-player]]): one action across the whole cell, values added
  over distinguishable continuations. A thinned sampled cell is clairvoyant, and the cell
  floor guards it on sampled fibers. An enumerated fiber is the exact posterior.
- **walt** endgame solver ([[walt]], `walt/solver.py`) and texas-42's `rob` over
  `arena/rob_play.py`: exact information-set solvers.

## The fence, in layers

1. **The token.** `STRATEGY-FUSION FENCE` sits at the top of every module that defines a
   clairvoyant API and in the doc comment of each public item: the type, its constructor
   and the predicate. It is also at the top of every file that calls one, stating why that
   use is sanctioned. At 2026-10-13, 104 code files carry it.
2. **Names that warn at the call site** (renamed everywhere, no aliases):

   | was | now |
   |---|---|
   | `silas::Solver`, `silas/src/solve.rs` | `silas::ClairvoyantSolver`, `silas/src/clairvoyant.rs` |
   | `silas::grade_line` | `silas::clairvoyant_grade_line` |
   | `(dd-make)` (Access `World`) | `(clairvoyant-dd-make)` (Access `Clairvoyant`) |
   | `silas/examples/book/solve_grade.rs`, `tests/solve.rs` | `examples/book/clairvoyant_grade.rs`, `tests/clairvoyant.rs` |
   | `src/game/ai/minimax.ts`, `minimaxEvaluate`, `MinimaxConfig`/`Result` | `src/game/ai/clairvoyant-minimax.ts`, `clairvoyantMinimaxEvaluate`, `ClairvoyantMinimaxConfig`/`Result` |
   | `scripts/minimax-eval.ts`, `minimaxFullGame` | `scripts/clairvoyant-minimax-eval.ts`, `clairvoyantMinimaxFullGame` |

   Forge's names (`forge.oracle`, `Stage1Oracle`, `ClairvoyantPolicy`) already say
   "oracle" or "clairvoyant" and were not renamed. `LensPlay` and the E[Q] functions
   carry the token instead.
3. **A compile-time gate.** In Silas, the exports of `clairvoyant.rs` and the predicate
   `(clairvoyant-dd-make)` exist only under the non-default cargo feature `clairvoyant`.
   The crate's tests and its showcase enable it explicitly, through a dev-dependency of
   the crate on itself. `scenario/core` (`default-features = false`) does not get it. In a
   plain build a query that names the predicate is refused as unregistered, and the
   refusal names the fence. The module itself still compiles, because the refuter inside
   `(certain-make)` reads it as a bound (below).
   No gate for the TypeScript minimax: the game's intermediate AI imports it on the live
   path ([[intermediate-ai]]), so its fence is the token, the name and the guard test.
   No gate for forge either: it is a finished era, and every one of its entry points
   carries the token.
4. **Policies cannot read it.** Silas already refused World atoms in a policy and in viewer
   evaluation. The clairvoyant predicate now has its own access, `Access::Clairvoyant`, and
   its refusal names the fence: `rule r uses the clairvoyant predicate clairvoyant-dd-make;
   a policy sees only its viewer (STRATEGY-FUSION FENCE: …)`. The refusal holds through a
   definition too. Test: `silas/tests/outcome.rs`,
   `a_policy_that_reads_the_clairvoyant_predicate_is_refused_at_the_fence`.
5. **The guard test.** `src/tests/architecture/strategy-fusion-fence.test.ts` (Vitest) reads
   every code file git sees, tracked or new (`.py .rs .ts .tsx .js .mjs .cjs .svelte .sh .toml .silas`). It
   fails if a file names a clairvoyant API without the token. It also fails if a defining
   module lacks the token in its first lines, if a guarded pattern stops matching anything,
   if an old unwarning name (`(dd-make)`, `minimaxEvaluate`, `fn grade_line`) comes back,
   if Silas's default features gain `clairvoyant`, or if this page or the `CLAUDE.md` line
   goes missing. `scripts/check` runs it in `ts-unit`. It also runs as its own suite,
   `fence`, which any code change triggers under `--fast`.

The guard does not catch a new hand-written double-dummy search that uses none of these
APIs. It also does not catch code that reads oracle-labeled data files directly. Those
cases rely on the `CLAUDE.md` hard rule and on review.

## Inventory (2026-10-13)

(A) is clairvoyant per world: strategy fusion if its value is used as a value. (B) is an
information-state solver, which is allowed. (C) is only a mention.

| Where | What | Class | Fence |
|---|---|---|---|
| `silas/src/clairvoyant.rs` | `ClairvoyantSolver`, `clairvoyant_grade_line`, `OpenPosition` and friends: exact double-dummy alpha-beta | A | token, names, `clairvoyant` feature |
| `silas/src/outcome.rs` | `(clairvoyant-dd-make)`: per-world double-dummy make | A | token, name, feature, `Access::Clairvoyant`, policy refusal |
| `silas/src/outcome.rs` `refuted` | the solver inside `(certain-make)`: one world where double dummy fails refutes every viewer strategy | A, sanctioned (a bound, never a value) | token |
| `silas/src/outcome.rs` | `(lay-down)`, `(certain-make)`, `(certain-set)`, `forces`, `lay_down` | **B** | none needed |
| `silas/examples/book/clairvoyant_grade.rs` | grades the book's narrated lines of Hands 28–35 | A, sanctioned showcase | token, `required-features` |
| `silas/tests/{clairvoyant,outcome,book_leads,book_repaired,book_odds,cli}.rs`, `tests/common/leads.rs` | tests of the solver; bound and book checks | A, sanctioned | token |
| `src/game/ai/clairvoyant-minimax.ts`, `monte-carlo.ts` | the intermediate AI: PIMC, clairvoyant minimax per sampled world, averaged | A, tolerated (opponent AI only; better if replaced by Zeb) | token, names; no gate (live path) |
| `src/tests/ai/clairvoyant-minimax.test.ts` | its tests | A | token |
| `scripts/clairvoyant-minimax-eval.ts` | engine vs oracle cross-validation on one known deal | A, sanctioned | token, name |
| `forge/oracle/` (`solve.py`, `campaign.py`, `generate.py`, `__init__.py`) | the GPU oracle: exact perfect-information solve per deal | A, oracle era | token |
| `forge/eq/` (`oracle.py` `Stage1Oracle`, `generate/` E[Q]), `forge/zeb/` (`eq_player`, `mcts`, `oracle_value`, …), `forge/cli/`, `forge/modal_app.py` | the network distilled from the oracle, and E[Q]: per-world Q averaged | A, oracle era | token (39 files) |
| `arena/lens_play.py` and callers (`cli`, `jud_play`, `jud_search`, `midhand_eval`, `slough_override`, `fate_slough_play`, `test_snapshots`) | `LensPlay` = E[Q] n=10, the old champion | A, sanctioned baseline (what Zeb is graded against) | token |
| `otis/tiedroll.py` | `ClairvoyantPolicy`: the fusion-gap baseline | A, sanctioned (measures fusion) | token |
| `champion/` (10 files incl. `belief_bidder`, `play`, `jud_net`, `bidder`), `gus/` (2), `w42/` (13), `burl/` (4), `lem/` (4), `scripts/validate_eq_computation.py` | oracle-era players, probes and datasets built on E[Q]/Stage1Oracle | A, oracle era | token |
| `champion/optimism_meter.py` | oracle make-rate vs realized | A, sanctioned (measures fusion) | token |
| `scenario/planner.py`, `scenario/core` (Walt/Zeb) | one action per cell; thin sampled cells floored | **B** | none needed |
| `walt/solver.py`, `arena/rob_play.py` | exact information-set solvers | **B** | none needed |
| `scenario/core/src/mold.rs`, `scenario/atlas.py` | the shape (position counts) of a known deal's perfect-information tree; the labels are Walt's | C | none |
| `scenario/hf.py`, `scenario/lumps_voi.py`, `zeb/src/lib.rs`, `arena/walt_play.py`, `champion/value_bidder.py`, `margin_net.py`, `bid_decoder.py`, `scripts/check` | mentions (contrast, provenance, a test-speed note) | C | none |

## Sanctioned uses, and why each is "clearly more appropriate"

- **A bound that refutes** (`refuted` in `(certain-make)`). Double dummy is an upper bound
  on every strategy of the viewer's. One world where it fails proves the guarantee false.
  It never supplies a value or a choice.
- **Measuring how loose the bound is.** `tests/outcome.rs` holds the information-state
  guarantees under the double-dummy bound. Hand 26 is where the two part: double dummy
  makes in all 756,756 worlds, but the bidder alone cannot force the make. The other
  measurements of this kind are [[splat-search-bounds]] § Average DD against the planner,
  `champion/optimism_meter.py` and otis's fusion gap.
- **Book claims about fully known positions.** `examples/book/clairvoyant_grade.rs` and
  `tests/book_leads.rs` read the book's narrated deals with every hand on the page, where
  the book itself argues from the open hands. A short grade is a question for the book,
  not a verdict ([[silas]]).
- **Counterexample deals** (`tests/book_repaired.rs`). In one known deal, a lead that fails
  even double dummy fails every real line. The deal is then played out with every
  bidder-side play forced.
- **"The only way" claims** (`tests/book_odds.rs`, Hands 16 and 34). The book names the
  layouts that set the bid, and `(clairvoyant-dd-make)` checks that those are exactly
  the layouts where even double-dummy play sets. With two tricks left each seat has at
  most one real choice, and the claim is itself one about layouts, so per-world best play
  is its own standard.
  Those are the "cannot be set double dummy" claims the ruling names.
- **Baselines.** E[Q] n=10 (`LensPlay`) is the opponent Zeb is graded against. The oracle
  era stays as a measuring stick ([[expected-q-value]], [[the-oracle]]).
- **Tolerated, not sanctioned:** the game's intermediate AI ([[intermediate-ai]]) plays by
  PIMC. It is fenced and named, and it is better replaced by Zeb.

## Replaced

No (A) use in Silas's book work could be replaced without losing what it checks. The
replacement the ruling asks for had already been made where it fits: Hand 26's "cannot be
set" claim is `examples/book/cannot-be-set.silas`, which states `(certain-make)`, not the
double-dummy bound. That is how it found the claim false as the bidder's own guarantee
though double dummy makes everywhere. The remaining uses check a claim *about* the
double-dummy outcome, or use it as a bound, so an information-state predicate is a
different statement there. They stay fenced.

No core pin changed: no `.silas` file or prelude names the renamed predicate
(`silas/tests/data/core-pins.txt` untouched).

## Links

[[strategy-fusion]] | [[the-wall]] | [[silas]] | [[splat-search-bounds]] | [[belief-policy-value-algebra]] | [[scenario-player]] | [[intermediate-ai]] | [[the-oracle]] | [[expected-q-value]]
