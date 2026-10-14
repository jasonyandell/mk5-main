# Log

**This is a changelog, not a chronicle.** If an entry needs a paragraph, the paragraph belongs on a page — write the page and link it. An entry is the header line plus at most ~5 lines of pointers (touched pages, added, retired, one flag). The log carries no claim that isn't reachable through a link.

Entry prefix convention: `## [YYYY-MM-DD | shortsha | subject]` so `grep "^## \[" log.md log-archive.md` gives the full timeline.

**Adding an entry:** insert it directly under the `---` below the digest, newest first. Nothing else: a branch never rotates. Parallel additions merge with git's union driver (`.gitattributes`), so branches don't conflict here.

**Rotation** happens on main after a merge, by script: `python -u scripts/wiki_log_rotate.py` keeps the newest 10 entries once there are more than 15, moves the rest verbatim into [[log-archive]] and sets the digest's count; whoever runs it folds their gist into the digest.

## Digest (2026-04-09 → 2026-10-13, 230 archived entries)

What the log recorded, phase by phase. The story itself lives in the hubs and trails linked here; this is only a map of when the wiki ingested what.

- **LEM** (2026-04-09 → 04-17, ~17 entries) — the wiki's first trail: [[stage-0-adapter]] v1→v10, the STaR plateau at 38–48%, the Gemma→[[qwen3-1.7b]] pivot, [[qwen3-14b]] capacity, the SFT mask fix. Hub: [[lem]].
- **Burl** (2026-04-18 → 04-23, ~10 entries) — sibling project kickoff: Moves 1–4, iters 0–5, the reasoning-coherence bottleneck named ([[candlewax]]), wax_museum, [[belief-trajectory]]. Hub: [[burl]].
- **Gus** (2026-04-20 → 04-22, ~10 entries, interleaved with Burl) — [[joint-world-tensor]] through the [[lamir1-ceiling]] to Gus FINAL in five days: v0→v3 students, blunder detector, belief calibration, co-train falsified. Hub: [[gus]].
- **Burl harvests + perf sprint** (2026-04-25 → 04-28, ~25 entries) — STaR run-3/3b/3c (preserve-thoughts is load-bearing), resumable checkpointing, the 2000-decision harvest, and the [[perf-sprint]] playbook's many small revisions plus [[burl-perf-phase0]]–phase3.
- **burl-lab / burl-chat / microscope + w42** (2026-04-30 → 05-07, ~65 entries; the log's densest stretch) — two parallel campaigns: the [[burl-lab]] workbench lane, and the [[w42]] book-validation campaign (claim ledger, branch atlases, four phases, waves 1–4, lens head-to-head). Trail: [[w42-book-validation]]. Also [[book-strategy-player]] design and the first wiki-curation pilot (hubs + trails).
- **Champion rungs** (2026-06-09 → 06-14, ~16 entries, after a five-week gap) — the [[champion]] ladder: arena harness, rungs #21–#28 (auction-conditioned belief the measured win; several honest nulls), [[jud]] direction and vocabulary, the Fable provenance corrections. Hub: [[champion]].
- **jud + the wall** (2026-07-05 → 07-06, 6 archived entries) — jud v0/v1 built and graded ([[jud]], [[champion]], [[w42-jud-v1]], [[w42-plateau-probe]]), the 2.38× arena perf pass, value-native endorsed via [[rank-vs-price]].
- **The wall named + the archaeology** (2026-07-06 → 07-10, 5 archived entries) — [[the-wall]] stated precisely (distill-for-what, [[candlewax]] concordance); the ~30-page era backfill + ~173-page staleness audit and anti-rot rules; [[w42-book-second-pass]]; the 162-experiment-page artifact audit (43 corrected); the log's changelog-not-chronicle rotation.
- **Partnership spine + consolidation** (2026-07-11, 6 archived entries) — the [[partnership-wall-research]] measurement spine (sampler audit, result vocabulary, review surface, four gates), the docs→wiki consolidation (game-of-42 + engine reference clusters), prior-sweep completion and review repairs. Hub: [[partnership-wall-research]]. Later 07-12+ entries remain below.
- **Blueprint/continuation ideation** (2026-07-12, 4 archived entries) — [[convention-aware-blueprint-search]] preserved unselected (book-seeded initialization, framing rebalance) and [[belief-weighted-jud-mcts]] synthesized; all IDEATED, none promoted to builds.
- **Continuation frontier + otis** (2026-07-13 → 07-14, 6 entries) — the wiki reorg (schema v2, routing trails, status truth), champion folded into [[jud]], the continuation-frontier research night (Stage 0 closes, Lane A validates, Lane B graded both rounds), and the [[count-fate-ledger]] formulated as the consumption object.
- **Walt, the scenario player, rebuilt** (2026-10-09 → 10-10, 6 archived entries) — [[scenario-player]] rebuilt from the mathematics, the compiled core, Walt beats E[Q] at play (+0.74), belief night 1 nulls, the atlas (splat + mold), the forecast's 20-point lie and the cell floor named. Hub: [[scenario-player]].
- **Zeb: the floor, the pour, the field net** (2026-10-10 → 10-11, 10 archived entries) — the cell floor and its dividend, the pour published ([[scenario-player]]), the field net at rung −1 and its two targets, the 50,000- and 250,000-deal pours, the live bidder graded against Plunge's bid book.
- **Zeb 0.1.0, the viz, research night, the first repair** (2026-10-11 → 10-12, 6 archived entries) — the live bidder graded against Plunge's frozen bid book, `@jasonyandell/zeb` 0.1.0 in WebAssembly, the splat visualizer and its region query, research night (T1/T1′/T2, Lean, the counterexample lab; [[research-night-2026-10-12]]), the field net graded by regret not agreement (#109). Hub: [[scenario-player]].
- **Zeb's foundation repairs and the convention overlay** (2026-10-12, 6 archived entries) — the floor only on sampled fibers (Zeb 0.2.0, #110), the rung-free sample seed (#112), the stochastic field (#108), O2 over the O1 fixed point on endgames ([[endgames-o2-vs-o1-2026-10-12]], #113), the posterior sampler (#111), the convention overlay. Hub: [[scenario-player]].
- **The book graded with the honest tools** (2026-10-12, 1 archived entry) — [[book-study-2026-10-12]]: the rules priced on the pour, the bid book, the conventions null in the arena.
- **Silas arrives** (2026-10-12 → 10-13, 8 archived entries) — [[silas]] moved from texas-42 and renamed, on Zeb's rules, the 21 gates written and priced, the record in the frame, the engineering pass; Zeb's reason report ([[scenario-player]]); server-only system authority ([[multiplayer-pattern]]).
- **The book as the book means it** (2026-10-13, 6 archived entries) — [[book-study-2026-10-12]]: the 14 faithful rules and their re-price (Zeb 54% vs 40%), the dead-double and trump-set encoding fixes, the reason report's grown space ([[scenario-player]]), [[silas]]'s solved outcomes and the first repaired rules (C10, C12B).
- **Silas grows up** (2026-10-13, 6 archived entries) — [[silas]]: cross-game slice 1 (the match frame), `scripts/check` and the conflict-free log, the perf pass (no GPU path), the short forms and coverage funnel, slice 2 (the auction), the book's second encoding batch (71 rows).
- **Generated data to Hugging Face** (2026-10-12, 1 archived entry) — [[publish-data-to-huggingface]]: the 250k Zeb pour and the day's research artifacts public. Hub: [[scenario-player]].
- **The book face-off** (2026-10-12, 2 archived entries) — [[book-study-2026-10-12]] § The face-off: the same gates asked of random, jud, E[Q] n=10 and the field net; closed by a k1,n2048 re-plan, a counterexample the standard for siding against the book (#121).
- **The book corrected** (2026-10-12, 1 archived entry) — [[book-study-2026-10-12]] and the w42 pages: chance baselines everywhere; #122–#125 opened.
- **otis, the algebra, walt endgame, parlor** (2026-07-15 → 07-18, 6 entries) — [[otis]] [[otis-v0]] graded, the referees doctrine on [[count-fate-ledger]], [[belief-policy-value-algebra]] promoted, [[jud-v2-retrain-probe]], [[walt]] endgame solver (+3.00 over jud), [[parlor]] promoted.

---

## [2026-10-13 | 53294297 | lumps by a hidden tile: 96.5% robust beyond the bar, the 5-5 the tile most worth knowing (0.39 pp of make beyond the null), restricted choice exact in Silas]

**Added:** [[lumps-voi-2026-10-13]] (`scenario/lumps_voi.py`). **Updated:** [[metasplat-2026-10-13]], [[scenario-player]], `index-experiments.md`. Data: `zeb/lumps-voi/` in `jasonyandell/texas-42-zeb-research-2026-10` with a SHA256SUMS manifest.

## [2026-10-13 | 9f14a106 | searching over splats: what is sound, what is tractable (dominance, admissible heuristics, lump bounds; probe at tricks 2-4)]

**Added:** [[splat-search-bounds]]. **Updated:** [[scenario-player]], [[metasplat-2026-10-13]], [[splat-visualizer]], `index-topics.md`.

## [2026-10-13 | 64b3db69 | the metasplat: Zeb's splat against the book's at every live firing, nine shapes fixed before counting; 44% of departures a make edge (27% dominant, 3% safer tail at a lower mean), 45% even make, 11% ties]

**Added:** [[metasplat-2026-10-13]] (`scenario/metasplat.py`, `reason.explain`, the page's book view). **Updated:** [[book-study-2026-10-12]] § Finer than a price, [[splat-visualizer]] § Book against Zeb, `index-experiments.md`. Data: `book/metasplat/` in `jasonyandell/texas-42-zeb-research-2026-10` with a SHA256SUMS manifest.

## [2026-10-13 | scratch/book-h2h | the book player against Zeb: the 18 rules as a Silas stack lose −1.88 marks/game with Zeb's tile where no rule fires; each departure from Zeb rolled out, −0.051 marks, +0.036 over chance, no rule gains]
**Added:** [[book-head-to-head-2026-10-13]] (pre-registered; data at texas-42-zeb-research-2026-10 `book/head-to-head/`).
**Updated:** [[book-study-2026-10-12]] (§ The book at the table), [[scenario-player]] (the book player; marks bids planned as 42), index-experiments.
**Code:** `arena/book_play.py` (book play, `SilasBidder`), `SilasPolicy.bid`/`declare` in `scenario/core`.

## [2026-10-13 | scratch/book-reprice-3 | the book's 18 rules as the book means them re-priced on the whole pour at current main, same method and bands: Zeb's pick is the book's move at 56% vs a 43% random floor (0.24); sides with 3, weak on 9, no relation on 2, between on 3, against none]

**Updated:** [[book-study-2026-10-12]] § The book's rules as the book means them (fidelity, prices, carriers, reading; summary verdicts in place); [[w42]], [[silas]], `index-experiments.md`, `CLAUDE.md`. Data: `book/book-reprice-3/` in `jasonyandell/texas-42-zeb-research-2026-10` with a SHA256SUMS manifest.

## [2026-10-13 | 2fe46194 | the TS engine no longer caps mark bids at four (#142): the enumerator's bound was a literal, the validator and the rules page already let one-mark steps run on; the 834-prefix fixture now agrees on five marks too]

**Touched pages:** [[silas]] (the auction's legality line; `silas/README.md`, `tests/auction.rs`). Engine: `src/game/core/rules.ts` (`highestNamableMarks`), `layers/base.ts`, URL codes for 4 and 5 marks; `src/tests/rules/bidding/mark-bid-ceiling.test.ts`. No config or variant: the cap was a bug.

## [2026-10-13 | 26f91c07 | Hand 26 "cannot be set" reconciled (#143): the setting deal is a world of the same position, after trick 2; the claims differ in the partner's play, not the position]

**Touched pages:** [[silas]] (`silas/GAPS.md` 8, `examples/book/cannot-be-set.silas`; `tests/outcome.rs` pins both numbers: the bidder's side 36 at double-dummy play, 31 with the 3-2 kept). No claim retracted; the position is now stated.

## [2026-10-13 | 49392914 | silas reads the plays as evidence (#128 for the play): consistent likelihood by default, uniform on request, counted exactly; S-0140-e's "you know" is 2/15 from the seat and 1 read under protect-the-count; C21's bait under crisis is 91/167, not certainty]

**Touched pages:** [[silas]] § The plays as evidence. `silas/README.md` § Semantics, The record as evidence, The `silas` command; `silas/GAPS.md` gap 1 narrowed again (a rule acting on a reading remains).

## [2026-10-13 | aefa35be | worlds never compress: endgame equivalence census ported from origin/worktree-equiv-census (aca3dacb, c3e1374c)]

**Touched pages:** [[scenario-player]] [[silas]]
**Added:** [[endgame-equivalence-census]]

## [2026-10-13 | 2442dcb0 | the private source adjudications carried into the book fixture: Hands 24, 26, 31, 32, 33, 34 read as adjudicated; EX-11's 53% is 43/81 on the book's model; Hand 28's 1-1 premise, not the sum, fails]

**Touched pages:** [[silas]] (`silas/examples/book/hands.json`, `leads.md`, `setter-leads.silas`; `tests/book_fixture.rs`, `tests/book_leads.rs` comments). No pinned value changed; the book tests pass.

## [2026-10-13 | 413e5b9d | silas reads the bid as evidence (#134; #128 for the auction): a belief conditioned on named seats' calls under their bidding policies, counted exactly, exact_given the policies; C12B's 194/285 moves a point either way, Hand 29's partner puts the 4-4 with the 30-bidder at 6003/8669]

**Touched pages:** [[silas]] § The bid as evidence. `silas/README.md` § The auction as evidence, The `silas` command; `silas/GAPS.md` gaps 1 and 4 narrowed; `silas/docs/cross-game.md` § 4 (amendment).

## [2026-10-13 | fd78c782 | silas aggregate (#135): card, points and (<= a b), three-valued by intervals; the bid rules rewritten with it, every call and declaration pinned equal on all 1,184,040 hands (risk-budget 150 → 20 µs); the crisis keeps the double with the most tiles out]

**Touched pages:** [[silas]] § The aggregate. `silas/README.md` § Semantics, Short forms, The auction; `silas/GAPS.md` gap 5 closed for what is built; `silas/docs/cross-game.md` § 4, Slice 2 as built (amendment).

## [2026-10-13 | 9f371174 | silas: the five trump-set rules are rare, not mis-guarded (#141)]
- [[silas]]: the book narrates three trump sets (Hands 32, 34, 35) and no position where the rules apply; over 3,000 random deals the three ruffs never fire (29 frames reach the ruff, one holds a trump), the feed fires 3 times and the lead 41; the pour fires the ruffs at 328 of 4,343,426 positions ([[book-study-2026-10-12]]).
- By population (sampled, 3,000 contracts each, `scratch/study141.rs`; the optimists and the marginal 30 are Silas bid policies, the sim is Rust): a ruff rule answers in 0 games under `bid-only-enough`, `risk-budget` and a marginal 30 (risk 13-14), 30 and 61 under an optimist on the top two tiles of two and of three suits; without the second-trump clause 25, 13, 18 and 146, 250.
- Added: positive pins for the last-seat ruff and the lead, a pin that the book's trump sets reach the ruff with no trump, and the coverage counts (`silas/tests/book_rules.rs`, `cli.rs`); README § Diagnostics corrected ("never fire" held only at 100 deals).
