# Log

**This is a changelog, not a chronicle.** If an entry needs a paragraph, the paragraph belongs on a page — write the page and link it. An entry is the header line plus at most ~5 lines of pointers (touched pages, added, retired, one flag). The log carries no claim that isn't reachable through a link.

Entry prefix convention: `## [YYYY-MM-DD | shortsha | subject]` so `grep "^## \[" log.md log-archive.md` gives the full timeline.

**Adding an entry:** add it as a new file, `wiki/log.d/<UTC timestamp>-<desk>-<slug>.md`, with exactly one entry. Never edit this file: a branch never rotates or stitches, so no two sessions share a file.

**Stitching and rotation** happen on main after a merge, by script: `python -u scripts/wiki_log_rotate.py` first stitches the `log.d/` fragments in here, newest first, then keeps the newest 10 entries once there are more than 15, moves the rest verbatim into [[log-archive]] and sets the digest's count; whoever runs it folds their gist into the digest.

## Digest (2026-04-09 → 2026-10-16, 333 archived entries)

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
- **Silas reasons from evidence** (2026-10-13, 6 archived entries) — [[silas]]: the aggregate (#135), the bid as evidence (#134), the plays as evidence (#128), the trump-set rules rare not mis-guarded (#141), the private adjudications carried into the book fixture; [[endgame-equivalence-census]] ported (worlds never compress).
- **The book measured, the solver bounded** (2026-10-13, 6 archived entries) — [[book-study-2026-10-12]]'s re-price 3, the head-to-head ([[book-head-to-head-2026-10-13]]) and the metasplat ([[metasplat-2026-10-13]]); [[splat-search-bounds]]; [[silas]] reconciles Hand 26 (#143); the TS engine's mark-bid cap lifted (#142).
- **Reading, ties, the fence, the investigation** (2026-10-13 → 10-14, 7 archived entries) — lumps by a hidden tile ([[lumps-voi-2026-10-13]]); [[strategy-fusion-fence]]; make-then-margin ([[tie-margin-2026-10-14]], #120); the book investigated ([[book-investigation-2026-10-14]]); Zeb reads plays ([[reads-plays-2026-10-13]]); constellations ([[constellations-2026-10-14]]); [[silas]] ergonomics round 2.
- **The overnight shift** (2026-10-14, 6 archived entries) — [[brainstorm-2026-10-14-ledger]]; the exploiter ([[exploiter-endgames-2026-10-14]], #77) and its margin cross-check; the book's encoding repaired (#146) and its own table (#147, [[book-reads-by-the-book-2026-10-14]]); the reading softened ([[reads-plays-soft-2026-10-14]], #145).
- **Replications and Silas numbers** (2026-10-14, 7 archived entries) — the soft reading's two replications (uniform +0.37, consistent fails; [[reads-plays-soft-2026-10-14]]), the tie band and its replication ([[tie-band-2026-10-14]], closed), the farmboy field ([[field-eps-play-2026-10-14]], null), [[silas]] functions as numbers (#144) and the computed call (#148).
- **Generated data to Hugging Face** (2026-10-12, 1 archived entry) — [[publish-data-to-huggingface]]: the 250k Zeb pour and the day's research artifacts public. Hub: [[scenario-player]].
- **The book face-off** (2026-10-12, 2 archived entries) — [[book-study-2026-10-12]] § The face-off: the same gates asked of random, jud, E[Q] n=10 and the field net; closed by a k1,n2048 re-plan, a counterexample the standard for siding against the book (#121).
- **The book corrected** (2026-10-12, 1 archived entry) — [[book-study-2026-10-12]] and the w42 pages: chance baselines everywhere; #122–#125 opened.
- **Desks, the license line, the Demon, the book's first measurements** (2026-10-14, 6 archived entries) — [[standing-desks]], the license line ([[strategy-fusion-fence]] inventory), [[the-demon]], [[book-partners-read-2026-10-14]] (#150), the header audit (#153; [[book-assumptions]]), [[book-small-n-2026-10-14]] (#154).
- **The tournament, the Book Authority, Zeb 0.3.0** (2026-10-14, 6 archived entries) — [[tournament]] (production Walt, Zeb, fenced E[Q] on a Bradley–Terry ladder), [[book-authority]] and [[book-assumptions]] (the charter and eleven assumptions), [[zeb-030-confirmation-2026-10-14]], [[regulars-table-2026-10-14]], the fly connectome as a player, [[legacy-42-bots]].
- **otis, the algebra, walt endgame, parlor** (2026-07-15 → 07-18, 6 entries) — [[otis]] [[otis-v0]] graded, the referees doctrine on [[count-fate-ledger]], [[belief-policy-value-algebra]] promoted, [[jud-v2-retrain-probe]], [[walt]] endgame solver (+3.00 over jud), [[parlor]] promoted.

- **Match bidding, readings and exact search** (2026-10-14, 6 archived entries) — [[match-aware-bidding-2026-10-14]], [[match-equity]], [[publish-data-to-huggingface]], [[silas]], [[deviant-field-2026-10-14]], [[demon-search-tables-2026-10-14]].
- **The reader without a plan, the archive's cleanup, the Demon's power** (2026-10-14, 7 archived entries) — [[no-plan-2026-10-14]], [[crisis-read-reprice-2026-10-14]], [[huggingface-assets]], [[demon-power-2026-10-14]], [[demon-next-form-2026-10-14]], [[the-demon]].
- **The Demon against native Zeb, match-aware overcalls** (2026-10-15, 6 archived entries) — [[demon-vs-zeb-2026-10-15]], [[demon-zeb-attempt-1]], [[the-demon]], [[match-aware-bidding-2026-10-14]], [[match-equity]], [[legacy-42-bots]].
- **The ledger, the count lapse, fair clocks, the rescue** (2026-10-16, 6 archived entries) — [[silas]] (#130 slice 4, the arena ledger), [[count-lapse-2026-10-15]], [[fair-walt-2026-10-16]], [[branch-rescue-2026-10]], [[the-demon]].
- **The Demon's exact bidding, Lem's first audit** (2026-10-16, 6 archived entries) — [[the-demon]] (exact declaration elimination, exact bid bounds and support, the preemption and early-opening sweeps), [[lem-field-net-audit-2026-10-15]], [[branch-rescue-2026-10]].
- **The Demon's early forty + the auction read** (2026-10-16, 6 archived entries) — the Demon's higher early opening bids selected and confirmed against Zeb 0.4.1, exact root reuse under FAIR CLOCKS ([[demon-ascent-2026-10-16]], [[demon-instructions-2026-10-16]]); the auction read pre-registered (#117, [[auction-read-2026-10-16]]) and Silas cross-game loose ends ([[silas]]).

---

## [2026-10-16 | lem | G1 pre-registered: belief probes from the field net to the teacher's exact ownership posterior]

- [[lem-belief-probe-2026-10-16]]: targets P(seat holds tile) exact under shown voids minus the counting prior; ridge probes from raw inputs and both hidden layers; deal-disjoint 20k/20k; shuffled chance; four predictions (the inputs state most of it; the hidden layers add nothing beyond them). `lem/probe/belief.py`.

## [2026-10-16 | lem | The paper: a living draft and its Gaps list as the desk's work order]

- `lem/paper/README.md` (CC0): "Interpreting a distilled imperfect-information player against an inspectable teacher"; the lab, the teacher, the students, the concept audit, belief probes [gap], concept formation across data size, the feature-count curve, models of others [gap], honest limits; nine gaps G1–G9. [[lem-the-teacher]] § The paper links it.

## [2026-10-16 | book | Silas: a seat's hand law; the losing 31 and the 35's two offs as queries (#134)]

- [[silas]] gains `Belief::hand_law` (and `--law ROLE`): a read seat's whole hand as a belief of its own, held to brute force; `silas/GAPS.md` § 4 narrowed a third time.
- W42-p054-a: a 35 in fours read as `risk-budget` is two offs 1,587/1,751 (77/646 unread); a 36 in blanks, one off certainly (Hands 33 and 32).
- W42-p054-b: a losing 31 lifts "a double" to 24,033/24,989 as `bid-only-enough` reads it, to 1 as `risk-budget` (the deal of Hand 33, our auction).
- **All desks:** stale venvs must rebuild scenario_core, which embeds silas: the evaluator now orders clauses at compile time. `scripts/check` builds its own.

## [2026-10-16 | book | Seed-scatter caveats on single-arm margins against a one-seed field (#190)]

- One line each on [[rematch-zeb-041-vs-demon-2026-10-15]], [[fallible-field-2026-10-15]] and [[count-lapse-2026-10-15]]: their single-arm margins exclude planner-seed scatter of about ±0.1 marks/game, and the paired differences on shared seeds are unaffected. Results whose ends lie inside the scatter: the rematch's +0.330 [+0.055, …] marks, #151's Z10 and Z20, and #172's Z5. The Mayor asked for this.

## [2026-10-16 | book | Seed-varied Zeb seats, and #117 re-registered against a seed-varied field (#190)]

- `arena.seed_varied.SeedVaried`: a Zeb seat that plays each hand at a planner seed picked from a pool by the deal (keyed as a set of hands; FNV plus splitmix64's mixer, because FNV's low bits alone reached 4 of 16 seeds). It plays the same seed in every arm and both halves, and at a pool of one it is the plain player byte for byte. Its `.walt` is a pooled planner, so RegularsPlay and BookPlay get the deal's seed too.
- [[seed-varied-2026-10-16]] is pre-registered: #117's K, U and Za against a seed-varied field, primary K read against an honest A/A; 352 batches.

## [2026-10-16 | lem | Pour A pre-registered: positions on all nine playable declarations, the book's menu declaring, a labelled forced share]

- [[lem-pour-a-2026-10-16]]: 250,000 deals in 50 slot-sized chunks on the Mac (Jason: "the Mac is available for a pour"); the book's menu (#155) declares, one pip deal in eight forced to doubles or no trump and listed in the manifests; pips over the shipped net, doubles/no-trump over the one-rung Walt; `lem/pour/a.py`.
- [[lem-the-teacher]] stage 2 links it. The field-v3 canon (branch `lem/field-v3-r2`) is with the Mayor; the first v3 net trains on this pour.

## [2026-10-16 | book | The auction read: +0.17 marks a game by the known method, but the A/A moved too (#117)]

- Results on [[auction-read-2026-10-16]]: 176 of 176 batches, none dropped. Primary K = +0.17 [+0.06, +0.29] marks/game clears its bar. The A/A (plain Zeb at seed 1) also clears, +0.12 [+0.02, +0.23], so K − Za = +0.05 [−0.09, +0.19]. The read does not cost, and its gain is not separable from planner-seed scatter. U +0.07, K − U +0.10 (both nulls); to 250, PK 53.4%. The A4 row is updated. #190 files seed-varied fields. Data: HF `book/auction-read/` (7877cd92, SHA verified).

## [2026-10-16 | demon | Repository extraction preparation]

- Added [[demon-repository-prep-2026-10-16]]: native export, dependency/license audit and complete-player cutover gates; routed from [[the-demon]].
- Follow-up [#188](https://github.com/jasonyandell/mk5-main/issues/188); no new repository or production cutover.

## [2026-10-16 | lem | The audit read against Zeb's earlier nets (exploratory): concepts rise with data, the play stays dense]

- [[lem-field-net-audit-2026-10-15]] § Exploratory: clean share 0.297 → 0.519 → 0.597 across the 5k, 50k and 250k nets (one net per size, chance 0.000 each), declaration detectors flat (68, 71, 69), knee 176 → 170 → 165; every "master" pole is a trump master; the trick-position poles are conjunctions. Read-outs in `audit/reads.md`, series on HF.

## [2026-10-16 | book | The full declaration menu at the arena: doubles as trumps and no trump, opt-in and provisional (#155)]

- **Substrate.** The arena engine plays whole doubles-trump and no-trump hands by the rules, checked against the rules written out independently of `forge.oracle.tables`. Zeb's plan on the random field decides under both, with the compiled core equal to the Python referee at every decision of a whole hand (`arena/test_declarations.py`). Deployed Zeb's field net plays pip trumps only and refuses decls 7–9. At a menu table a net seat is `DTNT=random` (`arena.walt_play.MenuWalt`): its random-field twin plays those hands, and it is the net byte for byte under a pip trump. Every Ch. 9 number from it is provisional until a net covers them (#177). The engine refuses a table where a bidder may declare what a seat cannot play (`arena.play.check_menu`).
- **The bidder.** `MenuBidder` (opt-in, `menu` in the CLI) is `HeuristicBidder` below four doubles. From four it declares doubles when the best suit leaves a high off and no trump when the hand lacks the double-six and every high off and no trump prices better (catalog rows W42-p075-b, W42-p078-b, W42-p079-a, W42-p080-a). It declares Hands 23–25 as the book does and keeps the book's suit on its other hands of four doubles. It changes the bid or the declaration on 0.8% of hands. `HeuristicBidder` is pinned byte-identical by digest.
- **Silas.** `bidding/menu.silas` is `MenuBidder`, one digest on all 51,108 hands of four doubles. `risk-budget` declares doubles or no trump where the table offers them, and is unchanged where only pip trumps are offered. [[book-assumptions]] A10, [[scenario-player]]
- **All desks:** stale venvs must rebuild scenario_core, because lib/bidding.silas is embedded in it. `scripts/check` builds its own; a local `scratch/venv` fails `arena/test_book_play` until its core is rebuilt.
