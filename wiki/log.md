# Log

**This is a changelog, not a chronicle.** If an entry needs a paragraph, the paragraph belongs on a page — write the page and link it. An entry is the header line plus at most ~5 lines of pointers (touched pages, added, retired, one flag). The log carries no claim that isn't reachable through a link.

Entry prefix convention: `## [YYYY-MM-DD | shortsha | subject]` so `grep "^## \[" log.md log-archive.md` gives the full timeline.

**Adding an entry:** insert it directly under the `---` below the digest, newest first. Nothing else: a branch never rotates. Parallel additions merge with git's union driver (`.gitattributes`), so branches don't conflict here.

**Rotation** happens on main after a merge, by script: `python -u scripts/wiki_log_rotate.py` keeps the newest 10 entries once there are more than 15, moves the rest verbatim into [[log-archive]] and sets the digest's count; whoever runs it folds their gist into the digest.

## Digest (2026-04-09 → 2026-10-13, 242 archived entries)

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
- **Generated data to Hugging Face** (2026-10-12, 1 archived entry) — [[publish-data-to-huggingface]]: the 250k Zeb pour and the day's research artifacts public. Hub: [[scenario-player]].
- **The book face-off** (2026-10-12, 2 archived entries) — [[book-study-2026-10-12]] § The face-off: the same gates asked of random, jud, E[Q] n=10 and the field net; closed by a k1,n2048 re-plan, a counterexample the standard for siding against the book (#121).
- **The book corrected** (2026-10-12, 1 archived entry) — [[book-study-2026-10-12]] and the w42 pages: chance baselines everywhere; #122–#125 opened.
- **otis, the algebra, walt endgame, parlor** (2026-07-15 → 07-18, 6 entries) — [[otis]] [[otis-v0]] graded, the referees doctrine on [[count-fate-ledger]], [[belief-policy-value-algebra]] promoted, [[jud-v2-retrain-probe]], [[walt]] endgame solver (+3.00 over jud), [[parlor]] promoted.

---

## [2026-10-14 | 78bbad51 | silas: the computed call (#148): (bid (- 42 (risk t))); risk-budget's thirteen levels one rule, equal on all 1,184,040 hands at every least raise; (- a b), number definitions]

**Updated:** [[silas]] (§ The computed call), `silas/README.md` (§ Queries, Semantics, The auction), `silas/GAPS.md` (§ Closed). One core pin moved, `risk` added, `less-at-risk` removed (`silas/tests/core_pins.rs`). Tests: `silas/tests/computed.rs`, `silas/tests/bidding.rs`.
## [2026-10-14 | 501a086a | make-then-margin within a near-tie band (#120 follow-up): δ* = 0.01 misses both bars against exact ties (49.1% [47.7, 50.4] to 250; marks +0.02 [−0.050, +0.086]); A/A exact; δ = 0.02 53.4% exploratory]

**Added:** [[tie-band-2026-10-14]] (pre-registered at 501a086a; `Config.tie_band`, `Tie::Margin { band }`, arena `t=margin:<band>`, default 0). **Updated:** [[tie-margin-2026-10-14]], [[scenario-player]], `index-experiments.md`. Data: `zeb/tie-band/` in `jasonyandell/texas-42-zeb-research-2026-10` with SHA256SUMS.

## [2026-10-14 | 16964d90 | silas: functions are numbers (#144): (to-set), (bid), (least-raise) in comparisons; bid-only-enough one rule, lead-sets one sum; equal on all 1,184,040 hands; risk-budget waits on #148]

**Updated:** [[silas]] (§ Functions as numbers), `silas/README.md` (§ Semantics, Short forms, The auction), `silas/GAPS.md` (§ Closed). Thirteen core pins moved (`silas/tests/core_pins.rs`). Tests: `silas/tests/functions.rs`, `silas/tests/bidding.rs`.
## [2026-10-14 | c9852cf6 | the soft reading's replication against Zeb (#145): consistent +0.12 [−0.02, +0.27] fails its registered bar; uniform +0.23 holds; cost ≤1.7× plain]

**Updated:** [[reads-plays-soft-2026-10-14]] (§ Replication: pre-registration, results, ε sweep, cost), [[scenario-player]], [[reads-plays-2026-10-13]], [[lumps-voi-2026-10-13]], `index-experiments.md`. Data: `zeb/reads-plays-soft-rep/` in `jasonyandell/texas-42-zeb-research-2026-10`.

## [2026-10-14 | 7188cd90 | the reading softened (#145): no fallbacks; +0.23 marks/game at a net table, +0.30 / +0.29 against Zeb; restricted choice null]

**Added:** [[reads-plays-soft-2026-10-14]] (`Reading.eps`, `scenario/reads_soft.py`). **Updated:** [[scenario-player]], [[reads-plays-2026-10-13]], [[lumps-voi-2026-10-13]], `index-experiments.md`. Data: `zeb/reads-plays-soft/` in `jasonyandell/texas-42-zeb-research-2026-10`.

## [2026-10-14 | f1ec6d67 | the exploiter against make-then-margin Zeb (#77 × #120): G +0.50 → +0.29 [+0.24, +0.34], paired drop +0.21 [+0.14, +0.28], contract exploit unchanged; two-rung +4.1 points/hand, marks unsettled]

**Updated:** [[exploiter-endgames-2026-10-14]] (§ Cross-check results and pre-registrations), [[tie-margin-2026-10-14]] (§ Recommendation, Caveats). Code: `scenario/exploiter.py` (`--tie`, `compare`). Data: `zeb/exploiter-margin/` on the research repo.

## [2026-10-14 | 153a5d0c | the book's own table (#147): opponents who read the book player by the book change nothing for it (−0.04 [−0.27, +0.19] marks/game); plain Zeb beats such a reader +0.50 [+0.26, +0.74]]

**Added:** [[book-reads-by-the-book-2026-10-14]] (pre-registered at 153a5d0c). **Updated:** [[book-investigation-2026-10-14]], [[scenario-player]], `index-experiments.md`. Code: `arena/book_read.py` (`bookread:`), `arena/test_book_read.py`, the batch belief seam in `scenario/compiled.py`. Data: `book/reads-by-the-book/` on the research repo.

## [2026-10-14 | 0dab46ff | the book's encoding repaired (#146): ten rules back inside their headers; block b re-measured −1.41 marks/game (+0.47 [+0.33, +0.62]), 38.4% to 250]

**Updated:** [[book-investigation-2026-10-14]] (§ After the repairs: pre-registration, results), [[silas]] (§ The encoding's repairs), `silas/GAPS.md` (§ The book's silences). Twelve core pins moved (`silas/tests/core_pins.rs`). Data: `book/encoding-repairs/` on the research repo.

## [2026-10-14 | d7db9e36 | an exploiter against Zeb on three-trick endgames (#77): G = +0.50 [+0.42, +0.58] margin, neither sturdy nor shown exploitable; half is lowest-id ties after the contract is decided; +0.004 in contract probability]

**Added:** [[exploiter-endgames-2026-10-14]] (pre-registered at d7db9e36). **Updated:** [[scenario-player]], [[brainstorm-2026-10-14-ledger]], `index-experiments.md`. Code: `scenario/exploiter.py`, `scenario/tests/test_exploiter.py`. Outputs: `jasonyandell/texas-42-zeb-research-2026-10` `zeb/exploiter/`.

## [2026-10-14 | b0e90912 | brainstorm notes ledger: "Results so far" are ours, the +0.111 is not (+0.11 marks/game, CI includes zero); demon labels/critics fenced; criticality pilot; next: an endgame exploiter against Zeb]

**Added:** [[topics/brainstorm-2026-10-14-ledger|brainstorm-2026-10-14-ledger]] (claim-by-claim ledger, assessments of the exploiter and fusion-free AlphaZero, criticality histogram on two pours, one pre-registered recommendation). **Updated:** `index-topics.md`, [[strategy-fusion-fence]] (link). Cross-game lab cited from `~/code/plunge-walt-bridge/lab/`.

## [2026-10-14 | e94f857f | Silas ergonomics, round 2: a suit's face order before the declaration, least/most, coverage over a bidding policy's contracts with a stated play model]

**Updated:** [[silas]] (§ Ergonomics, round 2), `silas/README.md`, `silas/GAPS.md`. **Added:** `silas/examples/populations/` (#141's bidders and play model, one command instead of Rust).

## [2026-10-14 | 492fef99 | constellations resurrected as Zeb's late-hand table: a 0.5 µs form in the core, exact under a structural field, not under the net; the late-hand ladder settles by level 3]

**Added:** [[constellations-2026-10-14]]. **Touched pages:** [[scenario-player]] [[endgame-equivalence-census]]. Code: `scenario/core/src/constellation.rs`, `scenario/constellation.py`, `field_train` warm start. Outputs: `jasonyandell/texas-42-zeb-research-2026-10` `zeb/constellations/`.

## [2026-10-14 | 5ee71159 | Zeb reads the other seats' plays: consistent and uniform likelihoods both null in marks, restricted choice +0.10 [−0.15, +0.36]]

**Added:** [[reads-plays-2026-10-13]] (`Config.reading`, `scenario/reads_plays.py`). **Updated:** [[scenario-player]], [[lumps-voi-2026-10-13]], `index-experiments.md`. Data: `zeb/reads-plays/` in `jasonyandell/texas-42-zeb-research-2026-10` with a SHA256SUMS manifest.

## [2026-10-14 | 902786f5 | the book against Zeb, investigated: the method holds, 44% of the loss is the encoding, the book loses its own game (points to 250) but wins it as Zeb's tie-break (53%, replicated); readers of plays change nothing]

**Added:** [[book-investigation-2026-10-14]] (pre-registered; data at texas-42-zeb-research-2026-10 `book/investigation/`).
**Updated:** [[book-head-to-head-2026-10-13]] (corrected in place: the chance baseline, the forecast, the per-rule ranking, block c, b-rb's one gaining rule, the concentration list), [[book-study-2026-10-12]] § The book at the table, [[scenario-player]], index-experiments.
**Code:** `arena/engine.py` `points_to_win` and `arena/auction.py` `settle_points` (the book's game), `arena/book_play.py` `ties=union` and `w<eps>`, `SilasPolicy.bid`/`declare` take the match's scoring; the reading Zeb of 3262eea0 merged.
**Issues:** #146 (the encoding's repairs), #147 (a reader by the book's policies).

## [2026-10-14 | 07abdbbf | make, then margin: Zeb's exact ties broken by the plan's margin (#120); marks null, +4.5 points a hand, book agreement 52.2% → 54.3%]

**Added:** [[tie-margin-2026-10-14]] (`Config.tie`, arena `t=margin`, default off). **Updated:** [[scenario-player]], `index-experiments.md`. Data: `zeb/tie-margin/` in `jasonyandell/texas-42-zeb-research-2026-10` with SHA256SUMS.

## [2026-10-13 | 0241f66f | the strategy-fusion fence: every clairvoyant solver fenced by one grep token, renamed to warn, Silas's gated behind a cargo feature and refused in policies, a guard test; information-state solving welcome]

**Added:** [[strategy-fusion-fence]] (the ruling, the inventory, the sanctioned uses). **Updated:** [[silas]], [[the-wall]], [[splat-search-bounds]], [[strategy-fusion]], [[intermediate-ai]], `index-topics.md`, `CLAUDE.md` (Hard rules). Guard: `src/tests/architecture/strategy-fusion-fence.test.ts`.

## [2026-10-13 | 53294297 | lumps by a hidden tile: 96.5% robust beyond the bar, the 5-5 the tile most worth knowing (0.39 pp of make beyond the null), restricted choice exact in Silas]

**Added:** [[lumps-voi-2026-10-13]] (`scenario/lumps_voi.py`). **Updated:** [[metasplat-2026-10-13]], [[scenario-player]], `index-experiments.md`. Data: `zeb/lumps-voi/` in `jasonyandell/texas-42-zeb-research-2026-10` with a SHA256SUMS manifest.
