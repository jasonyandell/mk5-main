# Log

**This is a changelog, not a chronicle.** If an entry needs a paragraph, the paragraph belongs on a page — write the page and link it. An entry is the header line plus at most ~5 lines of pointers (touched pages, added, retired, one flag). The log carries no claim that isn't reachable through a link.

Entry prefix convention: `## [YYYY-MM-DD | shortsha | subject]` so `grep "^## \[" log.md log-archive.md` gives the full timeline.

**Adding an entry:** add it as a new file, `wiki/log.d/<UTC timestamp>-<desk>-<slug>.md`, with exactly one entry. Never edit this file: a branch never rotates or stitches, so no two sessions share a file.

**Stitching and rotation** happen on main after a merge, by script: `python -u scripts/wiki_log_rotate.py` first stitches the `log.d/` fragments in here, newest first, then keeps the newest 10 entries once there are more than 15, moves the rest verbatim into [[log-archive]] and sets the digest's count; whoever runs it folds their gist into the digest.

## Digest (2026-04-09 → 2026-10-16, 345 archived entries)

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
- **The full menu, Pour A, the audit against earlier nets, the A/A** (2026-10-16, 6 archived entries) — the ten declarations at the arena ([[book-assumptions]], #155), the Demon's repository extraction ([[the-demon]]), the auction read by the known method (#117), [[lem-pour-a-2026-10-16]] registered, the audit read against Zeb's earlier nets ([[lem-field-net-audit-2026-10-15]] § Exploratory), seed-varied seats and #117 re-registered (#190).
- **The paper, belief in the inputs, the hand law** (2026-10-16, 6 archived entries) — the living draft and its Gaps (`lem/paper/README.md`), G1 registered and run ([[lem-belief-probe-2026-10-16]]), G7 registered ([[lem-fidelity-by-concept-2026-10-16]]), [[silas]]'s hand law (#134), seed-scatter caveats (#190).

---

## [2026-10-17 | lem | Formation (#194): units keep their concept family as they form; an untrained net is the honest floor]

[[lem-formation-2026-10-17]]: six runs (16×8 and 128×64, three seeds), 15 in-run checkpoints each.
0.82 of the wide net's clean units end in the family they first became clean in (null 0.28).
Position and trick units form first, and count and mastery later. Untrained nets already score
0.15–0.28 clean, a floor the shuffled-label chance (0.000) misses. Late-in-the-trick units form
coarse first. Two predictions held, one held pooled, one was wrong, one was ill-posed. Evidence
`lem/formation-2026-10-17/` (9c16180e).

## [2026-10-17 | book | Ch. 9 on field-v3: no trump turns the book's way, doubles trump still costs our planner (#189)]

- [[ch9-at-the-table-2026-10-17]], re-run on Lem's field-v3: 120 of 120 batches. Where the menu changes the declaration it is level overall, −0.010 [−0.028, +0.008] marks/hand. No trump reads +0.024 [+0.002, +0.045], the book's sign. Doubles trump is still −0.067 [−0.100, −0.034], and −0.072 with one player for both declarations, where field-v3 is weakest. The provisional cost was mostly the random field's. A10 is updated, and #201 is the doubles follow-up. Data: HF `book/ch9-field-v3/` (55dd9418, SHA verified).

## [2026-10-17 | book | Ch. 9 re-run on Lem's field-v3 net, pre-registered (#189)]

- [[ch9-at-the-table-2026-10-17]]: two arms on the same 4,800 changed four-double deals. v3dt plays doubles and no trump on field-v3 and pips on the deployed net; v3all plays every declaration on field-v3, so both declarations have one player. `MenuWalt` takes a `twin`, and `WaltPlay.plays` asks its net (field-v3 covers 7 and 9). The net is pinned by SHA-256. 120 batches.

## [2026-10-17 | book | Ch. 9 at the table, provisional: our random-field player cannot carry a doubles hand (#189)]

- Results on [[ch9-at-the-table-2026-10-17]]: 288 of 288 batches. The menu costs −0.045 [−0.065, −0.025] marks/hand on the 4,234 deals where it changes the declaration: doubles trump −0.106 [−0.142, −0.070], no trump −0.009 [−0.033, +0.015]. Every seat plays doubles and no trump on the random field, so this is the scope of our player, not a reading of the book's declaration. The A10 row is updated. Next is the re-run on Lem's field-v3 net (decls 7 and 9). Data: HF `book/ch9-provisional/` (97769d4e, SHA verified).

## [2026-10-17 | lem | Pour A published; the first field-v3 net plays all ten declarations as cleanly as the pip net]

[[lem-pour-a-2026-10-16]]: 250,000 deals, 4,349,960 positions, on `jasonyandell/texas-42-lem-pour-a`
(CC0); coverage predictions held (book doubles/no trump 0.56%, forced 12.49%).
[[lem-field-v3-audit-2026-10-16]] (G4): validation regret 0.0087 (pip 0.0082, doubles/no trump
0.0120), clean share 0.628 (halves 0.633/0.625) vs the shipped net's 0.602; concept list v2 renames
clean units rather than adding them; the declaration itself is barely named. Evidence
`lem/field-v3-audit-2026-10-16/` (3936f450).

## [2026-10-17 | lem | The squeeze: narrow field nets are cleaner, not mixed (G2 width pilot)]

Jason's pilot idea (undersized nets, look for coarse features), from Cammarata's "Already Clean"
squeeze. Seven nets retrained from scratch on the 250k pour; clean share 0.610 at 192 neurons, 0.833
at 24 (seeds 0.65–0.83); broadening within families, chiefly "late in the trick"; 24 neurons close
0.684 of the random-to-teacher gap vs 0.804. Predictions graded, the main one backwards.
[[lem-squeeze-2026-10-16]]; evidence `lem/squeeze-2026-10-16/` (cbb42932).

## [2026-10-17 | book | The book's Ch. 9 at the table, registered (#189), provisional (random field for 7–9)]

- [[ch9-at-the-table-2026-10-17]]: the chair's full menu (`MenuBidder`) against pip trumps only, on deals where it holds four doubles or more, every seat `MenuWalt`; primary a − b in marks per hand, the book predicts > 0; 288 batches, 23,040 deals, MDE about 0.026 on the changed deals. `scenario/ch9.py`.
- The arena's ledger gains a deal filter (`keep`, default unchanged, fixtures byte-identical) and writes all nine declarations in `offered` at a menu table; Silas replays a menu ledger's doubles and no-trump rows (`silas/tests/data/ledger-menu.json`).

## [2026-10-17 | book | A reporting rule: calibrate before you chase an A/A miss (#191)]

- [[book-authority]] § How to report a book result, rule 6: replicate an A/A that misses before explaining it, because the miss that starts a search is chosen for having missed and is no evidence for what the search finds. #117 → #190 → #191 is the worked case. The Mayor asked for this rule.

## [2026-10-17 | book | The estimator is calibrated: both A/A misses were chance (#191)]

- [[estimator-calibration-2026-10-16]], part 1: 960 of 960 batches. R_F = 1.06, 2 of 20 misses, δ(1,0) = −0.004 [−0.040, +0.031] marks/game, σ_seed = 0. The Book arena's intervals cover as they say, so #117's and #190's A/A misses were chance. With part 2 (the tournament, calibrated) and part 3 (Lem, 1.17× narrow from top-set selection), the ±0.15 / ±4-point allowance is withdrawn for the Book arena and the tournament.
- The caveats on #151 and #172 are replaced; A4 is updated; the #117 and #190 pages carry the resolution. *Exploratory:* the auction read pooled over both studies is +0.14 [+0.06, +0.22] marks/game. Data: HF `book/calibration/` (b1286005, SHA verified).

## [2026-10-16 | book | The tournament's pair bootstrap is calibrated: an A/A covers at n = 64 (#191 part 2)]

- [[estimator-calibration-2026-10-16]] Part 2: Zeb 0.4.1 contested against itself is an exact mirror (its host always sends seed 0), so the A/A is seed 1 against the ladder's seed 0 through a seed shim. Twelve replicates of 64 duplicate pairs on disjoint faceoff seeds, 1,536 games: R_F 0.85 in share and 1.02 in marks (calibrated range [0.59, 1.41]), 0 of 24 intervals missed the pooled mean, δ(1,0) −0.05 [−0.21, +0.12] marks/game, per-pair excess kurtosis −0.15, 25.4% of pairs at |margin| ≥ 6. The ±0.15 / ±4 allowance does not apply to the tournament's intervals for equal players; [[rematch-zeb-041-vs-demon-2026-10-15]] notes it. `tournament/calibration.py`; data HF `book/calibration-tournament/` (1ee36c0d, SHA verified).

## [2026-10-16 | lem | Split-half A/A of the audit's intervals: row Wilson 1.17× too narrow, the clean share replicates (#191)]

Inez (#191) asked how Lem's intervals are made and whether an A/A exists; the Mayor ruled the audit's
0.597 [0.581, 0.608] provisional because its Wilson intervals are over rows, not deals. Registered and
ran the A/A the same afternoon ([[lem-field-net-audit-2026-10-15]] § Split-half A/A): deals halved,
the audit rerun per half, 372 live poles. The row Wilson SEs are about **1.17× too narrow**
(|z| > 1.96 on 0.093 of non-degenerate poles vs 0.05), not the several-fold the desk predicted; a
deal-clustered bootstrap barely helps (1.12×) because a top set of 2,174 rows spans 1,730 deals.
The noise the row interval misses is the top set's own selection. The clean share replicates at
0.594 / 0.599 and is no longer provisional. The belief probes' R² now carry deal-clustered bands
(±0.004, ±0.007, ±0.003) and agree in both split directions ([[lem-belief-probe-2026-10-16]]).
Code `lem/probe/aa.py`; evidence `lem/aa-2026-10-16/` in the research repo (3eafdc57).

## [2026-10-16 | book | Calibrating the estimator, pre-registered: forty A/A replicates (#191)]

- [[estimator-calibration-2026-10-16]]: twenty replicates of seed 1 against seed 0 on disjoint deals (does the house interval cover the fixed δ(1,0)?) and twenty of fresh seed pairs (the seed-to-seed spread), 288 game pairs each; 960 batches. Deployed Zeb's choices proved independent of batch composition first (40 states, 0 differences). `scenario/calibration.py`.

## [2026-10-16 | book | The seed-varied re-run: the A/A missed again, the other way, and seed scatter is not why (#190 → #191)]

- Results on [[seed-varied-2026-10-16]]: 352 of 352 batches, none dropped. The A/A read Za −0.14 [−0.25, −0.03] marks/game, and PZa won 53.9% of games, so K = +0.12 [+0.00, +0.23] is reported, not claimed. Post-hoc checks found no per-seed effect (per-position spread below noise) and no batch correlation, so planner-seed scatter is not the mechanism. #191 calibrates the estimator directly.
- The caveats on [[rematch-zeb-041-vs-demon-2026-10-15]], [[fallible-field-2026-10-15]] and [[count-lapse-2026-10-15]] are restated as estimator caveats: an extra ±0.15 marks/game and ±4 points of share on single-arm margins. A4 is updated. Data: HF `book/seed-varied/` (c62bf021, SHA verified).

## [2026-10-16 | lem | G7 result: the field net strays on lead and near the contract, holds where the trick is decided]

- [[lem-fidelity-by-concept-2026-10-16]] complete: mean regret 0.0084 [0.0083, 0.0086] (434,535 rows), top decile carries 0.825; highest with masters-2plus, lead, to-make-le-10, is-bidder, holds-five; lowest where the trick is decided; regret 0.0008 at teacher ties, 0.0214 at gaps > 0.1. Prediction 1 held, 2 wrong (lead is high, not low), 3 half right. Paper § 4b and G7.
