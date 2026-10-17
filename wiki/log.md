# Log

**This is a changelog, not a chronicle.** If an entry needs a paragraph, the paragraph belongs on a page — write the page and link it. An entry is the header line plus at most ~5 lines of pointers (touched pages, added, retired, one flag). The log carries no claim that isn't reachable through a link.

Entry prefix convention: `## [YYYY-MM-DD | shortsha | subject]` so `grep "^## \[" log.md log-archive.md` gives the full timeline.

**Adding an entry:** add it as a new file, `wiki/log.d/<UTC timestamp>-<desk>-<slug>.md`, with exactly one entry. Never edit this file: a branch never rotates or stitches, so no two sessions share a file.

**Stitching and rotation** happen on main after a merge, by script: `python -u scripts/wiki_log_rotate.py` first stitches the `log.d/` fragments in here, newest first, then keeps the newest 10 entries once there are more than 15, moves the rest verbatim into [[log-archive]] and sets the digest's count; whoever runs it folds their gist into the digest.

## Digest (2026-04-09 → 2026-10-16, 339 archived entries)

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

---

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

## [2026-10-16 | lem | G7 pre-registered: where the field net's regret under the teacher's plan sits]

- [[lem-fidelity-by-concept-2026-10-16]]: regret by the audit's 48 concepts, legal-move count and the teacher's best-minus-second gap, deal-clustered bootstrap; three predictions (regret concentrated in the top decile; highest with can-take-it and count-on-trick; rising with legal moves). `lem/probe/fidelity.py`.

## [2026-10-16 | lem | G1 result: the teacher's belief is in the field net's inputs, not its hidden layers]

- [[lem-belief-probe-2026-10-16]] complete: R² on the void-moved posterior entries, raw inputs 0.791, first hidden layer 0.241, second 0.035, shuffled ≈ 0 (deal-disjoint 20k/20k); the field net plays without a readable belief over who holds what. Predictions: inputs-state-most just under (0.79 vs ≥ 0.8), a1-near-x wrong, a2-below-a1 right, no-layer-beats-x right.
- The paper's § 5 and G1 carry it; evidence on HF `lem/belief-probe-2026-10-16/`.

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
