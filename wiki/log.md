# Log

**This is a changelog, not a chronicle.** If an entry needs a paragraph, the paragraph belongs on a page — write the page and link it. An entry is the header line plus at most ~5 lines of pointers (touched pages, added, retired, one flag). The log carries no claim that isn't reachable through a link.

Entry prefix convention: `## [YYYY-MM-DD | shortsha | subject]` so `grep "^## \[" log.md log-archive.md` gives the full timeline.

**Adding an entry:** add it as a new file, `wiki/log.d/<UTC timestamp>-<desk>-<slug>.md`, with exactly one entry. Never edit this file: a branch never rotates or stitches, so no two sessions share a file.

**Stitching and rotation** happen on main after a merge, by script: `python -u scripts/wiki_log_rotate.py` first stitches the `log.d/` fragments in here, newest first, then keeps the newest 10 entries once there are more than 15, moves the rest verbatim into [[log-archive]] and sets the digest's count; whoever runs it folds their gist into the digest.

## Digest (2026-04-09 → 2026-10-16, 321 archived entries)

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

---

## [2026-10-16 | book | Silas cross-game loose ends: one unit check, samples that name their stream, net seats named by hash (#183)]

- `Trial::compare` now refuses a utility whose unit is not its table's scoring, through the same `field::check` the ledger uses. The rule found one test of its own that read a marks table as points.
- `Sample` carries its `Stream`. A ledger's deals are `Sample::arena`, labelled `sampled(n, seed, arena)`, and a trial refuses an arena sample. `Authority::drawn(sample)` joins `sampled(n, seed)`.
- `arena.ledger.play_name` names a play seat with its seed and each field net with its SHA-256. A deployed Zeb seat sits in the chair in `arena/test_ledger.py`. Choices in `silas/docs/cross-game.md` § Slice 4 as built, items 3, 5 and 6. [[silas]]

## [2026-10-16 | book | The auction read, pre-registered: does reading a bid by a known method pay by itself? (#117)]

- [[auction-read-2026-10-16]]: the #149 reader alone, under the field's true bidding method (K) and an unknown one (U), against plain deployed Zeb, with a re-seeded A/A (Za). Primary K; 576 pairs per marks arm and 192 to 250; 176 batches. `scenario/auction_read.py`.

## [2026-10-16 | Demon | Early-forty Demon confirmed against Zeb 0.4.1]

[[demon-ascent-2026-10-16]] completes 512 fresh duplicate pairs: 911/1,024 wins, 88.96% [86.72, 91.11], at 0.13584 [0.13355, 0.13818] times total seat CPU (97.5% intervals). The opponent-specific early-40 form passes both criteria, all 9,759 hands audit, and the exact frozen participant is registered and rostered with prior records unchanged. All raw evidence is public.

## [2026-10-16 | Demon | Early forty selected for independent confirmation]

[[demon-ascent-2026-10-16]] publishes all 576 round-6 games and audits. The fixed selection wins 59/64 at 0.139 times pinned Zeb 0.4.1 CPU; this is development. Its exact source/config is frozen for 512 fresh duplicate pairs, with a predeclared joint win/CPU criterion. No participant promotion yet.

## [2026-10-16 | Demon | Higher early opening bids]

[[demon-ascent-2026-10-16]] records the fifth fixed sweep: selected 33/64, no established win, all evidence public. The fenced bidder accepts explicit early openings 30–40 with default records preserved; a source audit motivates testing 40 against pinned Zeb 0.4.1. The margin-band replay remains a parked prototype. No participant promotion.

## [2026-10-16 | Demon | Early-opening sweep is cheaper but has no established win]

[[demon-ascent-2026-10-16]] records all 512 round-4 games and the unchanged source: selected bait875play400 wins 46.9% [34.4%, 59.4%] against native Zeb 0.4.1 at CPU ratio 0.407 [0.384, 0.429]. No promotion or confirmation starts. All raw evidence is public; fresh verification resumed after a resolver-rate limit without rerunning games or uploading again. Source-policy and settled-contract audits are published separately.

## [2026-10-16 | lem | The field-net audit (Lem stage 1): clean neurons, play that is not sparse]

- [[lem-field-net-audit-2026-10-15]] complete: 222 of 372 live poles clean, 0.597 [0.581, 0.608], shuffled chance 0.000; 122 by a Silas concept, 100 by a raw feature of which 43 are declaration detectors (a concept the list lacked); Haiku readers detect 0.758 of poles vs 0.124 under the shifted null; greedy curve knee at k = 165 of 192 (prediction 40–120 wrong); 12 dead layer-2 neurons. Evidence on HF `lem/field-net-audit-2026-10-15/`.
- [[lem-the-teacher]] and [[clean-neurons-42]] carry the result line; `lem/probe/ablate.py` scores candidates incrementally (same statistic, by test).

## [2026-10-16 | Demon | Exact bid support and the failed preemption sweep]

[[demon-ascent-2026-10-16]] records all 384 round-3 games: no tested jump bidder establishes a win; the minimum-raise reference remains selected. An exact support predicate matches all 1,934 saved forecasts at native CPU ratio 0.4754 [0.4652, 0.4853] against the already-pruned best forecast. The opt-in adapter and experimental early-opening bidder are implemented; no participant is promoted. Generated evidence is public, and the study retains every negative result.

## [2026-10-16 | sweep | Branch rescue: Book's two worktrees merged and gone]

- [[branch-rescue-2026-10]]: `book/count-lapse` and `silas/cross-game-slice4` merged by the Book desk; off the live list.

## [2026-10-16 | Demon | Exact bidding bounds and preemptive bidding capability]

[[demon-ascent-2026-10-16]] integrates the exact bounded forecast, with all 1,934 original forecast winners/counts preserved, and adds an opt-in highest-supported-point bidder using common-world monotone bisection. Existing participant records remain unchanged; no new roster entry or strength claim. Both completed development rounds and all raw evidence remain public. The next trial tests preemption against pinned native Zeb 0.4.1.

## [2026-10-16 | Demon | Exact declaration elimination]

[[demon-ascent-2026-10-16]] records the complete first cutoff sweep (no established win), the frozen second development round, and an isolated bidding prototype. Incumbent bounds preserve all 1,934 saved forecast winners/counts while using 0.5210 [0.5124, 0.5295] of full-forecast native CPU. Public source, replay inputs and outputs are linked; this is kernel performance, not whole-player strength or promotion.
