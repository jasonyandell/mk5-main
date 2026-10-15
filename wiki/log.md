# Log

**This is a changelog, not a chronicle.** If an entry needs a paragraph, the paragraph belongs on a page — write the page and link it. An entry is the header line plus at most ~5 lines of pointers (touched pages, added, retired, one flag). The log carries no claim that isn't reachable through a link.

Entry prefix convention: `## [YYYY-MM-DD | shortsha | subject]` so `grep "^## \[" log.md log-archive.md` gives the full timeline.

**Adding an entry:** insert it directly under the `---` below the digest, newest first. Nothing else: a branch never rotates. Parallel additions merge with git's union driver (`.gitattributes`), so branches don't conflict here.

**Rotation** happens on main after a merge, by script: `python -u scripts/wiki_log_rotate.py` keeps the newest 10 entries once there are more than 15, moves the rest verbatim into [[log-archive]] and sets the digest's count; whoever runs it folds their gist into the digest.

## Digest (2026-04-09 → 2026-10-14, 268 archived entries)

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
- **The tournament, the Book Authority, Zeb 0.3.0** (2026-10-14, 6 archived entries) — [[tournament]] (production Walt, Zeb, fenced E[Q] on a Bradley–Terry ladder), [[book-authority]] and [[book-assumptions]] (the charter and eleven assumptions), [[zeb-030-confirmation-2026-10-14]], [[regulars-table-2026-10-14]], the fly connectome as a player, [[legacy-42-bots]].
- **otis, the algebra, walt endgame, parlor** (2026-07-15 → 07-18, 6 entries) — [[otis]] [[otis-v0]] graded, the referees doctrine on [[count-fate-ledger]], [[belief-policy-value-algebra]] promoted, [[jud-v2-retrain-probe]], [[walt]] endgame solver (+3.00 over jud), [[parlor]] promoted.

---

## [2026-10-14 | archive | the two superseded Zeb pours deleted (every chunk matched by SHA-256 in the 250k pour), and the four unlicensed repos dedicated CC0 with cards]

**Updated:** [[huggingface-assets]] (17 datasets; licenses; mk5-fleet-payload belongs to [[otis-phase-r]], not the February fleet), [[scenario-player]] and [[publish-data-to-huggingface]] (the deleted repos named as history). Cards for mk5-run-evidence, mk5-fleet-payload and zeb-42-examples drafted by a Sonnet agent and checked against the Hub before upload.

## [2026-10-14 | 18ac096d | the Demon: exact search-table locality, compact 100- and 400-world tournament challengers]

**Added:** [[demon-search-tables-2026-10-14]], [[18ac096d]]. **Updated:** [[the-demon]], [[tournament]]; the roster retains the original and adds compact-table builds. Parity, resource measurements, the fixed duplicate-match pilot, and public data are on the experiment page.

## [2026-10-14 | reading-text | Silas: a reading written in a policy file (#162); the book's crisis as Hand 31 reads it]

**Updated:** [[silas]], `silas/README.md`, `silas/GAPS.md` 1. **Added:** `(reading ...)` in a policy and `PolicyProgram::compile_reading`; the `silas` command finds a reader beside its file; `examples/book/crisis-read.silas` (one new pin, none moved); `silas/tests/reading_text.rs`. The arena's half of #162 (gated) is next.

## [2026-10-14 | 98f5d16c | The named deviant field (#152): the book's bidding costs 0.92 [0.60, 1.22] marks/game against a field that bids its hand and 0.57 [0.22, 0.91] against one 10 points past it; level in the book's game]

**Updated:** [[deviant-field-2026-10-14]] (complete; 192 batches through `scripts/cpu-slot`, none dropped; C10 − C0 +0.35 [−0.02, +0.71], MDE 0.52; S-0114-b seen in the field's own arms), [[book-assumptions]] (A6 measured in one dimension). Data `book/deviant-field/` in texas-42-zeb-research-2026-10. **Filed:** #163, the book's kind of overbidder, powered.

## [2026-10-14 | guard-reads | Silas: a guard that reads (#128): a rule acts on a reading of another seat's plays]

**Updated:** [[silas]] (§ the record as evidence; next), `silas/README.md`, `silas/GAPS.md` 1. **Added:** `ReadingPredicate` and `ReadChair` in `silas/src/reading.rs`, `silas/tests/guard_reading.rs` (Hand 31's two readings as guards at every seating). No core pin moved. **Filed:** #162, a reading written in a `.silas` file and registered by the arena.

## [2026-10-14 | archive | recover everything to Hugging Face: the Archive desk's inventory and first batch (five repos, ~8 GB, screened, verified by download)]

## [2026-10-14 | c9a868cb | match-aware bidding: Zeb's bidder in the arena, the 84 priced by match equity wins 53.4% [52.5, 54.3] of games vs the hand-only bidder; 51.4% [50.6, 52.2] vs the same 84 priced score-blind; A/A exact; no drops]

**Added:** [[match-aware-bidding-2026-10-14]] (pre-registered at c9a868cb, model frozen at ee2c95b7), [[match-equity]]. **Updated:** [[scenario-player]] (§ Match-aware bidding), `index-experiments.md`, `index-topics.md`. Code: `arena/zeb_bid.py`, `arena/match_equity.py` (default off). Data: `zeb/match-aware-bidding/` in `jasonyandell/texas-42-zeb-research-2026-10` with SHA256SUMS. Follow-ups: #160.

## [2026-10-14 | bf16fdfe | The book at small n (#154): the near-tie break is free at 16, 64 and 160 worlds; the book's sets cost a 16-world plan 0.21 [−0.02, 0.44] marks/game and a 160-world plan 1.10 [0.93, 1.27]]

**Added:** [[book-small-n-2026-10-14]] (pre-registered at bf16fdfe, 280 batches through `scripts/cpu-slot`, none dropped; D16 −0.01 [−0.18, +0.17], MDE 0.25; A/A −0.03), `scenario/small_n.py`; data `book/small-n/` in texas-42-zeb-research-2026-10. **Updated:** [[book-assumptions]] (A7 measured to 16 worlds; #161 the reader with no plan). Pre-registered, queued: [[deviant-field-2026-10-14]] (#152, `scenario/deviant_field.py`, 98f5d16c).

## [2026-10-14 | 5807425c | The header audit (#153): the gates are history]

The 21 rows that stayed on a gate are listed in their rule's header or recorded as no rule's; `setter-leads` repaired at Hand 27 (a low tile that calls for a five and no ten goes last); `low-from-a-pair` learns the spare (Hand 29); the three "book against its rule" rows re-read under [[book-authority]]'s presumption, one reconciled. Ledger in [[book-assumptions]] (gate-only 21 → 0); rule table in [[silas]].

## [2026-10-14 | desks | the standing desks: Mayor, Book, Media, Archive; blob first, link second; the links registry in the feed's db]

**Added:** [[standing-desks]].

## [2026-10-14 | b88c5432 | the Demon summoned: a fenced CPU PIMC player (Walt's sampler, Silas's double-dummy solve per world, 100 worlds, margin tiebreak, heuristic bidder) seated as `demon-pimc-n100-b88c5432`; 32 ms a decision; smoke round vs E[Q] n=10 11–9, +0.35 [−0.98, +1.68] marks/game over 10 pairs (close cousins), vs Walt 4–2 and Zeb 0.3.0 1–5 (3 pairs each, a smoke test)]

**Added:** [[the-demon]], `tournament/demon/` (Rust), `tournament/demon_fenced_seat.py`, `tournament/tests/test_demon.py`, the `demon` suite in `scripts/check`. **Updated:** [[strategy-fusion-fence]] (inventory, sanctioned uses, guard), [[tournament]] (participants, seating, cloud worker), [[scenario-player]], `index-topics.md`, the fence guard test (no player may reach the Demon; only its crate turns on `clairvoyant`). Records: 16 faceoffs in `jasonyandell/texas-42-tournament`.

## [2026-10-14 | 70e2c569 | the license line: the science (engine, Walt, Silas, wiki, data) CC0, Zeb MIT; SPDX on every live code file, REUSE.toml for the rest, `reuse lint` in scripts/check; planner's mixed files CC0-1.0 AND MIT until #158]

**Added:** [[license-line]], `LICENSE` (rewritten), `REUSE.toml`, `LICENSES/`, `docs/powered-by-walt.svg`. **Updated:** [[scenario-player]] (Links), `index-decisions.md`, `CLAUDE.md` (hard rule), READMEs. Issue: #158.

## [2026-10-14 | cf89472b | book partners read each other by the book (#150, A2's receiving half): plays by the stack, calls by bid-only-enough; vs deployed Zeb 0.3.0 +0.10 [−0.06, +0.26] marks/game, A/A −0.05; null at this power, every sign positive]

**Added:** [[book-partners-read-2026-10-14]] (pre-registered at `cf89472b`). **Updated:** [[book-assumptions]] (A2 row, opinion, order), [[book-authority]], [[w42]] (frontier, routes, leaves-out), `index-experiments.md`. Code: `arena/book_play.py` (`read=`, shadow), `arena/book_read.py` (`seats`, `calls_weight`), `arena/walt_play.py` (`s<seed>`). Data: `book/partners-read/` in `jasonyandell/texas-42-zeb-research-2026-10` with SHA256SUMS.
