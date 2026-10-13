# Log

**This is a changelog, not a chronicle.** If an entry needs a paragraph, the paragraph belongs on a page — write the page and link it. An entry is the header line plus at most ~5 lines of pointers (touched pages, added, retired, one flag). The log carries no claim that isn't reachable through a link.

Entry prefix convention: `## [YYYY-MM-DD | shortsha | subject]` so `grep "^## \[" log.md log-archive.md` gives the full timeline.

**Adding an entry:** insert it directly under the `---` below the digest, newest first. Nothing else: a branch never rotates. Parallel additions merge with git's union driver (`.gitattributes`), so branches don't conflict here.

**Rotation** happens on main after a merge, by script: `python -u scripts/wiki_log_rotate.py` keeps the newest 10 entries once there are more than 15, moves the rest verbatim into [[log-archive]] and sets the digest's count; whoever runs it folds their gist into the digest.

## Digest (2026-04-09 → 2026-10-13, 218 archived entries)

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
- **Generated data to Hugging Face** (2026-10-12, 1 archived entry) — [[publish-data-to-huggingface]]: the 250k Zeb pour and the day's research artifacts public. Hub: [[scenario-player]].
- **The book face-off** (2026-10-12, 2 archived entries) — [[book-study-2026-10-12]] § The face-off: the same gates asked of random, jud, E[Q] n=10 and the field net; closed by a k1,n2048 re-plan, a counterexample the standard for siding against the book (#121).
- **The book corrected** (2026-10-12, 1 archived entry) — [[book-study-2026-10-12]] and the w42 pages: chance baselines everywhere; #122–#125 opened.
- **otis, the algebra, walt endgame, parlor** (2026-07-15 → 07-18, 6 entries) — [[otis]] [[otis-v0]] graded, the referees doctrine on [[count-fate-ledger]], [[belief-policy-value-algebra]] promoted, [[jud-v2-retrain-probe]], [[walt]] endgame solver (+3.00 over jud), [[parlor]] promoted.

---

## [2026-10-13 | fd78c782 | silas aggregate (#135): card, points and (<= a b), three-valued by intervals; the bid rules rewritten with it, every call and declaration pinned equal on all 1,184,040 hands (risk-budget 150 → 20 µs); the crisis keeps the double with the most tiles out]

**Touched pages:** [[silas]] § The aggregate. `silas/README.md` § Semantics, Short forms, The auction; `silas/GAPS.md` gap 5 closed for what is built; `silas/docs/cross-game.md` § 4, Slice 2 as built (amendment).

## [2026-10-13 | 9f40de95 | silas cross-game slice 2: the auction in the record (contract derived, legality held to the arena and the engine), bidding and declaring in policies; bid-only-enough is the arena's HeuristicBidder on every hand; risk-budget's exact opening histogram (54.7% pass)]

**Touched pages:** [[silas]] § The match and field-free comparison. `silas/README.md` § The auction; `silas/docs/cross-game.md` § 4, Slice 2 as built; `silas/GAPS.md` gaps 2 and 4 narrowed (#134 now needs only #128).

## [2026-10-13 | 8d42bf70 | silas short forms (inferred roles; optional out, pattern, fallback; and/or/not; `(play x)`) read to the identical core, pinned for every file; lint warnings and `silas coverage` for rules that never fire; book rules, prelude and examples rewritten (−453 lines), gates kept explicit]

**Touched pages:** [[silas]]

## [2026-10-13 | fae0495a | silas perf: the time was CPU defects, not a GPU gap — fiber count over reachable states, allocation-free clause evaluation (re-price 12 → 4.5 µs/call), debug test profile at opt-level 1 (killed at 300 s → 34 s); exact MPS world counting 450 ms against Silas's 2.2 ms, so no GPU path]

**Touched pages:** [[silas]]

## [2026-10-13 | scripts/check | one front door for every suite (`scripts/check`, `--fast`); the log merges without conflict and rotates only on main; the archive restored from a bad merge (210 entries)]

**Updated:** `wiki/AGENTS.md` § Log format and Lint (branches never rotate; `scripts/wiki_log_rotate.py` on main; LG02, LG03); `CLAUDE.md` § Quality gates; [[log-archive]] restored.

## [2026-10-13 | silas | cross-game slice 1: the match frame (points to 250, marks to seven, the settlement under both and both readings of a set 84), checked match-win values and the score-grid programme, and `(dominates ...)` exact over the hand's 43 outcomes; the last mark's 42-over-84 decided under both scorings]

[[silas]] § The match and field-free comparison (new); `silas/README.md` § The match; `silas/GAPS.md` gap 2 narrowed (cluster F closed, B's three dominance rows and A's contract half stated, G re-tagged); `silas/examples/book/cross-game/`.

## [2026-10-13 | silas | the repaired book, first two rules (#121): C12B's guard of the second trump as a query (194/285 at every hand of the shape, so no policy), C10's off permission withdrawn by the partner's-ruff probability (never over 1/2); counterexample deals both ways, double-dummy; C10 re-priced +0.16 → +0.48]

**Updated:** [[book-study-2026-10-12]] § The repaired book (new); `silas/examples/book/repaired/` (new), `silas/tests/book_repaired.rs`; `silas/README.md` § Examples. Outputs: `scratch/repaired/`.

## [2026-10-13 | silas | solved outcomes: `(lay-down)`, `(certain-make)`, `(certain-set)` exact or refused, `(dd-make)` the double-dummy bound above them; the book's lay-down rule stated, Hands 21 and 22 certain, Hand 26's "cannot be set" false for the bidder alone though double dummy makes in all 756,756 worlds]

[[silas]] § Solved outcomes (new); `silas/README.md` § Solved outcomes; `silas/GAPS.md` (gaps 8 and 9: the side's guarantee, outcomes other than the contract).

## [2026-10-13 | scenario/reason | the reason report's space grown (position facts, count at risk, the prelude's nouns, every two facts joined) with coverage and a threshold and chance level set before looking: a reason for 32 of 44 face-off prices, none for 12; C12B's recurring reason, the guard of the missing trump, is a condition the book's rule omits]

**Updated:** [[scenario-player]] § The reason report (the grown space, the threshold, results by rule) and the parts table; [[book-study-2026-10-12]] § Worked reasons (three of six clear the chance level); [[silas]] § Where it sits. Outputs: `scratch/reason-v2/`.

## [2026-10-13 | scratch/book-reprice-fix | the dead double and the trump set's feed fixed to the book's conditions and re-priced on the whole pour: neither prices against now; Zeb 54% vs 40% (0.23), no relation on 3, against none]

**Updated:** [[book-study-2026-10-12]] § The book's rules as the book means them (two price rows, two rule rows, the reading); [[w42]], [[silas]] (rule table, Where it sits), `index-experiments.md`, `CLAUDE.md`; `silas/lib/book.silas` (`can-take-it`, `may-yet-trump`).

## [2026-10-13 | scratch/book-reprice | the book's 14 rules as the book means them priced on the whole 250k pour: Zeb sides with 3, prices against 2 (dead double, trump set's feed: each fired where a condition the book states fails); only 3 of 21 gates were the book's rule]

**Updated:** [[book-study-2026-10-12]] § The book's rules as the book means them (fidelity, prices, worked reasons; summary verdicts corrected in place); [[w42]], [[silas]] § Where it sits, `index-experiments.md`, `CLAUDE.md`.

## [2026-10-13 | silas | the book's rules as the book means them: fourteen policies in `examples/book/`, the 21 gates moved to `examples/book/gates/`; every catalogued mis-encoding tested on the book's own positions, gate failing and rule passing]

[[silas]] § The book, rule by rule (rewritten: the rule table); `silas/GAPS.md` (gap 3 down to the donation lead, gap 5's "what sets" closed for one called tile, three self-contradictions recorded).
