# Log

**This is a changelog, not a chronicle.** If an entry needs a paragraph, the paragraph belongs on a page — write the page and link it. An entry is the header line plus at most ~5 lines of pointers (touched pages, added, retired, one flag). The log carries no claim that isn't reachable through a link.

Entry prefix convention: `## [YYYY-MM-DD | shortsha | subject]` so `grep "^## \[" log.md log-archive.md` gives the full timeline.

**Adding an entry:** insert it directly under the `---` below the digest, newest first. Nothing else: a branch never rotates. Parallel additions merge with git's union driver (`.gitattributes`), so branches don't conflict here.

**Rotation** happens on main after a merge, by script: `python -u scripts/wiki_log_rotate.py` keeps the newest 10 entries once there are more than 15, moves the rest verbatim into [[log-archive]] and sets the digest's count; whoever runs it folds their gist into the digest.

## Digest (2026-04-09 → 2026-10-15, 293 archived entries)

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

---

## [2026-10-15 | mayor | Clean neurons for 42: the "already clean" thesis, written up as an experiment]

- [[clean-neurons-42]]: Cammarata/Goodfire's "Neural Networks Are Already Clean" (linked, summarized in our words, not copied; the essay states no license), why 42's concepts generalize recursively (mastery, count, voids), and a cheap first experiment: audit all 384 poles of Zeb's shipped field net against Silas predicates and the book, then watch them form. [#175](https://github.com/jasonyandell/mk5-main/issues/175)'s GPT is the bigger second step.

## [2026-10-15 | silas/cross-game-slice3 | Silas plays a whole hand and compares two policies over deals (#130 slice 3)]

- `silas/src/field.rs`: a field of four Silas players, `play_out` (the auction, the declaration, all 28 plays, each seat at its own frame), `Deals::{Listed, Uniform}`, the paired `Trial::compare` with flips, the chance arm and an empirical-Bernstein interval (exact given the field over a list; never `exact`). Choices in `silas/docs/cross-game.md`, Slice 3 as built; Fiber and Hands deferred. [[silas]]

## [2026-10-15 | book/count-lapse | The field that loses the count pre-registered (#172)]

- [[count-lapse-2026-10-15]]: `arena/count_lapse.py`, a seat's view with a played and a live tile exchanged (a legal game, sticky for the hand); not a fiber, so the shared record is untouched; primary ΔG(0.2), 384 batches. [[book-assumptions]] A3 scope line.

## [2026-10-15 | archive | the s100001 field net in layout field-v2, published with its hash (#136)]
- `field_make_d5000_v2` converted once (13 zero bid rows); `31ecd76d` + the converted net and `2577f404` + the original
  both reproduce the published s100001 chunk bit for bit (512 deals, 8,844 positions). The manifest's `31ecd76d` is a
  save-time stamp; the run itself was `2577f404`. `margin_d50k_h128x64` loads as is.
- Touched: [[book-study-2026-10-12]], [[lumps-voi-2026-10-13]], [[metasplat-2026-10-13]]; the pour's card on HF.

## [2026-10-15 | rematch | Zeb 0.4.1 vs the overcalling Demon: pre-registered before any game]

**Added:** [[rematch-zeb-041-vs-demon-2026-10-15]] (256 duplicate pairs per arm, 0.4.0 match-aware as the equal-deals reference). **Added:** participants `zeb-0.4.1-contested`, `zeb-0.4.0-match` (kind `zeb-native`, roster only). **Result:** 0.4.1 wins 57.03% [53.12, 60.94] vs the Demon, 0.4.0 32.62% [28.91, 36.13] on the same seeds; +24.4 points [+19.5, +29.3]; no drops. Evidence on HF, faceoffs in the store.

## [2026-10-15 | book/fallible | The fallible field (#151): a field that forgets voids pays, and the book player and Zeb collect it equally]

- [[fallible-field-2026-10-15]] complete: ΔG(0.4) = −0.03 [−0.14, +0.08] marks/game (MDE 0.16), p* beyond the grid; the field's cost of forgetting 0.32 [0.23, 0.41] at p = 0.4.
- [[book-assumptions]] A3 row and reading. The book's ploys lean on losing the count, filed as #172.
- Data: HF `book/fallible/`.

## [2026-10-15 | book/fallible | The fallible field pre-registered (#151): the book's play against opponents who lose track]

- [[fallible-field-2026-10-15]]: deployed Zeb forgetting a seat's voids with probability p per decision (`void_miss`), p ∈ {0, 0.1, 0.2, 0.4}; primary ΔG(0.4) and p*; 384 batches; `scenario/fallible.py`.

## [2026-10-15 | book/overbidders | The book's own overbidders: its bidding loses less ground against them (#163)]

- [[overbidders-2026-10-14]] complete: X_S = +0.15 [+0.02, +0.28], X_T = +0.23 [+0.03, +0.41] marks/game, the book's prediction; `risk-budget` still trails the control in every field; the ten-count 35 sets itself (+1.30 [1.13, 1.45]).
- [[book-assumptions]] A6 row and reading; [[deviant-field-2026-10-14]] notes that its k = 0 "level to 250" did not replicate on new seeds (byte-identical re-run, so the draw).
- Data: HF `book/overbidders/`.

## [2026-10-15 | demon/confirmation-2 | Demon beats both pinned native Zeb bidders within the runtime bar]

**Completed:** [[demon-overcalling-2026-10-15]]: all 2,048 games audited and public; both predeclared strength/runtime criteria met. **Added:** [[demon-zeb-attempt-2]] and the exact frozen participant to the roster. **Updated:** [[the-demon]] [[tournament]] [[scenario-player]].

## [2026-10-15 | 2e233d29 | Demon overcalling development complete; second confirmation frozen]

**Updated:** [[demon-overcalling-2026-10-15]]: all 192 development games audited and public; fixed rule selects Over 0.50. Exact candidate and 2,048-game fresh confirmation protocol published before play. No promotion; #165 remains open.

## [2026-10-15 | demon/overcalls | Contest opponent-held point bids without more search]

**Added:** [[demon-overcalling-2026-10-15]] and [[2e233d29]]: explicit opponent cutoff, boundary and whole-game checks, three fixed development arms and fresh attempt-2 protocol. **Updated:** [[the-demon]]; attempt 1 remains complete and unpromoted.
