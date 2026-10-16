# Log

**This is a changelog, not a chronicle.** If an entry needs a paragraph, the paragraph belongs on a page — write the page and link it. An entry is the header line plus at most ~5 lines of pointers (touched pages, added, retired, one flag). The log carries no claim that isn't reachable through a link.

Entry prefix convention: `## [YYYY-MM-DD | shortsha | subject]` so `grep "^## \[" log.md log-archive.md` gives the full timeline.

**Adding an entry:** add it as a new file, `wiki/log.d/<UTC timestamp>-<desk>-<slug>.md`, with exactly one entry. Never edit this file: a branch never rotates or stitches, so no two sessions share a file.

**Stitching and rotation** happen on main after a merge, by script: `python -u scripts/wiki_log_rotate.py` first stitches the `log.d/` fragments in here, newest first, then keeps the newest 10 entries once there are more than 15, moves the rest verbatim into [[log-archive]] and sets the digest's count; whoever runs it folds their gist into the digest.

## Digest (2026-04-09 → 2026-10-15, 302 archived entries)

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

## [2026-10-16 | book | Silas reads the arena's ledger; the differential check finds no disagreement (#130 slice 4)]

- `silas/src/ledger.rs`: `Ledger::from_json` (outcomes, never utilities; any other key refused), `Paired::from_ledger` replaying and settling every row under Silas's rules, `silas compare --ledger FILE`. `arena/ledger.py`: the fixed-state runner and ledger writer, `SilasSeat`, the chance arm, `overbidding_field(k)`. Choices in `silas/docs/cross-game.md`, Slice 4 as built. [[silas]]
- Round trip: two arena ledgers of a Silas-only field (marks with the straight auction; points from 200-180 with the forced shaker) are Silas's own `play_out` hand for hand, 192 hands, and read to the arena's paired mean exactly (`silas/tests/ledger.rs`, `arena/test_ledger.py`).

## [2026-10-16 | sweep | table42 closed out: issues #65 and #66 closed, worktree-table42 tombstoned]

- [[branch-rescue-2026-10]]: table42 landed (`f529d358`, fence ruling `9cebd296`); `attic/worktree-table42` tagged, branch and PR #68 closed.

## [2026-10-16 | sweep | table42 lands on main: host, act/wait, the #66 probes, entity and game-night playbook]

- [[table42]] (entity), [[table42-game-night]] (playbook), [[playing-from-the-inside]]; code in `tools/table42/`. Rescued from the `worktree-table42` branch (PR #68, July) by squash; [[jud]] and [[otis]] gain their table42 pointers. See [[branch-rescue-2026-10]].

## [2026-10-16 | lem | Lem's population is a league: mixed-partner self-play, partners from the population, Nash-averaged ratings]

- [[lem-the-teacher]]: Jason's direction (educational play, "the meanest nastiest models with lots of diverse competitors"); stage 2 is a league with Silas and LLM seats as the bootstrap, stage 3 trains from the league's games, stage 5 is scored on the payoff matrix with Nash averaging (#182, filed by the Mayor); the overcall's rock and paper as the first cycle. Epics #177, #178, #180 point at #182.

## [2026-10-16 | lead | cpu-slot admits by busy cores; runs report CPU and instructions beside wall (#173)]

**Touched pages:** [[light-every-core-2026-10-16]] [[standing-desks]] [[tournament]]
**Added:** [[light-every-core-2026-10-16]]: what each job kind keeps busy, who thinks on a clock (only Walt), and the pre-registered drain: 60 batches in 65.9 s instead of 196.1 s, 14.1 busy cores of 18 instead of 3.9, outputs byte-identical, instructions equal to 0.03%.
**Updated:** `scripts/cpu-slot` (load mode, `--threads`, cost line), `arena/cost.py`, book-desk batch metas and tournament records carry CPU and instructions, `noisy_timed` on clocked faceoffs, `scenario/exploiter.py` takes a node budget.

## [2026-10-16 | sweep | Branch rescue: verdicts on every branch]

- [[branch-rescue-2026-10]] § Other verdicts: recursive-generalization merged, table42 sent as `rescue/table42` (fence question), the July stack waits on Jason, live branches untouched.

## [2026-10-16 | sweep | Burl perf wiki pages rescued from the perf/batch and perf/cheap branches]

- Added [[burl-perf-phase1]] (a link that had been dead since the era-6 audit), [[mlx-cohort-bench-discipline]], [[harvest-cohort-abstraction]], all `status: superseded`. The branches' bench JSON stays in the `attic/perf/*` tags. See [[branch-rescue-2026-10]].
- Tombstones for `perf/batch` and `perf/cheap` in [[branch-rescue-2026-10]].

## [2026-10-16 | sweep | Branch rescue: nine branches tombstoned under attic/ tags]

- [[branch-rescue-2026-10]] § Tombstones: beads-sync, gascity, mk9, one-hand-per-page, worktree-walt, worktree-fix-jud, worktree-parlor, worktree-parlor-promote, worktree-wiki-overhaul; tags pushed first, branches deleted after.

## [2026-10-16 | sweep | Branch rescue: every unmerged branch inventoried and sorted]

- [[branch-rescue-2026-10]]: 23 remote + 19 local unmerged branches, one table; the July Walt/Hoyt stack (ten PRs) is waiting on Jason; verdicts and tombstones are added to the page as each branch is acted on.

## [2026-10-16 | lem | The old LEM's code removed; its tombstone and seven lessons on the entity page]

- [[lem]]: code gone at b4a3e5b7 under the no-legacy rule (the Mayor's go); § Tombstone (cd5683c0): dates and commits, every finding with its source and "no interval reported", why [[burl]] superseded it, seven sourced lessons each paired with what [[lem-the-teacher]] does differently, where the adapters, W&B runs and git history live.
- [[lem-the-teacher]] takes the one-pipeline rule for its scaling family; the loss-mask check is filed on #178.

## [2026-10-16 | lead | The log splits: one file per entry in wiki/log.d/]

**Touched pages:** [[standing-desks]]
**Updated:** `wiki/AGENTS.md` (Log format), `scripts/wiki_log_rotate.py` (stitches fragments, then rotates), `scripts/wiki_lint.py` (fragments count as entries; LG04), `scripts/tests/test_wiki_log_rotate.py`

## [2026-10-15 | 8df99e8b | Recursive generalization as an information-honest concept-learning proposal]

- Added [[recursive-generalization]]: original CC0 research note, ordinary training with checkpoint lineage, Silas transfer/counterexamples, causal tests and separate play-strength gates. No experiment run; constructor dependency is unmerged PR #166.
- Routed from [[suit-algebra]] and the topics catalog.

## [2026-10-15 | lem/stage1 | The Lem desk opens: the plan page, five epics, the field-net audit pre-registered]

- [[lem-the-teacher]]: the north star, the two claims (a small feature set? a model of others?), the five stages, the budget tiers, the fence, prior work. Epics #176–#180.
- [[lem-field-net-audit-2026-10-15]]: stage 1 pre-registered (concept list sha256 13e4d525…, statistics, Haiku readers, neuron-count curve, four predictions); the probe kit `lem/probe/` and its tests.
- [[lem]] links forward; [[clean-neurons-42]] is absorbed as stage 1.

## [2026-10-15 | mayor | Clean neurons for 42: the "already clean" thesis, written up as an experiment]

- [[clean-neurons-42]]: Cammarata/Goodfire's "Neural Networks Are Already Clean" (linked, summarized in our words, not copied; the essay states no license), why 42's concepts generalize recursively (mastery, count, voids), and a cheap first experiment: audit all 384 poles of Zeb's shipped field net against Silas predicates and the book, then watch them form. [#175](https://github.com/jasonyandell/mk5-main/issues/175)'s GPT is the bigger second step.

## [2026-10-15 | silas/cross-game-slice3 | Silas plays a whole hand and compares two policies over deals (#130 slice 3)]

- `silas/src/field.rs`: a field of four Silas players, `play_out` (the auction, the declaration, all 28 plays, each seat at its own frame), `Deals::{Listed, Uniform}`, the paired `Trial::compare` with flips, the chance arm and an empirical-Bernstein interval (exact given the field over a list; never `exact`). Choices in `silas/docs/cross-game.md`, Slice 3 as built; Fiber and Hands deferred. [[silas]]
