# Log

**This is a changelog, not a chronicle.** If an entry needs a paragraph, the paragraph belongs on a page — write the page and link it. An entry is the header line plus at most ~5 lines of pointers (touched pages, added, retired, one flag). The log carries no claim that isn't reachable through a link.

Entry prefix convention: `## [YYYY-MM-DD | shortsha | subject]` so `grep "^## \[" log.md log-archive.md` gives the full timeline.

**Adding an entry:** add it as a new file, `wiki/log.d/<UTC timestamp>-<desk>-<slug>.md`, with exactly one entry. Never edit this file: a branch never rotates or stitches, so no two sessions share a file.

**Stitching and rotation** happen on main after a merge, by script: `python -u scripts/wiki_log_rotate.py` first stitches the `log.d/` fragments in here, newest first, then keeps the newest 10 entries once there are more than 15, moves the rest verbatim into [[log-archive]] and sets the digest's count; whoever runs it folds their gist into the digest.

## Digest (2026-04-09 → 2026-10-20, 494 archived entries)

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
- **G7's result, the estimator calibrated** (2026-10-16 → 10-17, 6 archived entries) — [[lem-fidelity-by-concept-2026-10-16]], [[estimator-calibration-2026-10-16]] (the Book arena, the tournament and Lem's split-half A/A; #190 → #191), [[lem-field-net-audit-2026-10-15]].
- **Ch. 9 at the table, the squeeze and the scope clause** (2026-10-17, 28 archived entries) — Rule 6 and [[ch9-at-the-table-2026-10-17]] (provisional, then field-v3: no trump the book's way, doubles still costs our planner); Lem's [[lem-squeeze-2026-10-16]], [[lem-formation-2026-10-17]], [[lem-width-tree-2026-10-17]], [[lem-dropout-order-2026-10-17]], [[lem-naming-2026-10-17]], [[lem-causal-2026-10-17]] and steering left open (#205); the scope clause and the book-interp skill on [[clean-neurons-42]]; the first tiny observer GPT ([[lem-observer-2026-10-17]], #203, #204).
- **Gus: training, depth and release** (2026-10-18, 41 archived entries) — [[gus]] named and published: fixed-field curriculum, frozen benchmark, full-hand outcomes ([[gus-fixed-field-curriculum-2026-10-18]], [[gus-frozen-benchmark-2026-10-18]], [[gus-full-hand-outcomes-2026-10-18]]), exposure budget ([[gus-exposure-budget-2026-10-18]]), the population and depth campaign to twelve layers ([[gus-population-iteration]]).
- **Gus: recentred on winning; tournament batching** (2026-10-19, 19 archived entries) — the learning charter, leave-one-out baseline, partner-history and public-score instruments, [[gus-blank-lead-2026-10-19]], [[gus-first-look-2026-10-19]], PPO/value comparison and outcome learning without teacher anchoring; Hub publication batched for the tournament.
- **Gus confirmation and routing; the July shelf** (2026-10-19 → 10-20, 6 archived entries) — [[gus-sampled-confirmation-2026-10-19]] (the sampled-policy gain did not replicate), [[gus-unanchored-outcomes-2026-10-19]], [[gus-first-look-2026-10-19]], [[gus-route-rescue-2026-10-20]], [[gus-tiny-gpt]] canonical delivery; the sweep's [[branch-rescue-2026-10]] (the Walt/Hoyt stack in attic/ tags).
- **Gus value numerics, #201's doubles cost, the first transformer audit** (2026-10-20, 6 archived entries) — [[gus-value-routing-2026-10-20]] (stopped before its primary), [[gus-value-numerics-2026-10-20]]; the Book desk's #201 (the doubles cost in thin top doubles) and `scenario.hf verify`; Lem's [[lem-gus-formation-2026-10-20]].
- **Lem's conditioned reads, steering, words and pour A** (2026-10-20, 6 archived entries) — [[lem-conditioned-audit-2026-10-20]] and [[lem-squeeze-conditioned-2026-10-20]] (both later superseded by [[lem-fixed-conditioning-2026-10-20]]), [[lem-matched-steering-2026-10-20]], [[lem-overbake-units-2026-10-20]], [[lem-words-need-data-2026-10-20]], [[lem-squeeze-pour-a-2026-10-20]].
- **Lem's raw re-read and unit papers; Gus's contingent teaching** (2026-10-20, 37 archived entries) — [[lem-family-statistic-2026-10-20]] (fails its calibration), [[lem-coarse-or-mixed-2026-10-20]] (void by its own rule), [[lem-unit-origin-2026-10-20]] (#231), the raw headline re-read (#233) and unit papers 1–4 (#235); the Book desk's #201 registration and confirmation; the Mayor's fresh-read-before-sealing rule and tidy; Gus's fixed-checkpoint transfer ([[gus-contingent-transfer-2026-10-20]]), [[gus-u65-formation-2026-10-20]], the contingent teacher, cost, breadth and practice runs ([[gus-contingent-teacher-2026-10-20]], [[gus-contingent-cost-2026-10-20]], [[gus-contingent-breadth-2026-10-20]], [[gus-contingent-practice-recovery-2026-10-20]]).
---

## [2026-10-22 | mayor | Desks become agent files; The 42 Ladder page]

- Four desks now (Mayor, Media, Book, Lem), each an agent file in `.claude/agents/` with its
  model and effort, run as short sessions that start from the latest handoff in
  `~/code/mk5-main/scratch/desks/<desk>/` and end by writing the next one. Archive and Sweep fold
  into the Mayor as subagents; the Mayor's watch is a Haiku session (`mayor-loop`). Playbook:
  [[standing-desks]].
- The tournament ladder's loop moved from a job's temp folder to `scripts/ladder-loop` (180 s
  chunk budget, so the last faceoff ends inside cpu-slot's 300 s cap), and its standings export to
  `scripts/ladder_export.py`. The page: [The 42 Ladder](https://claude.ai/artifact/TJYRL2ac7afwMookxGPdhp),
  18,247 faceoffs and 37,514 games as of 2026-10-22. Zeb 0.4.1 (contested) leads at 1812
  [1797, 1826]; the Demon with the c102 straight bidder is next at 1787 [1773, 1802], intervals
  overlapping. See [[tournament]].

## [2026-10-21 | pending | Gus continuation teaching tools]

**Added:** [[gus-continuation-teaching-tools]] — native solver, generation and pour inventory; Rust counted-belief extraction is the preferred bridge candidate, not an integrated G0/Zeb teacher.
**Updated:** [[gus-tiny-gpt]], [[gus-book-silas-tools]] — stopped loop and current capability route; Silas constructors remain on open PR #166.
**Questions opened:** whether richer four-tile lessons transfer; no new experiment or training launched.

## [2026-10-21 | Gus | Broader contingent teaching result]
<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

**Updated:** [[gus-contingent-maskbreadth-2026-10-20]] completed; fresh narrow contrast favorable, parent uncertainty keeps combined gate failed. [[gus-tiny-gpt]] and training skill route to verified evidence; no full-hand test or promotion.

## [2026-10-20 | f9d9fe1 | Gus historical choice teaching wish]

**Touched pages:** [[topics/gus-historical-choice-wish|gus-historical-choice-wish]] [[topics/gus-continuation-teaching-tools|gus-continuation-teaching-tools]]
**Added:** IDEATED history-conditioned choice-value and consequential-fork teaching wish; explicitly not built and no training restarted.
**Updated:** [[topics/gus-continuation-teaching-tools|gus-continuation-teaching-tools]] routes to the wish.

## [2026-10-20 | claude | lem: unit paper 7 graded: 16×8 s0 unit 8 is a dial on trump still out against me (#235, #202)]

- Predictions 1, 3 and 5 held. Prediction 2 held on sign and missed on rank: unit 2 is the larger
  brake on leading trump. Prediction 4 missed on followers: 4+ trumps in hand shut the unit
  0.12–0.25 of the time.
- Its graded variable is the trumps neither mine nor played: R² 0.525 on leads, and held + gone
  0.793. The lead subtracts. Forced on, the net leads trump 0.330 less and leads off doubles and
  count instead.
- The #202 κ 0.02 gap is vocabulary: list v2 has no count of trumps out. The candidate is
  `trumps-out-count`.
- Evidence is on HF at `lem/unit-pull-trump-2026-10-20/`, in [[lem-unit-pull-trump-2026-10-20]].

## [2026-10-20 | claude | lem: unit paper 7 registered: 16×8 s0 unit 8, the deepest list-v2 gap (#235, #202)]

- [[lem-unit-pull-trump-2026-10-20]] is registered and sealed after a fresh-context Opus review of
  the essay's full text. The review moved the label from trump length alone to two candidate verbs:
  a brake on leading trump, or "trump is still contested".
- The instrument is `run_lead` in `lem/probe/unit.py`, run with `python -m lem.probe unit … --lead`.
- Disclosed on the page: a smoke test exposed seed 0's formation, so formation is read off, not
  graded.

## [2026-10-20 | book | the book's surface count for doubles trump, encoded and unread (#230 A)]

`arena.hand_metrics.ch9_doubles_loss` counts what a doubles-trump hand can lose the way Ch. 9 does: trump losers, each open off's worst count (its own included), a point per trick. It matches the book's tallies on Hands 23–25 (checked privately; the tests use synthetic hands). No player reads it, so every ledger is unchanged. [[book-assumptions]] A10 now says the menu's ceilings omit these terms, and that the no-trump read at the book's sign did not replicate (#201).

## [2026-10-20 | archive | the Gus bid-30 pilot synced to the canonical tournament store (#216)]

- 31 missing records added in one commit (`9d02d5a3`): 17 participants (G5–G20) and 15 faceoffs in
  `faceoffs/marks7-bid30/20261018/`. All 37 checked anonymously against the manifest. The table
  built from the published bytes matches the pilot's exactly.
- Touched: [[gus-tournament-pilot-2026-10-18]].

## [2026-10-20 | claude | mayor: desks are background sessions after the reboot; the Lem frame is gated by session id]

- The machine rebooted at 18:17 CDT and took the tmux server, every desk, the ladder and the
  watcher with it. The desks came back as `claude --bg --resume <id>` sessions; the ladder and
  watcher were restarted by the Mayor.
- `.claude/hooks/after-compact.sh` gated the Lem essay reload on a tmux session name, which a
  background session doesn't have, so every desk would have paid for the ~4,300-word frame at
  each compaction. It now reloads the frame only for a session whose id is listed in
  `scratch/lem/sessions` (local, gitignored), or a tmux `desk-lem`; everyone else gets one line.
- [[standing-desks]] records the new launch form and the session list.

## [2026-10-20 | claude | lem: the audit reports what the baseline excluded and the on/off match (#236)]

- `lem/probe/audit.py` adds two report-only columns per pole, beside the label (never used to
  classify):
  - the best concept the play-count baseline excluded;
  - the concept, or its negation, that best agrees with the unit's on/off, against the agreement
    of a constant.
- On 16×8 s0 the labels are unchanged, and the columns recover the unit papers' hand findings:
  - L1+005 `last` (excluded 1.00; on/off 0.971 against chance 0.800);
  - L2+003 `lead` (0.96; 0.911 against 0.605);
  - L1±014 `¬partner-winning` (0.831);
  - L1±007 `¬trump-led` (0.925), where list v2 lacks the true boundary (#202).
- Reproduce with `python -u -m lem.probe audit SAMPLE LABELS OUT --net NET --baseline n_plays --shuffles 0`.
