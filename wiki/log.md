# Log

**This is a changelog, not a chronicle.** If an entry needs a paragraph, the paragraph belongs on a page — write the page and link it. An entry is the header line plus at most ~5 lines of pointers (touched pages, added, retired, one flag). The log carries no claim that isn't reachable through a link.

Entry prefix convention: `## [YYYY-MM-DD | shortsha | subject]` so `grep "^## \[" log.md log-archive.md` gives the full timeline.

**Adding an entry:** add it as a new file, `wiki/log.d/<UTC timestamp>-<desk>-<slug>.md`, with exactly one entry. Never edit this file: a branch never rotates or stitches, so no two sessions share a file.

**Stitching and rotation** happen on main after a merge, by script: `python -u scripts/wiki_log_rotate.py` first stitches the `log.d/` fragments in here, newest first, then keeps the newest 10 entries once there are more than 15, moves the rest verbatim into [[log-archive]] and sets the digest's count; whoever runs it folds their gist into the digest.

## Digest (2026-04-09 → 2026-10-21, 504 archived entries)

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
- **Lem unit papers 7–8, the audit's baseline report, Gus's contingent teaching tools** (2026-10-20 → 10-21, 10 archived entries) — unit papers 7 and 8 (#235, #202) and the audit's baseline and on/off report (#236); the Mayor's background desks and session-gated Lem frame; the Gus bid-30 pilot in the tournament store (#216); the Book desk's doubles-trump surface count (#230 A); Gus's historical-choice wish and broader contingent teaching ([[gus-tiny-gpt]]).
---

## [2026-10-22 | claude | lem: Night-Probe-2 — posterior-statistic labels and the decision-only audit (#239)]

- Six posterior facts per observed play (unseen trumps and count, voids and exhausted suits per
  seat, unseen tiles outranking the play, unseen tiles still following) plus a likelihood bucket.
  The finest lawful label leaves 7 roots with a decision collision (hand-relative: 47) but sees 0
  of 128 fresh roots; no label routes better than `seat` without seeing fewer. The 11 surviving
  groups differ in nothing the family states: the joint placement of hidden tiles decides them.
  [[gus-posterior-labels-2026-10-22]]
- Decision-only (the subset construction) fails at exactly the decision-collision roots in every
  row. Every label's quotient refines fork, so 56 of 128 bounds fresh roots seen; the solved label,
  the next decision's answer (four symbols at a root), reaches it and covers 95 decision-only.
- `lem/quotient.py --labels`, default report byte-identical (14fde1ac…). Evidence on HF
  `lem/gus-posterior-labels-2026-10-22/`; the research card now maps the night's three label dirs.

## [2026-10-22 | claude | lem: Night-Probe — the decision class in Gus, and tile-free routing labels (#239, #240)]

- Gus does not linearly carry the decision class or the Q contrasts at the query token on fresh
  roots (three-way: 107 fitting rows for 13 classes, underpowered). Pretraining put two-way
  flatness there: parent layer-2 AUC 0.767, +0.157 [+0.048, +0.281] over untrained twins. The
  masked lessons wore it down (wide −0.085 [−0.156, −0.017]) and moved the net's own three-way
  choice instead (+0.129 [+0.016, +0.258]). [[gus-decision-class-probe-2026-10-21]]
- No label in an eight-fact tile-free family routes the decision quotient, and none sees more than
  7 of 128 fresh roots (fork: 56). Under the finest label, 47 roots keep a decision collision whose
  blocks agree in every fact: the identity of a hidden tile decides the next choice.
  [[gus-observation-labels-2026-10-21]]
- `lem/probe/decision.py`; `lem/quotient.py --labels`, default report byte-identical. Evidence on
  HF `lem/gus-decision-class-probe-2026-10-21/`, `lem/gus-observation-labels-2026-10-21/`.

## [2026-10-22 | claude | night-math: the whole ending in Silas; good play meets only the knots]

- Silas rebuilds the note's 560-deal no-trump ending on its own rules (`silas/src/family.rs`,
  `silas/tests/controller.rs`) and reproduces every number:
  - 14 thresholds and 31 classes;
  - the 767-state controller, root ladder 1, 18, 31, 31;
  - 302 states with 2-1 committed.
- It derives the gap (5p − 1)/(1 + 18p) that Lean had taken as given.
- New: the root optimum has five affine pieces, so five fixed plans suffice and are needed
  (`scenario/lean/Stab/Plans.lean`, two lemmas).
- Along good play only the four knots of the optimum move the best play. The path controller
  has 132 states; the other ten thresholds are counterfactual.
- The comparison dimension (`silas/src/comparison.rs`) is 2 at the note's local history and
  168 at the root.
- Page: [[comparison-algebra-2026-10-21]].

## [2026-10-22 | claude | night-math: the minimal comparison algebra in Lean, the decision quotient in Silas]

- Lean (`scenario/lean/Stab/Hexagon.lean`, `Residual.lean`, `Mixing.lean`): 22 theorems on the
  three standard axioms, no sorry. They cover:
  - the note's two numbers per decision and the hexagon;
  - the 13 sign patterns as the sequential classes;
  - the coarsest output-keeping right congruence, attained by the quotient machine;
  - refinement exact after n − 1 blocks;
  - convex mixing forcing equality at the 1/5 boundary.
- Two brief claims proved false. The root Q triple has 4 weak orders, not 13, and one block is
  not enough (a three-tile witness needs two). [[gus-decision-quotient-2026-10-21]]'s "every
  ladder stabilizes after one block" is corrected: shape needs two.
- Silas `quotient.rs` and `silas quotient --course FILE` (fed by `lem/quotient.py --export`)
  reproduce `gus-decision-quotient-v1/report.json` in every count: 8,257 raw, 477 shape and 144
  fork states, 0 differences.
- Gate call: these Lean files are additive proofs in `scenario/lean`, as the brief placed them,
  with no Walt code touched, so they merged ungated (precedent cb95168e). Silas changes avoid
  core pins, physics and frame. [[comparison-algebra-2026-10-21]].

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

## [2026-10-21 | claude | lem: unit paper 10 graded: 16×8 s0 layer-1 unit 1 saves the doubles (#235, #202)]

- L1 1 keeps the doubles back and spends singles and lesser trumps. Ablated on trump-led follows
  with the trump double legal, the double played rises +0.427 [0.417, 0.436], rank 1 of 16. On
  discards the low doubles rise +0.051 [0.047, 0.055].
- Registered between verbs A ("take with trump") and B ("keep the doubles back"). Prediction 1
  missed toward B, and that miss is the finding. The lead null the arithmetic implied before
  sealing was a dose: forced on at a follow dose, it holds doubles back on no-trump leads too
  (−0.154).
- 4 of 5 graded held:
  - rank 3 of the trump-in votes;
  - L-27 +0.020;
  - the trump-lead job before the kill (−0.193 at step 30,540);
  - the kill's reason, "let the trump double cover": where the gate is missing, ablating raises
    the double +0.242 [0.199, 0.285].
- v3 boundary κ 0.554 vs 0.447 for v2. Evidence on HF `lem/unit-hold-trump-2026-10-21/`.
  [[lem-unit-hold-trump-2026-10-21]], card synthesis updated. This was the last unit of the #202
  queue.

## [2026-10-21 | claude | lem: unit paper 9 graded: 16×8 s0 layer-2 unit 1 leads the top off double under fives and sixes (#235, #202)]

- Ablated on leads where I hold the top double (the 6-6, or the 5-5 under sixes), that lead falls
  0.633 → 0.197, rank 1 of 8. Its gate is "slot 14 holds no ten" (κ 0.786 against 0.437 for the
  best v2 concept), so it corrects the layer's slot-14 vote: L2 0, 3 and 4 each zero slot 14 when
  ablated.
- Graded 3 of 3 held:
  - with L2 3 jointly ablated, slot 14 falls 0.771 → 0.040;
  - on sixes follows with the 5-5 legal, B beat A: laying it on partner's trick −0.571 against
    following fives −0.279;
  - the follow job grew after the lead job (step 152,700 against 30,540, after a death and rebirth).
- One confirmatory missed: with both doubles in hand, the net leads the 4-4 first (0.447 against
  0.244).
- Evidence on HF `lem/unit-top-double-2026-10-21/`. [[lem-unit-top-double-2026-10-21]], card
  synthesis updated.

## [2026-10-21 | f811360 | Gus decision quotient of the contingent course]

**Touched pages:** [[experiments/gus-decision-quotient-2026-10-21|gus-decision-quotient-2026-10-21]] [[gus-tiny-gpt]] [[gus-historical-choice-wish]]
**Added:** exact decision quotient (coarsest right congruence of the menu/maximizing-set interface) over the 256 eligible maskbreadth roots: 8,257 raw states, fresh roots never repeat a raw state, 139 fork structures; `lem/quotient.py` with tests; a measurement on saved data, no training.
**Updated:** [[gus-tiny-gpt]] routes to it; [[gus-historical-choice-wish]] names it as the first measurement of the adopted formulation.

## [2026-10-21 | claude | lem: unit paper 9 registered and sealed: 16×8 s0 layer-2 unit 1, leading the top off double under fives and sixes (#235, #202)]

- Raw read: on 0.201, at 0.96–0.99 of leads under fives and sixes and about 0.05 elsewhere. Its gate
  is "slot 14 holds no ten" (L1 15, `x:count:14`). Its canonical push leads slot 20, the top off
  double (the 6-6, or the 5-5 under sixes), over slot 14 and trump: the correction of paper 4's L2 3.
- Graded: the joint ablation with L2 3 (composition), a causal A/B on sixes follows with the 5-5
  legal (cash it vs lay it on partner's trick), and a formation order (the follow job as a later
  branch).
- Reviewed against the essay's full text by a fresh Opus. It found three of four first predictions
  already settled by what was seen, and prediction 3 mixing feeding with taking. Both were fixed.
  [[lem-unit-top-double-2026-10-21]], `run_double`.

## [2026-10-21 | claude | lem: unit paper 8 graded: 16×8 s0 layer-2 unit 4 is the net's "feed partner's trick" unit under ones–fours (#235, #202)]

- Predictions 1, 2, 3 and 5 held. Under declarations 1–4, ablated where partner holds the trick,
  count played falls 0.669 → 0.086 (−0.583 [−0.589, −0.575]), rank 1 of 8. L2 2 does the same job
  under 5–6.
- 4 missed narrowly: partner's trump lowers the dial 0.265 of an opponent's trump-lead dip, against
  a bar of 0.25. 6 missed: verb B ("feed, don't take") lost to A. The dip holds where I cannot take
  the trick back (0.894).
- The #202 κ 0.00 is a 1.3% tail of a dial. The v3 word is `trick-holder-kind` × seat (R² 0.640).
- The card's synthesis now has papers 7 and 8. Evidence is on HF at
  `lem/unit-count-on-it-2026-10-21/`, in [[lem-unit-count-on-it-2026-10-21]].
