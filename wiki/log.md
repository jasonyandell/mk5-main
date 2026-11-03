# Log

**This is a changelog, not a chronicle.** If an entry needs a paragraph, the paragraph belongs on a page — write the page and link it. An entry is the header line plus at most ~5 lines of pointers (touched pages, added, retired, one flag). The log carries no claim that isn't reachable through a link.

Entry prefix convention: `## [YYYY-MM-DD | shortsha | subject]` so `grep "^## \[" log.md log-archive.md` gives the full timeline.

**Adding an entry:** add it as a new file, `wiki/log.d/<UTC timestamp>-<desk>-<slug>.md`, with exactly one entry. Never edit this file: a branch never rotates or stitches, so no two sessions share a file.

**Stitching and rotation** happen on main after a merge, by script: `python -u scripts/wiki_log_rotate.py` first stitches the `log.d/` fragments in here, newest first, then keeps the newest 10 entries once there are more than 15, moves the rest verbatim into [[log-archive]] and sets the digest's count; whoever runs it folds their gist into the digest.

## Digest (2026-04-09 → 2026-10-22, 519 archived entries)

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
- **Lem unit papers 8–10, Gus's decision quotient, desks as agent files** (2026-10-21 → 10-22, 6 archived entries) — unit papers 8, 9 and 10 graded ([[lem-unit-count-on-it-2026-10-21]], [[lem-unit-top-double-2026-10-21]], [[lem-unit-hold-trump-2026-10-21]]; #235, #202), [[gus-decision-quotient-2026-10-21]], the Mayor's desks-as-agent-files and the ladder page ([[standing-desks]]).
- **Lem unit papers 7–8, the audit's baseline report, Gus's contingent teaching tools** (2026-10-20 → 10-21, 10 archived entries) — unit papers 7 and 8 (#235, #202) and the audit's baseline and on/off report (#236); the Mayor's background desks and session-gated Lem frame; the Gus bid-30 pilot in the tournament store (#216); the Book desk's doubles-trump surface count (#230 A); Gus's historical-choice wish and broader contingent teaching ([[gus-tiny-gpt]]).
- **Comparison algebra, Gus routing probes and book lines** (2026-10-22, 9 archived entries) — [[comparison-algebra-night]], [[comparison-algebra-2026-10-21]], [[gus-decision-class-probe-2026-10-21]], [[gus-observation-labels-2026-10-21]], [[gus-posterior-labels-2026-10-22]], [[gus-joint-labels-2026-10-22]], [[book-line-play-out-2026-10-22]], [[walt-endgame-probe-2026-10-22]].
---

## [2026-11-02 | claude | Walt bids: the honest plan's root splat as a live bidder, piloted]

**Touched pages:** [[walt-bidder-2026-11-02]], [[scenario-player]], [[arena]]
**Added:** `arena/walt_bid.py`, spec `waltbid[:<walt play options>][,c<cutoff>]`: Zeb's bidder (`ZebBidder`) holding Walt's own planner, so Walt's honest forecast (floor 16) is read as a bid curve the way Zeb's is. A clock-sized pilot against the heuristic bidder with Walt play on both sides wins 34/48 games, +1.25 marks/game [+0.35, +2.12]; Walt bids 25% of hands at a mean of 32.7 and makes 88.8% against a mean forecast of 0.840 at the call, the heuristic 55.4% at 30.4; 48 games, 3.04e13 instructions. Data: `walt-bidder/` in [texas-42-zeb-research-2026-11](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-11). Follow-ups filed as #254.
**Questions opened:** whether the forecast at the chosen call is honest at power (the max over seven declarations and thirteen targets); a Walt mode for the package so Plunge can bid live with Walt's play.

## [2026-11-02 | codex | Zeb weight-space audit]

**Added:** [[zeb-weight-space-2026-11-02]] — shipped weight identity, untrained bid inputs, one direct edit and a reproduced regression; candidate not promoted.
**Updated:** [[scenario-player]] routes to the audit; evidence archived on Hugging Face.

## [2026-10-22 | claude | exact-gpu: k = 2 for the seven trick-1 leads; the rate]

**Touched pages:** [[exact-gpu-2026-10-22]]
**Added:** k = 2 rungs for six leads (d12 refused at B2 = 3e12, 3.33e12 plays), every bound nested inside k = 1. Five tiles over the 1.2e8 leaf cap were split by the next seat's reply, 15 pieces each, with masses adding to 399,072,960 per tile. PREREG-GPU-RATE: 14/14 pieces byte-equal; Metal 10–11× the wall of 12 CPU threads, host instructions 1,027–2,347× fewer (GPU instructions not counted).
**Questions opened:** whether deal 12's k = 2 rung is worth a raised B2 (Exact-T1's ruling); Hugging Face publication of the k = 2 triples and SHA-256 receipts (Exact-T1).

## [2026-10-22 | claude | exact-1: S0 and S1 measured; T4 stopped partial on Jason's instruction]

**Touched pages:** [[exact-player-s0-2026-10-22]], [[exact-player-s1-2026-10-22]]
**S0:**
- G1 exact at all 11,336 decisions; G2 on all 4,000 hands; G6 identical.
- G7 passes, leaning optimistic. As run, it reads the uniform kernel's misfit to Walt's records,
  not the construction.
- Three own tiles, undecided: median 4.7×10³ choices and 4.5×10⁸ instructions; plan controller
  median 57 states.

**S1 (lean-low/lean-high mixture, every p):**
- P: 6,064 of 6,065 exact, with G1 and G6 holding. For 7 decisions in 8 at three tricks left, the
  act is the same at every p.
- Median wall 0.9–3.2× Walt's live rung (data only).
- T4 is partial, stopped on Jason's instruction: 41 of the 76 zeb positions attempted were solved
  exactly; 23 were refused on the read budget, 10 on the wall cap and 2 on memory. The champion
  pour was not reached.

**Fixed:**
- The drivers' dead throttle: `jobs -r` ran in a subshell.
- `exact_report`: `--walt`, pooled instructions, Walt's zero times.
- The Hub file count now follows pagination.

**Questions opened:** #253 (cpu-slot orphans a wrapped solver at the cap). S1's G6 on T4 was not run. Trick 4's memory per course choice (about 1.2 KB)
bounds the fibers that fit. The pure calibration gate (worlds drawn from the posterior) is still to
be built.

## [2026-10-22 | claude | exact-gpu: follow roots climbed on Metal; the depth-2 split; the leafless count; stopped]

**Touched pages:** [[exact-gpu-2026-10-22]]
**Added:** Nine follow roots from Exact-T1 were climbed from the rung its CPU reached. Two settled, d12 v3 at k = 3 (6-3) and d15 v2 at k = 4 (5-0, Γ 6.4‰); the rest remain unsettled at k = 3–4. Two new splits: by reply, and by the play after it (`Piece`). The viewer's node folds by max, with every piece holding all its worlds, and another seat's node sums; tested at both kinds of node. `fold` builds a root's rung from its triples lines. Fixed a hang: at k = 4 one command could fill the 2^25-slot leaf table, after which every insert probed every slot. The count now touches no table and stops partway through a tile past its budget, a command's worlds follow the leaves it claims, and an insert refuses after 4,096 probes. All pins hold. Deal 1's k = 2 bytes (33 GB) were deleted on Exact-T1's release; their digests stay in `k2-triples.txt`.
**Questions opened:** the open climbs at the stop, queued but never run: d5 v2 k = 5 (depth-2 split), and d0 v2, d1 v2, d5 v3 and d39 v3 at k = 4; d39 v1 never started. Hugging Face publication rests with Exact-T1. Issue #248 (the silas-metal suite in `scripts/check`) waits on the Mayor.

## [2026-10-22 | claude | exact-t1: the trick-1 ladder, how far the bidder's lead is from exact]

**Touched pages:** [[trick1-ladder-2026-10-22]] (new), [[silas]], [[strategy-fusion-fence]] (inventory row for `silas/src/player/trick1.rs`), index-experiments.
**Added:** `silas/src/player/trick1.rs` (counted belief, the focal-horizon sandwich with a lawful lower bound and a refuting upper bound, Γ certified, budget in plays; gathers as bytes that merge by shard and join by root tile), `silas/tests/trick1.rs`, `silas/examples/trick1_roots.rs`. The seven leads: Γ falls every rung; at k = 2, 75–295‰ on five, deal 1 proved, deal 12 past B2. Strictly settled 0/7, proved 1/7. Jason's settling half not met, his modelling half met on 6/7 (as a guarantee gain; the law's tile survives on 7/7). The orchestrator's met on 6/7. Exact-T1's "no lead settles" fails by deal 1. The 21 follow roots, reported apart and partial at Jason's stop: 8 forced, 3 proved (deal 2), 2 settled with a choice (k = 3, 4), 1 ended past 1e12, 6 still open, 1 not measured; the law's tile is refuted at three. Data on HF at texas-42-zeb-research-2026-10 exact/trick1-2026-10-22/.
**Questions opened:** #250 (reduce as you go: the leaf map is the wall past k = 2), #251 (a family of types at trick 1; the record refutes one), #252 (how much of Γ is the clairvoyant upper side), #249 (cpu-slot's timeout misses a wrapped runner).

## [2026-10-22 | mayor | cpu-slot caps the whole job, not just its wrapper (#249)]

`scripts/cpu-slot` now starts the job in its own process group. The timeout ends the whole
group, and an interrupt or hangup to cpu-slot is passed on to it. Before this, a wrapper such as
`/usr/bin/time -l` exited on the cap and left the runner going outside any slot. One trick-1
rung ran 18 minutes at 21 GB after "timed out". Two new tests fail on the old script and pass on
the new one. The cost line still counts only the direct child, so a wrapper goes outside
cpu-slot, not inside it (documented in the script). The night's pause file for the ladder
(`scratch/ladder/PAUSED`, 988ded87) is in the same playbook spirit: a stop must outlive a respawn.

## [2026-10-22 | claude | exact-gpu: the field epoch, M3 and the trick-1 ladder on Metal]

**Touched pages:** [[exact-gpu-2026-10-22]], [[silas]], [[strategy-fusion-fence]]
**Added:** `silas-metal/`, a standalone crate. It holds the Metal field epoch (= CPU mirror = Silas reference), the M3 h8 gate (all four roots, H and C equal texas-42's AUDIT) and the trick-1 ladder gather, whose bytes are Exact-T1's `Gathered` (k = 0, 1 and 2 byte parity). Per-root-tile gathers, SHA-256 receipts, and refusals for leaves, free memory and B2.
**Questions opened:** #248 ([gated] a silas-metal suite in scripts/check); the rate PREREG (`PREREG-GPU-RATE.md` in the desk folder).

## [2026-10-22 | claude | exact-1: the exact player lands in Silas (S0 pre-registered)]

**Touched pages:** [[exact-player-s0-2026-10-22]]
**Added:** `silas/src/player/` (counted posterior over world × type, information-state solve for every p via `family.rs`, `play_hand`, the second construction `recount`), `silas exact`, `silas/tests/player.rs` (G1 second construction, G2 Bayes+Bellman, G4 the 767 rebuilt, G6). Python checks banned on the track (Jason, 07:55).
**Questions opened:** S0's measured batch (PREREG-S0 in the desk folder).

## [2026-10-22 | claude | book: #245 — an off's guard is the double or the top tile left in its suit]

- **Re-scoped from the policy log.** Of #245's four suspected faults, one was a real encoding gap: Hand 23's 6-5, which the book saves for next-to-last as the 6-1's double (W42 p. 77). Two were book silences (the bidder's duck, the low-trump first lead), and one was retracted: the "blind throws" are the book's own pounce.
- **Repair.** `stands-for-the-double` replaces the literal double in `bare-off` and in the two rules that play an off the bid-of-marks way. Four core pins moved on purpose. A new surface test checks that the double's own case is unchanged on every frame. For the bidder, `bare-off` changes on 13 random frames and none of the book's narrated ones.
- **Measured on step C's deals.** Hand 23's stack arm makes 0.575 [0.560, 0.591], up from 0.394: paired +0.181 [+0.168, +0.195]. The line − stack gap closes to −0.005. Hands 24 and 25 are unchanged.
- **Reproducing earlier numbers.** The thirteen studies that seat the book stack name the pre-#245 commit `64ecc6cf`. Re-runs follow on #245.
- Pages: [[book-line-play-out-2026-10-22]].
