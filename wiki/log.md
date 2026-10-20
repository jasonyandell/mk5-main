# Log

**This is a changelog, not a chronicle.** If an entry needs a paragraph, the paragraph belongs on a page — write the page and link it. An entry is the header line plus at most ~5 lines of pointers (touched pages, added, retired, one flag). The log carries no claim that isn't reachable through a link.

Entry prefix convention: `## [YYYY-MM-DD | shortsha | subject]` so `grep "^## \[" log.md log-archive.md` gives the full timeline.

**Adding an entry:** add it as a new file, `wiki/log.d/<UTC timestamp>-<desk>-<slug>.md`, with exactly one entry. Never edit this file: a branch never rotates or stitches, so no two sessions share a file.

**Stitching and rotation** happen on main after a merge, by script: `python -u scripts/wiki_log_rotate.py` first stitches the `log.d/` fragments in here, newest first, then keeps the newest 10 entries once there are more than 15, moves the rest verbatim into [[log-archive]] and sets the digest's count; whoever runs it folds their gist into the digest.

## Digest (2026-04-09 → 2026-10-20, 445 archived entries)

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
---

## [2026-10-20 | lem | the sharper family statistic (#200) fails its calibration: permuted-family nulls set the bar too low]

[[lem-family-statistic-2026-10-20]]: corrected for base rates, family size and lumpiness, the
coarse-versus-mixed statistic scores 0.583 on untrained nets and 0.425 on shuffled labels, not 0.5.
Trained nets are no higher than their untrained twins (16×8: 0.518 against 0.601). Concepts of one
family co-occur, so a null that permutes family labels sets the bar too low. The squeeze's and the
width tree's family claims are now "not yet shown against untrained twins" (#229), and both pages
carry a correction. The clean shares and narrower-is-cleaner are unaffected. 0 of 4 predictions
held.

## [2026-10-20 | lem | the squeeze on pour A (#199): narrower is cleaner with every declaration, and doubles trump is the declaration a narrow net keeps]

[[lem-squeeze-pour-a-2026-10-20]]: 12 field-v3 nets on pour A, read conditioned.
- The shares are 0.504 at 192 neurons, 0.626 at 24 and 0.677 at 12, against floors of 0.04–0.14.
- `trump-is-doubles` survives to 12 neurons after every pip trump is gone, which is the opposite of
  my prediction.
- No trump never gets a clean unit.
- 3 of 4 predictions held.

## [2026-10-20 | lem | the words come with data (#228): from one start, 4k deals 0.019, 50k 0.173, 500k 0.237]

[[lem-words-need-data-2026-10-20]]: the 50,000-deal observer is the middle point between the
overbake run and Gus, from the same seed-42 initialization.
- At step 20,000 the conditioned share is 0.019 on 4k deals, 0.173 on 50k and 0.237 on 500k.
- The 50k run forms words from step 2,000, and they keep their family (0.978 against 0.233).
- 4 of 4 predictions held. More data adds words without mixing them.

## [2026-10-20 | lem | the words need data (#207): a 4k-deal observer plays at 0.53 agreement with no nameable units]

[[lem-overbake-units-2026-10-20]]: the overbake observer was trained on 3,988 deals.
- At its validation best it agrees with the recorded move on 0.53 of held-out decisions, but its
  units stay at the untrained floor, conditioned (0.034 against 0.014).
- Overtraining memorizes: agreement falls to 0.43, and the words never form.
- Gus on 500,000 deals formed them after about 10,000 steps.

This is rung 3, narrow data, not mixing. 1 of 5 predictions held. The 50k point is filed.

## [2026-10-20 | lem | matched-row steering (#205): no unit-specific signal; ablation stays the evidence of use]

[[lem-matched-steering-2026-10-20]]: steering cost was compared within each play count and against
five random units on the same rows. It shows no signal specific to a named unit:
- `+` poles: 0.41–0.46 below the random median (0.6 needed);
- `−` poles: 0.49–0.54 above it;
- pooled median −0.0001.

All three predictions were wrong. The test reads cost, not where the move goes, so a
direction-based steering test is the open question. It will be filed once it has a definition.

## [2026-10-20 | lem | the squeeze holds with position out (#227): 24 neurons 0.549 against 192 neurons 0.406]

[[lem-squeeze-conditioned-2026-10-20]]: the 42 drop-out nets re-read conditioned on the play count.
- The field net at 24 neurons is still cleaner than at 192: 0.549 against 0.406 over three seeds,
  with floors of 0.014 and 0.077. Every width from 12 neurons up is 0.33–0.53 above its floor.
- Below 12 neurons the units do the play count's own job, the mover's place in the trick. A
  conditioned read cannot credit that by construction; unconditioned they are far above their
  floor.
- The last words beyond position are the mover's trump: `holds-boss-trump` and `no-trump-held`.
- 4 of 4 predictions held.

CLAUDE.md's sentence goes to the Mayor with the conditioned numbers.

## [2026-10-20 | lem | the conditioned audit (#226): position out, both architectures clean above a near-zero floor; Gus's units keep their family 0.981]

[[lem-conditioned-audit-2026-10-20]]: each unit is z-scored within each play count before its top
1%.
- The untrained floors fall to near zero: Gus 0.014, field net 0.077.
- Trained Gus reads 0.275 and the 128×64 field net 0.406, on different rows.
- Gus's conditioned curve rises from about 0.01 to 0.28 with no U-shape. Its units keep their
  family as they form, 0.981 against 0.266.
- The field net carried position too, about a third of its unconditioned share, so that
  prediction was wrong. Earlier comparisons stand, but earlier absolute shares overstated what
  units name beyond position.
- 4 of 5 predictions held.

## [2026-10-20 | lem | the first transformer audit (Gus): its untrained units read position for free; training builds declaration and contract units that keep their family]

[[lem-gus-formation-2026-10-20]]: an all-neuron audit of Gus `pretrained-v0`, read-only on Codex's
checkpoints (#203), over 15 checkpoints and three untrained floors, with list v2:
- Untrained Gus units are 0.91–0.94 clean, nearly all by the mover's seat in the trick, because a
  transformer's decision token reads position. Trained Gus (0.517) reads below that floor. This is
  a tools rung, and a position-conditioned audit is filed.
- Training dissolves the free units (0.29 by step 5,000) and builds units named by the
  declaration, the contract and the count. They keep their family, 0.825 against 0.291.
- 3 of 5 predictions held.

The clean-neuron skill's first new friction log since the fold came from this run.

## [2026-10-20 | book | `scenario.hf verify`: count and name every published file, not just one checksum]

- [[publish-data-to-huggingface]] Verify: the Hub must hold exactly the staged files; `scenario.hf
  publish` checks after uploading, and study folders run `scenario.hf verify` (Mayor-approved).
- Why: the ch9 datasets had published one batch each and passed the one-file checksum (#201).

## [2026-10-20 | book | #201: the doubles cost lives in thin top doubles; ch9 data republished whole]

- [[ch9-at-the-table-2026-10-17]] follow-up: with ≤1 of 6-6/5-5/4-4 the declared-doubles cost is
  −0.17 to −0.47; with ≥2 it is level. Exploratory, to register; the book's Hand 23 is in a costly
  cell, so the reading is play. HF `book/ch9-field-v3/analysis/`.
- Both ch9 HF datasets had been flattened to one batch; every batch now published, summaries unchanged.

## [2026-10-20 | Gus value-numerics | Fixed-row precision diagnostic]

**Touched pages:** [[gus-value-numerics-2026-10-20]] [[gus-value-routing-2026-10-20]] [[gus-book-silas-tools]] [[index-experiments]]
**Added:** The separately registered 90-call fixed-row diagnostic found one finite float32 crossing of the unchanged `2e-5` audit at logit 19 (`2.002716064453125e-5`); same-case float64 discrepancy was `2.6645352591003757e-14`. The verified archive and independent saved-array review support local precision/order sensitivity only. The original factorial remains stopped, with its cause and primary prediction unresolved.

## [2026-10-20 | Gus value-numerics | Fixed-row diagnostic preparation]

**Touched pages:** [[gus-value-numerics-2026-10-20]] [[gus-value-routing-2026-10-20]] [[gus-book-silas-tools]] [[index-experiments]]
**Added:** A separately scoped fixed-row float32/float64 forward-path diagnostic is in preparation for the row where the prior instrument stopped. No trained response or result is available; the wrapper's raw-shape capture gap must be fixed and independently reviewed before registration or execution.

## [2026-10-20 | Gus value-versus-routing | Instrument stopped before primary comparison]

**Touched pages:** [[gus-value-routing-2026-10-20]] [[gus-book-silas-tools]] [[index-experiments]]

**Added:** The registered value/routing study is terminally complete as a stopped instrument audit after a numeric/nonfinite assertion and 42 original-parent row/model records (840 intervention cases plus 42 baselines). Failure row is identified from run order as `292:6`; exact condition, error magnitude and failed audit conjunct were not saved. Cosine-final was never measured, so the primary prediction remains unevaluated. The anonymously verified partial archive is linked from the experiment page. No scientific or playing conclusion follows. A separate one-row numerical diagnostic is being prepared under its own registration; it is not a resume of this study.
