# Log

**This is a changelog, not a chronicle.** If an entry needs a paragraph, the paragraph belongs on a page — write the page and link it. An entry is the header line plus at most ~5 lines of pointers (touched pages, added, retired, one flag). The log carries no claim that isn't reachable through a link.

Entry prefix convention: `## [YYYY-MM-DD | shortsha | subject]` so `grep "^## \[" log.md log-archive.md` gives the full timeline.

**Rotation:** this file holds the digest below plus the most recent ~10 entries. When it grows past ~15, roll the oldest entries verbatim into [[log-archive]] and fold their one-line gist into the digest. Counting entries is the trigger — not judgment.

## Digest (2026-04-09 → 2026-10-12, 206 archived entries)

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
- **otis, the algebra, walt endgame, parlor** (2026-07-15 → 07-18, 6 entries) — [[otis]] [[otis-v0]] graded, the referees doctrine on [[count-fate-ledger]], [[belief-policy-value-algebra]] promoted, [[jud-v2-retrain-probe]], [[walt]] endgame solver (+3.00 over jud), [[parlor]] promoted.

---

## [2026-10-13 | d83c7b1d | Silas engineering pass: a policy is a function of the frame, one frame with derived views, physics as tables with the fiber Silas's own, the contract's kind and stake, authority a lattice; the book prelude; the `silas` command]

**Updated:** [[silas]] § The engineering pass; `silas/README.md` (§ Semantics, the denotation written down); `silas/GAPS.md` § Closed. Zeb: `ZebPhysics` a table provider, its own rules under differential test. Pins: `silas/tests/book_snapshot.rs`.

## [2026-10-13 | src/multiplayer | system authority only from the server: a client's forged `meta.authority` is refused; the auto-execute loop runs `executeSystemAction` (#96)]

**Updated:** [[multiplayer-pattern]] § Capability system (two authorization paths; `resolveSessionForAction` retired).

## [2026-10-13 | scenario/reason | Zeb's reason report (#124): the plan's line in both engines, opt-in and bit-identical; the separating facts as Silas queries with a noise bar; the 84's reason found by search at 33/95]

**Updated:** [[scenario-player]] § The reason report. Code: `scenario/reason.py`, `Walt.analyze(line=True)` / `core::planner::analyze_line`, `SilasBelief.condition_likelihood`. Worked examples: `scratch/reason/`.

## [2026-10-13 | scratch/silas-reprice | the book's rules as Silas policies, priced on the whole 250k pour: 17 of 21 fire exactly where their gates do; C22 as the book's rule sides more (+0.58, 5% against); C15 and C2 re-measured weak]

**Updated:** [[book-study-2026-10-12]] § The book's rules as Silas policies (fidelity and price tables; verdict lines for C2, C15, C22 corrected in place); [[w42]], `index-experiments.md` (sides with 8 rules).

## [2026-10-13 | 0c5d7f66 | Silas slice 3: the record in the frame, `none`, ten record predicates; 20 of 21 book rules encode their gate, at every seating]

**Updated:** [[silas]] § The book, rule by rule, § On Zeb's rules. Code: `silas/` (frame, eval, registry, policy), `silas/GAPS.md` (Closed: #126, #127, #129, #131, #133).

## [2026-10-13 | silas | the book, rule by rule: 21 gates as Silas policies on the book's own deals; at least 10 of 21 Python gates misread the book; ten gaps filed]

**Updated:** [[silas]] § The book, rule by rule; [[book-study-2026-10-12]] (the per-rule prices are the gates', not the book's). Also landed: Silas on Zeb's rules (`ZebPhysics`, differential test, Python bindings).
**Issues opened:** #126–#135 (`silas` label), the gaps in `silas/GAPS.md`.

## [2026-10-13 | silas | Silas on Zeb's rules: ZebPhysics over Rules + Fiber, differentially tested against Straight42 (no disagreement); Python bindings in scenario_core; legal_plays keyed by the led tile]

**Updated:** [[silas]] (§ On Zeb's rules, Next). Code: `scenario/core/src/physics.rs`, `python_silas.rs`, `tests/silas_physics.rs`; `scenario/silas.py`, `scenario/tests/test_silas.py`.
**Flag:** the hub must reinstall the core wheel (`python -m pip install ./scenario/core`): maturin now builds with `silas`.

## [2026-10-12 | silas | Silas: the relational query and policy language moved from texas-42, renamed, over a Physics boundary; exact numbers with authority labels; the 84 hand at exactly 33/95]

**Added:** [[silas]] (entity); `silas/` crate (54 tests). **Updated:** CLAUDE.md project bullet; index-entities.
**Ruling:** off-suit plays tie at the bottom of the trick key (texas-42 lean `Declaration.key`), not by pip.

## [2026-10-12 | wiki | the book studies corrected: chance baselines everywhere; neither era established a strategic claim from the book; Zeb prices moves but cannot yet explain one]

**Touched pages:** [[book-study-2026-10-12]] [[w42]] [[w42-book-claim-synthesis-and-ai-directions]] (§ What the evidence is worth) [[w42-book-validation-campaign]] [[w42-phase2-statistics-claims-ledger]] [[w42-book-validation]] and 44 more w42/chapter pages; CLAUDE.md player bullet.
**Updated:** Zeb era — 52% vs a 37% random floor, per-gate for/indifferent/against, "level" and the dead-row revival withdrawn. Oracle era — "supported" is arithmetic/definitions/controls/a tautology/a null; setter-side verdicts withdrawn (#122).
**Issues opened:** #122 (E[Q] sign), #123 (the book's examples as positions), #124 (Zeb's reason report), #125 (ledger CSV).

## [2026-10-12 | scratch/faceoff | face-off closed: the four "contradicts" verdicts survive a k1,n2048 re-plan (prices unchanged, 91–97% survive); dead rows revived at a live bid put Zeb level with the oracle on the book (55.1% vs 56.4%); the book is the authority on reasoning, a counterexample is the standard for siding against it (#121)]

[[book-study-2026-10-12]] § The face-off (re-plan table, "what agreement should mean"). Data: research archive `book/faceoff/`. #119 closed; #120, #121 open.

## [2026-10-12 | scratch/faceoff | the book face-off (#119): at the same gates the oracle's E[Q] n=10 picks the book's tile more often than Zeb (56.4% vs 51.5%), the field net more still; "Zeb and the book agree" corrected to "the book's tile is cheap under Zeb's plan"; the oracle-era verdicts agree in sign where they overlap]

[[book-study-2026-10-12]] § The face-off (new section; summary bullets corrected); [[w42]] paragraph corrected; CLAUDE.md player bullet. Regret under Zeb's q separates strong from weak; book agreement barely does. Ties: 17.6% of gated rows dead (#120). Stronger-Zeb re-plan still running.

## [2026-10-12 | scratch/book | the book graded with the honest tools: rules priced on the pour, bid book 25/30, three odds corrected, conventions null in S/R/SR arms (handoff −0.013 sender, reading +0.000)]

[[book-study-2026-10-12]] (new); [[scenario-player]] § A convention overlay (arena verdicts); [[w42]] (dated paragraph). Data: the research archive's `book/wave1/`, `book/wave2/`. Follow-ups #118.

## [2026-10-12 | scenario/hf | generated data goes to Hugging Face: the 250k Zeb pour whole, the day's research artifacts, the playbook]

[[publish-data-to-huggingface]] (new); [[scenario-player]] links texas-42-zeb-pour-250k and texas-42-zeb-research-2026-10; CLAUDE.md hard rule. `scenario.hf` ships field nets, titles, notes; splits may differ in the net.
