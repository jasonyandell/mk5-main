# Log

**This is a changelog, not a chronicle.** If an entry needs a paragraph, the paragraph belongs on a page — write the page and link it. An entry is the header line plus at most ~5 lines of pointers (touched pages, added, retired, one flag). The log carries no claim that isn't reachable through a link.

Entry prefix convention: `## [YYYY-MM-DD | shortsha | subject]` so `grep "^## \[" log.md log-archive.md` gives the full timeline.

**Rotation:** this file holds the digest below plus the most recent ~10 entries. When it grows past ~15, roll the oldest entries verbatim into [[log-archive]] and fold their one-line gist into the digest. Counting entries is the trigger — not judgment.

## Digest (2026-04-09 → 2026-10-11, 194 archived entries)

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
- **otis, the algebra, walt endgame, parlor** (2026-07-15 → 07-18, 6 entries) — [[otis]] [[otis-v0]] graded, the referees doctrine on [[count-fate-ledger]], [[belief-policy-value-algebra]] promoted, [[jud-v2-retrain-probe]], [[walt]] endgame solver (+3.00 over jud), [[parlor]] promoted.

---

## [2026-10-12 | scenario/core | the posterior sampled in the core (Config.draw, arena d=posterior): +0.026 ± 0.003 marks/hand vs a table of nets, 22,000 paired deals (#111)]

[[scenario-player]] § The posterior, sampled in the core. Default uniform; Zeb unchanged.

## [2026-10-12 | scratch/endgames-o2 | O2 beats the O1 fixed point on real three-trick endgames: +0.027 [+0.022, +0.033] margin vs the Walt partnership map, half of it a convention on reached sets (#113)]

[[endgames-o2-vs-o1-2026-10-12]]; report [[endgames-o2-vs-o1-2026-10-12-report]]. The first nonzero O2 − O1 number; the partnership operator is the next referent.

## [2026-10-12 | scenario/core | a stochastic field F_σ (field_eps, arena E=), default off, and the exact field likelihood: real deal at the 71st percentile, median ESS 8–16 of 512 mid-hand (#108)]

[[scenario-player]] § A stochastic field. Prerequisite for #111.

## [2026-10-12 | scenario/core | the sample seed no longer mixes the rung: the sampled ladder iterates one operator, as the Lean theorem assumes (#112)]

[[scenario-player]] § The seed no longer mixes the rung. Bytes moved, not distribution. Witness: [[research-night-2026-10-12-sources]] (11-lean).

## [2026-10-12 | scenario/core | the floor guards only a sample: an enumerated fiber is never floored, the only behavior in both engines; Zeb 0.2.0 (#110)]

[[scenario-player]] § The floor guards only a sample. Null in marks; correctness. Report: [[research-night-2026-10-12-sources]] (14-floor).

## [2026-10-12 | field_train | the field net is graded by regret under the splat's q, not agreement (#109): shipped 250k net 0.0064 make probability per decision, n = 714]

[[scenario-player]] § The net is scored by regret. Source: [[research-night-2026-10-12]] critic §3.2, §4e.

## [2026-10-12 | cb95168e | Research night: does Zeb find the optimal move in finite time?]

Seven agents (theory, Lean, counter lab, empirical, floor, an independent Fable
session, a Fable critic). Proved: exact Walt with the field's posterior is an O1 best
response in finite time; the fixed-belief ladder settles by the 24th application
(Lean, `scenario/lean/`, 15 theorems, no sorry); coherent beliefs cycle. Measured:
the belief gap is +0.028 marks/hand against a net table; Zeb is a one-step rollout
of its net after the root; floor-off-when-exact is correct and null. Pages:
[[research-night-2026-10-12]], sources [[research-night-2026-10-12-sources]]; hub
[[scenario-player]] gets a dated paragraph.

## [2026-10-12 | scenario/viz | what must be true, in the browser: pick a region of a move's splat, see the holdings that explain it; condition the position on up to three holdings; per-world sample (687 KB); zoom + touch stepping]

[[splat-visualizer]] § 2026-10-12. Reads `holding_predictors` / `makes` per world ([[scenario-player]] § What must be true).

## [2026-10-11 | scenario/viz | the splat visualizer: a pour in the browser — each legal move's splat per planned play, means-only strip, the real ending through every forecast, the hand as a ridgeline; exporter + 5 tests; 79 KB sample]

[[splat-visualizer]] (new); [[scenario-player]] § The splats, drawn. Successor to [[eq-browser-visualizers]].

## [2026-10-11 | 00a51f24 | Zeb, the package: @jasonyandell/zeb 0.1.0 — the cheap player in WebAssembly, bit-identical to the referee on 34 positions, 207 KB, 0.3 s for seven declarations; scenario/core a plain Rust library (rules derived in Rust, net as one blob, pyo3 behind a feature); release workflow via trusted publishing]

[[scenario-player]] § Zeb, the package. `zeb/README.md` is the user's document. Built and cold-reviewed by two Opus agents; the review's two should-fix items and nits all landed.

## [2026-10-11 | scenario-player | a live bidder graded against Plunge's frozen bid book: 4,680 panels, |Δpmake| 0.052 vs the book's own sd 0.066, calibrated through the middle, 3–5 pp bold at the top; seven declarations in 0.2–1.0 s]

[[scenario-player]] § A live bidder, graded against Plunge's book. Scripts in `scratch/bid/`. Not a deployment.
