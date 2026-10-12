# Log

**This is a changelog, not a chronicle.** If an entry needs a paragraph, the paragraph belongs on a page — write the page and link it. An entry is the header line plus at most ~5 lines of pointers (touched pages, added, retired, one flag). The log carries no claim that isn't reachable through a link.

Entry prefix convention: `## [YYYY-MM-DD | shortsha | subject]` so `grep "^## \[" log.md log-archive.md` gives the full timeline.

**Rotation:** this file holds the digest below plus the most recent ~10 entries. When it grows past ~15, roll the oldest entries verbatim into [[log-archive]] and fold their one-line gist into the digest. Counting entries is the trigger — not judgment.

## Digest (2026-04-09 → 2026-10-10, 184 archived entries)

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
- **otis, the algebra, walt endgame, parlor** (2026-07-15 → 07-18, 6 entries) — [[otis]] [[otis-v0]] graded, the referees doctrine on [[count-fate-ledger]], [[belief-policy-value-algebra]] promoted, [[jud-v2-retrain-probe]], [[walt]] endgame solver (+3.00 over jud), [[parlor]] promoted.

---

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

## [2026-10-11 | 7c31bbf9 | Zeb: the 250,000-deal pour (4.34M positions, four overnight chunks poured by the small net) — imitation improves (regret 0.38 → 0.26), the arena does not (+0.11 / +0.13 vs the champion); the data lever stops at parity]

[[scenario-player]] § The 250,000-deal pour; `field_train` takes several pours; dataset jasonyandell/texas-42-zeb-pour-200k (CC0). Champion unchanged.

## [2026-10-10 | 99e8ec32 | Zeb: the 50,000-deal pour (864,604 positions) and the field net at parity — +0.11 [−0.20, +0.40] / +0.06 [−0.25, +0.38] vs the champion; the name reclaimed]

[[scenario-player]] § The 50,000-deal run — and the name; [[zeb]] name note; dataset jasonyandell/texas-42-zeb-pour (CC0).

## [2026-10-10 | f533cb7a | the field net's two targets: faithful pmake BCE −0.46 [−0.76, −0.15], margin −0.01 [−0.31, +0.30]; bid one-hot (field-v2); MLX path]

[[scenario-player]] "The net in the field" § Two targets; `scenario/field_train.py` `--target`, `--device mlx`.

## [2026-10-10 | 882b36fe | the net in the field, graded on the make re-pour: −0.16 [−0.48, +0.14] vs the champion at a quarter of the cost — a cheaper equal; the τ-units bug]

[[scenario-player]] "The net in the field" § The make run; `scenario/field_train.py` target read off the margin splat (#107).

## [2026-10-10 | scenario-player | the net in the field: rung −1 as a learned integer policy (design + smoke)]

[[scenario-player]] "The net in the field": `scenario/field.py`, `field.rs`, `field_train.py`, arena `F=`; referee ↔ core bit-identical. Smoke only; the real run waits for the #107 `make` re-pour.

## [2026-10-10 | 867b2ed6 | leaves keep the margin under every utility (#107): the utility shapes the plan, the splat keeps the margin]

`scenario/planner.py` `_trace`, `scenario/core/src/planner.rs` `trace` (bit-identical), `atlas.in_plan_units`; see [[scenario-player]] § Leaves keep the margin. Re-pour under `make` in flight.

## [2026-10-10 | 567a8b9a | the pour published: jasonyandell/texas-42-walt-pour on Hugging Face, CC0, 5,000-deal train + 200-deal validation]

`scenario/hf.py` (positions / splats / deals Parquet + raw npz + card); see [[scenario-player]] § Published.

## [2026-10-10 | 3ecf46db | the pour: Walt self-play in waves, every legal move's whole splat kept — the advantage dataset]

`scenario/atlas.py`: `play_waves`, `pour`, save/load, `advantages`,
`paired_advantages`, `success_set`, `holding_predictors`, `calibration`; CLI.
117 tests. Feeds advantage-weighted policy training (GRPO-shaped, group =
legal moves) and #94-style pattern mining. See [[scenario-player]].

## [2026-10-10 | scenario-player | the floor's dividend: `k1,n512,m8,f=16` beats the old champion +0.56 [+0.26, +0.86], 384 mirrored games; honest plan is the arena default]

Compute saved by not searching a clairvoyant tail, spent on root worlds
(160 → 512), buys +0.56 marks/game at power (made 38.2% vs 33.1%). Floor 16
alone was a null (+0.13). `walt:k1,…` now defaults to `f=16`. See
[[scenario-player]].

## [2026-10-10 | 1406fdae | the cell floor lands: calibrated forecast (+10.4/35% vs real +11.5/34%), play a null leaning positive (+0.13 [−0.20, +0.45]) at under half the nodes]

`Rung(n, horizon, floor=m)`, arena `f=<m>`; below the root a cell thinner
than m hands the viewer's move to the rung below (resampling inside the
plan). Floor 8+ ≡ all interior decisions handed down. Floor 1 unchanged to
the bit; 104 tests. What-must-be-true receipt on the trick-5 hinge (per-world
paired outcomes → "6-1 makes iff partner holds the 6-6"). See [[scenario-player]].
