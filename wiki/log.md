# Log

**This is a changelog, not a chronicle.** If an entry needs a paragraph, the paragraph belongs on a page — write the page and link it. An entry is the header line plus at most ~5 lines of pointers (touched pages, added, retired, one flag). The log carries no claim that isn't reachable through a link.

Entry prefix convention: `## [YYYY-MM-DD | shortsha | subject]` so `grep "^## \[" log.md log-archive.md` gives the full timeline.

**Rotation:** this file holds the digest below plus the most recent ~10 entries. When it grows past ~15, roll the oldest entries verbatim into [[log-archive]] and fold their one-line gist into the digest. Counting entries is the trigger — not judgment.

## Digest (2026-04-09 → 2026-07-14, 172 archived entries)

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

---

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

## [2026-10-10 | scenario-player | the forecast lies by 20 points and the cells are singletons from the bidder's third decision; cell-size floor is the fix]

Hold-my-hand marginal (64 fiber deals, Walt everywhere): root splat +30 / 83%
make vs real endings +10 to +13 / 39–45%, the same against the modeled
opponent. Cell trace: 3.1 worlds per cell at the second own decision, 100%
singletons from the third — clairvoyant below trick 2. Horizon sweep recorded.
Fix in flight: `Rung(floor=m)` hands thin cells to the rung below. See
[[scenario-player]].

## [2026-10-10 | 332ccfe8 | the atlas: Walt.analyze (per-world outcomes, the splat) and the mold (a deal's 374M positions in 1.9 s); beliefs paused]

Position → outcome distribution on one deal's perfect-information tree, poured
without peeking. "Reverse DP" resolved to no induction, no direction: labels
never read children. Two instruments landed, referee ↔ core bit-identical (100
tests). Mold seed 1: 374,004,900 positions, peak layer 23 at 87.4M. Pour price
per layer and the first slice measured; pruning plan (cone, hinges, cheat map,
waves) recorded. See [[scenario-player]].

## [2026-10-11 | scenario-player | belief night 1: auction and dice inference are nulls vs uniform Walt; the field tilt hurts (−0.96) by effective-sample-size collapse]

Five 384-game blocks, belief vs uniform two-rung Walt (the oracle-free
measuring stick), mirrored deals: auction −0.05, dice −0.06, auction+dice
−0.19 (all CI across zero); field ε=0.5 −0.65, ε=0.1 −0.96 (both negative at
power). ESS measured: field keeps 2–8% of 160 worlds. Lesson for the seam: a
strong g must be sampled from, not used to reweight a uniform draw. Exact
endgame enumeration landed. See [[scenario-player]].

## [2026-10-10 | scenario-player | Walt beats E[Q] n=10 at play: +0.74 marks/game [+0.42, +1.05], 384 games, dropped-30]

Compiled two-rung `walt:k1,n160,m8` vs `lens:ev` on Jason's exact protocol:
233/384 games (60.7%), both halves positive, points −4.5/hand (make utility),
17.5 min wall. Rung 0 alone: +0.18 [−0.12, +0.49]. The receipt replicates;
the partner rung is the edge. Uniform belief; belief levers untouched. See
[[scenario-player]] (the measured table).

## [2026-10-10 | scenario-player | compiled core lands: Rust/PyO3, bit-identical to the Python planner]

`scenario/core` + `scenario/compiled.py`: the planner and fiber sampler in Rust,
checked bit for bit against the Python referee (22 tests). Defined randomness
(SplitMix64 streams, packed record key, CPython-3.12 compensated sums
mirrored). Rung 0 at 160 worlds: 0.056 s at trick 1 (50× Python); two-response
160/8 g4: 1.1 s (26×). `decide_batch` plans a lockstep arena round in parallel.
Arena spec `walt` now runs the core; `py` selects the referee. Diagnosed the
texas-42 bridge's slowness (BigRational, 16× tie refinement, 8 lockstep
processes); killed the two six-hour blocks. See [[scenario-player]].

## [2026-10-09 | scenario-player | Walt rebuilt here from the mathematics; the wall diagnosed as the oracle itself]

**Touched pages:** [[scenario-player]] [[the-wall-biography]] [[index-entities|index]]
**Added:** [[scenario-player]] — `scenario/` (rules, record, exact-uniform fiber, planner) + `arena/walt_play.py`, spec `walt`; 17 tests; trick-1 decision 0.5 s (k0 n40) / 6 s (k1 default) in pure Python; 8/8 vs random.
**Updated:** [[the-wall-biography]] — the 2026-10-09 chapter: the project's only truth source was a perfect-information value, so every oracle-derived student had a flat gradient toward exactly what E[Q] lacks; the two wins (realized-outcome bidder, exact info-set walt) were the two non-oracle labels. The dream arrived elsewhere: Plunge, played by the family daily.
**Pending:** paired blocks `walt` vs `lens:ev` (k0 n40 ×32, k1 ×16) running at this entry.
**Doctrine:** this wiki is a finished book; live frontiers (texas-42, plunge) keep a notebook and frozen dated papers, never a frontier wiki.

## [2026-07-18 | worktree-parlor-promote | parlor promoted: salon on the wiki, session logs to HF, skill+workflow extracted]

**Touched pages:** [[parlor]] [[huggingface-assets]] [[run-artifacts-policy]] [[index-entities|index]]
**Added:** [[parlor]] — the four-seat salon (Jason, Fable, Sol, Opus; SSJ turn-taking, PASS/MARGIN/BID); session 1 record: warm-vs-cold readers, the 4-plays postmortem, the table catching its own re-enactment live, the OPEN ledger with wake conditions, the N-models-agree-trust-it-least reflex.
**Convention:** conversation/session logs classified — raw seat `.jsonl` tier 3 (HF `parlor/` @ tag `parlor-session-1`, gitignored), curated transcript tier 2 (committed + mirrored); [[run-artifacts-policy]] § Conversation and session logs.
**Reuse:** `.claude/skills/parlor/` (SKILL + workflow) and `parlor/bin/publish-session.sh` close the run → session-log → HF loop.

## [2026-07-17 | worktree-walt | walt: exact endgame info-set solver — built, graded, integrated]

**Touched pages:** [[walt]] [[walt-spec]] [[the-wall-biography]] [[jud]] [[champion-ladder]] [[the-wall]] [[the-gestation]] [[index-entities|index]] [[index-topics|index]]
**Added:** [[walt]] — eq's two deletions un-deleted (info-set-consistent continuation + exact B(σ) as a 0/1 filter) at ≤4 tiles vs the jud field; all gates green incl. T2 strategy-enumeration exactness and T6 claim-vs-cash closure. [[walt-spec]] — the machine on one page. [[the-wall-biography]] chapter: the door gets a hinge.
**Graded:** walt(W1,H4) **+3.00 [2.77, 3.22] marks/game** over jud play (88.1% game wins, n=512 paired); paired W1−W0 +0.283 [0.184, 0.391] → MIGHT-#3 graded: **term 2 carries 84.9% of the edge; term 1 is only cashable through term 2**. Scope: field model exact (opponents ARE jud); transfer vs `lens:ev` is #72. Numbers on [[walt]].
**Updated:** walker definition corrected — unbeatable when led, not "promoted trash"; home on [[walt]], coinage forward-linked from [[the-gestation]].
**Questions opened:** [#71](https://github.com/jasonyandell/mk5-main/issues/71) (predictions P1–P4) and the gated trajectory: [#72](https://github.com/jasonyandell/mk5-main/issues/72) transfer, [#73](https://github.com/jasonyandell/mk5-main/issues/73) scar probe, [#74](https://github.com/jasonyandell/mk5-main/issues/74) throughput, [#75](https://github.com/jasonyandell/mk5-main/issues/75) opening net, [#77](https://github.com/jasonyandell/mk5-main/issues/77) exploits.

## [2026-07-17 | worktree-fix-jud | jud's table badness root-caused; MRV contamination graded field-null]

**Touched pages:** [[jud]] [[world-sampler-mrv-audit]] [[index-experiments|index]]
**Added:** [[jud-v2-retrain-probe]] — the 0.68 claim was the round-0 checkpoint at the table; 10-agent audit: no live line-bug; clean-sampler + ε-explore retrain reproduces v1 grades within CI; play wall is mechanistic (9-of-350-dim ply-1 signal × greedy 1-ply × single MC labels).
**Also:** float32-unsafe-regime sampler regression test (cpu+mps); `champion/jud_net_v2_r{0..2}.pt` + evidence under `champion/evidence/jud_v2/`.
**Receipts:** [#66](https://github.com/jasonyandell/mk5-main/issues/66) · [#69](https://github.com/jasonyandell/mk5-main/issues/69).

## [2026-07-16 | conversation | the belief/policy/value algebra promoted; measurement program filed]

**Touched pages:** [[belief-policy-value-algebra]] [[strategy-fusion]] [[count-fate-ledger]] [[index-topics|index]]
**Added:** [[belief-policy-value-algebra]] — tilt form (b = u·e^g; eq is the g≡0 limit), coupling theorem, information identity, eq located as "π deleted twice," exact gap and claim-vs-cash decompositions, the (ε, init) family; conclusions tiered CAN (mathematical) vs MIGHT (conjectures, each paired with its deciding probe).
**Questions opened:** [issue #64](https://github.com/jasonyandell/mk5-main/issues/64) — the M1–M6 measurement program (meaning map, tilt profile, channel bandwidth, accidental-convention detector, realized tiger, field docility).

## [2026-07-15 | conversation | the argument's referees — Phase 2 grading doctrine ratified]

**Touched pages:** [[count-fate-ledger]] [[strategy-fusion]] [[otis]]
**Added:** "The argument's referees" on [[count-fate-ledger]] — the outcome leak named ("would it have worked *more often*", belief-averaged tied grading), dispersion-triaged lesson harvesting, the three-referee split, the claim-vs-cash gap as the loop's convergence metric; "eq is not 42" coda on [[strategy-fusion]].
**Updated:** [[otis]] design commitments route to the doctrine; amendments filed as a comment on issue #55.

## [2026-07-15 | b347897b | otis v0: count-fate ledger built and graded overnight (#49)]

**Touched pages:** [[otis]] [[otis-v0]] [[count-fate-ledger]] [[world-sampler-mrv-audit]] [[jud]] `questions/open.md`
**Added:** [[otis]] [[otis-v0]] — the fate-ledger-native player, all seven registered predictions graded on the branch `worktree-otis-v0` night
**Updated:** [[count-fate-ledger]] (new Measured section; open question narrowed), [[world-sampler-mrv-audit]] (corpus-scale contamination quantified, issue #52), [[jud]] (sibling link)
**Questions opened:** issues #51 (doubles-suit engine representability), #52 (corpus regeneration on repaired sampler), #53 (retention-policy consumer — the remaining half of #49's question)
