# Log

**This is a changelog, not a chronicle.** If an entry needs a paragraph, the paragraph belongs on a page — write the page and link it. An entry is the header line plus at most ~5 lines of pointers (touched pages, added, retired, one flag). The log carries no claim that isn't reachable through a link.

Entry prefix convention: `## [YYYY-MM-DD | shortsha | subject]` so `grep "^## \[" log.md log-archive.md` gives the full timeline.

**Rotation:** this file holds the digest below plus the most recent ~10 entries. When it grows past ~15, roll the oldest entries verbatim into [[log-archive]] and fold their one-line gist into the digest. Counting entries is the trigger — not judgment.

## Digest (2026-04-09 → 2026-07-11, 162 archived entries)

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
- **Continuation frontier + research night** (2026-07-12 → 07-13, 7 archived entries) — [[belief-weighted-jud-mcts]] and [[convention-aware-blueprint-search]] preserved with framing rebalanced; [[research-lane-selection]] picked the night's lanes, Stage 0 closed, Lane A ([[auction-decoder]]) v0 validated, Lane B ([[dense-q-supervision]]) graded both rounds. Source: [[sources/research-night-2026-07-13|research-night-2026-07-13]].

---

## [2026-07-21 | conversation | full-metal Hoyt problem statement]

**Touched pages:** [[full-metal-hoyt]] [[hoyt]] [[perf]] `questions/open.md` [[index-topics|index]]
**Added:** [[full-metal-hoyt]] — move Hoyt's ragged root fleet, full-width build, CFR+ sweeps, and reference evaluation into a fast device-resident Metal lane; H4 is calibration, practical H5 and H6 lines are the goals.
**Gates:** held-out error-bar coverage for value, gap, convergence, and action damage; H5 throughput; H6 throughput; consumable reference artifacts.
**Non-goals:** bit replication, identical reductions, per-root CPU certification, and any H4-only win.

## [2026-07-17 | worktree-walt | walt: exact endgame info-set solver — built, graded, integrated]

**Touched pages:** [[walt]] [[walt-spec]] [[the-wall-biography]] [[jud]] [[champion-ladder]] [[the-wall]] [[the-gestation]] [[index-entities|index]] [[index-topics|index]]
**Added:** [[walt]] — eq's two deletions un-deleted (info-set-consistent continuation + exact B(σ) as a 0/1 filter) at ≤4 tiles vs the jud field; all gates green incl. T2 strategy-enumeration exactness and T6 claim-vs-cash closure. [[walt-spec]] — the machine on one page. [[the-wall-biography]] chapter: the door gets a hinge.
**Graded:** walt(W1,H4) **+3.00 [2.77, 3.22] marks/game** over jud play (88.1% game wins, n=512 paired); paired W1−W0 +0.283 [0.184, 0.391] → MIGHT-#3 graded: **term 2 carries 84.9% of the edge; term 1 is only cashable through term 2**. Scope: field model exact (opponents ARE jud); transfer vs `lens:ev` is #72. Numbers on [[walt]].
**Updated:** walker definition corrected — unbeatable when led, not "promoted trash"; home on [[walt]], coinage forward-linked from [[the-gestation]].
**Questions opened:** [#71](https://github.com/jasonyandell/mk5-main/issues/71) (predictions P1–P4) and the gated trajectory: [#72](https://github.com/jasonyandell/mk5-main/issues/72) transfer, [#73](https://github.com/jasonyandell/mk5-main/issues/73) scar probe, [#74](https://github.com/jasonyandell/mk5-main/issues/74) throughput, [#75](https://github.com/jasonyandell/mk5-main/issues/75) opening net, [#77](https://github.com/jasonyandell/mk5-main/issues/77) exploits.

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

## [2026-07-13 | wiki-reorg | schema v2, routing trails, status truth, split catalog]

**Schema:** [[AGENTS.md|AGENTS]] amended — 4-value status enum with lifecycle rules, date timestamps, kind decision tree, one-home-per-fact, routing rules, named sources codified; `scripts/wiki_lint.py` enforces mechanically (`--strict` clean at this entry).
**Added:** [[trails/gus-line|gus-line]] [[trails/burl-line|burl-line]] [[trails/champion-ladder|champion-ladder]] [[entities/stage-0-adapter-line|stage-0-adapter-line]] [[entities/burl-adapter-line|burl-adapter-line]] [[decisions/beads-to-gh-issues|beads-to-gh-issues]] [[playbooks/research-night|research-night]]; catalog split into `index-<kind>.md`.
**Updated:** hubs thinned to budget (gus 726→98, burl 586→116, champion/jud/arena/w42); ~120 stale statuses flipped; w42 rollups canonicalized on [[experiments/w42-book-claim-synthesis-and-ai-directions|w42-book-claim-synthesis-and-ai-directions]]; 84-claim + reentry verdicts reconciled; 17 pivot-dead questions moved to resolved.
**Moved:** 16 winning42 chapter digests → `sources/`; 7 era chronicles → `trails/`; 10 adapter receipts → `experiments/`; selfplay-arena → [[entities/burl-selfplay-arena|burl-selfplay-arena]].
**Record:** `docs/wiki-reorg-proposal-2026-07.md` (the adopted proposal; audit evidence in its appendix).
## [2026-07-13 | champion-fold | champion folded into jud — aspirational name retired]

**Decision:** "champion" named the player before it existed; the built thing is [[jud]]. Asset map, decision loop, and auction-dominance analysis moved to [[entities/jud|jud]]; teaching half to [[topics/the-wall|the-wall]] as a declared side benefit; [[entities/champion|champion]] reduced to a superseded pointer.
**Sweep:** 74 live pages retargeted by meaning (player → jud; era/rungs → [[trails/champion-ladder|champion-ladder]]; goal → the-wall). GitHub milestone **Champion** and repo dir `champion/` keep the name (BUILT); CLAUDE.md frontier line updated.

## [2026-07-14 | 45fe646b | count-fate ledger — the consumption object formulated]

**Touched pages:** [[count-fate-ledger]] [[the-wall]] [[past-belief-future-direction]] [[w42-phase2-hidden-domino-threat-attribution]]
**Added:** [[count-fate-ledger]] — hand value as a belief-weighted ledger of count-fate scenarios (IDEATED, conversation 2026-07-13→14, issue #49); guards/walkers as one junk-retention economy per-world E[Q] cannot price ([[strategy-fusion]]).
**Updated:** [[the-wall]] contextual-distribution direction now names its consumption object; threat-attribution grain framed as one factor of row probability.
**Questions opened:** tied-strategy rollouts pricing guard/walker retention (questions/open.md, issue #49).

## [2026-07-18 | worktree-walt-perf | walt wavefront engine + the perf umbrella topic]

**Touched pages:** [[topics/perf|perf]] [[topics/walt-spec|walt-spec]] [[entities/walt|walt]] [[topics/perf-on-the-table|perf-on-the-table]]
**Added:** [[topics/perf|perf]] — umbrella topic: eight measured laws synthesized from the two perf campaigns (2026-04 Burl inference, 2026-07 walt solver), plus current fast-path state.
**Updated:** [[topics/walt-spec|walt-spec]] §4 rewritten for the wavefront engine (14.8× at exact parity on 46 golden fixtures; world-cap error curve: material-flip is the right gate, raw flips are near-ties; #74 stage-1 "10×" corrected to 1.7–2.3×); corpus caveat added (§3: pilot arm A banked no serialized roots). [[entities/walt|walt]] engine bullet + index hooks. Receipts: PR #78, issues #74/#73.

## [2026-07-18 | worktree-walt-perf | hoyt — the net-free referee, promoted]

**Touched pages:** [[entities/hoyt|hoyt]] [[entities/walt|walt]] [[topics/walt-spec|walt-spec]] [[topics/perf|perf]] [[topics/perf-log|perf-log]]
**Added:** [[entities/hoyt|hoyt]] — the game's own referee: net-free kernel + verified CFR+ + exploitability meter + frozen eval anchor, promoted out of walt/kernel/ to `hoyt/` the morning after it was built. Named for "according to Hoyle." Player/referee split registered: walt = BR vs a modeled field; hoyt = values with no model in the loop (the stable eval across model generations).
**Updated:** walt-spec §7.6 now routes to hoyt; walt entity gains the referee-split bullet + first rent reading (median +1.9 pts/root vs jud); perf fast-path state; perf-log 2026-07-18f (promotion entry; a–e left as history per append-only rule).

## [2026-07-20 | worktree-equiv-census | endgame equivalence census — exact fungibility measured]

**Touched pages:** [[topics/endgame-equivalence-census|endgame-equivalence-census]] [[entities/hoyt|hoyt]] [[topics/suit-algebra-spec|suit-algebra-spec]] [[topics/play-phase-algebra|play-phase-algebra]]
**Added:** [[topics/endgame-equivalence-census|endgame-equivalence-census]] — the residual-signature instrument (`hoyt/equivcensus.py`, certified) and its verdicts: world/root compression exactly 1.000× on the frozen 200-root evalset (co-occurrence theorem — dead at every horizon); within-hand interchangeable pairs in 25/200 roots with 29/29 bitwise value-tie receipts via br_solve.
**Updated:** suit-algebra-spec §9 lead-direction bullet (only the monotone relabeling transports per-deal play trees; 0/200 measured); play-phase-algebra §8.3 sharpened; hoyt measured table.

## [2026-07-20 | worktree-equiv-census | cfr-primer — regret at the table]

**Touched pages:** [[topics/cfr-primer|cfr-primer]] [[entities/hoyt|hoyt]] [[index-topics]]
**Added:** [[topics/cfr-primer|cfr-primer]] — CFR in project units (info sets as table perspectives, counterfactual reach as the cross-seat coupling, RM+ inertness of forced isets, the measured-gap honesty line, one worked trick, code map, reading path incl. GameShrink/PBS anchors for the census lineage).

## [2026-07-20 | worktree-equiv-census | class-CFR incidence census — sizing the merge]

**Touched pages:** [[topics/endgame-equivalence-census|endgame-equivalence-census]]
**Added:** "Sizing the class-CFR lane" section — own-hand EQUALS predicate over all reachable CFR info sets (four seats, every depth; per-info-set instrument certified reproducing `interchangeable_pairs` incl. the 6-0/6-1/6-2 triple, with rank comparisons dropping own-hand tiles = relax_meme made local). cap-64 primary (10 stratified roots, 39.4M info sets): forced **82.6%**, non-forced strategy slots removed **2.94%**, class-forced 376k (5.5% of non-forced).
**Verdicts:** prior "incidence rises with depth" **refuted** — falls (4.4%→3.7%→2.9%, last trick 100% forced); slot-reduction prior confirmed but low (~3%, not 15%); class-forced confirmed; merge concentrated in hidden seats (3.60%) vs the visible seat (0.68%). Cap-sensitive: more worlds → slightly less merging (cap-16 3.22% → cap-64 2.94% same roots; cap-256 unrun, out of budget). Verdict: exact + ~1µs to detect, but a ~3% tidy on top of the ~83% forced-slot compression (#82), not a width win.
**Flag:** cap-64 shrunk to 10 roots (not 20) to stay in budget; cap-16 run on the full 20 for breadth (2.86%). Receipts in `scratch/equiv-census/`.
