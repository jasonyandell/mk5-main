# Log

**This is a changelog, not a chronicle.** If an entry needs a paragraph, the paragraph belongs on a page — write the page and link it. An entry is the header line plus at most ~5 lines of pointers (touched pages, added, retired, one flag). The log carries no claim that isn't reachable through a link.

Entry prefix convention: `## [YYYY-MM-DD | shortsha | subject]` so `grep "^## \[" log.md log-archive.md` gives the full timeline.

**Rotation:** this file holds the digest below plus the most recent ~10 entries. When it grows past ~15, roll the oldest entries verbatim into [[log-archive]] and fold their one-line gist into the digest. Counting entries is the trigger — not judgment.

## Digest (2026-04-09 → 2026-06-14, 145 archived entries)

What the log recorded, phase by phase. The story itself lives in the hubs and trails linked here; this is only a map of when the wiki ingested what.

- **LEM** (2026-04-09 → 04-17, ~17 entries) — the wiki's first trail: [[stage-0-adapter]] v1→v10, the STaR plateau at 38–48%, the Gemma→[[qwen3-1.7b]] pivot, [[qwen3-14b]] capacity, the SFT mask fix. Hub: [[lem]].
- **Burl** (2026-04-18 → 04-23, ~10 entries) — sibling project kickoff: Moves 1–4, iters 0–5, the reasoning-coherence bottleneck named ([[candlewax]]), wax_museum, [[belief-trajectory]]. Hub: [[burl]].
- **Gus** (2026-04-20 → 04-22, ~10 entries, interleaved with Burl) — [[joint-world-tensor]] through the [[lamir1-ceiling]] to Gus FINAL in five days: v0→v3 students, blunder detector, belief calibration, co-train falsified. Hub: [[gus]].
- **Burl harvests + perf sprint** (2026-04-25 → 04-28, ~25 entries) — STaR run-3/3b/3c (preserve-thoughts is load-bearing), resumable checkpointing, the 2000-decision harvest, and the [[perf-sprint]] playbook's many small revisions plus [[burl-perf-phase0]]–phase3.
- **burl-lab / burl-chat / microscope + w42** (2026-04-30 → 05-07, ~65 entries; the log's densest stretch) — two parallel campaigns: the [[burl-lab]] workbench lane, and the [[w42]] book-validation campaign (claim ledger, branch atlases, four phases, waves 1–4, lens head-to-head). Trail: [[w42-book-validation]]. Also [[book-strategy-player]] design and the first wiki-curation pilot (hubs + trails).
- **Champion rungs** (2026-06-09 → 06-14, ~16 entries, after a five-week gap) — the [[champion]] ladder: arena harness, rungs #21–#28 (auction-conditioned belief the measured win; several honest nulls), [[jud]] direction and vocabulary, the Fable provenance corrections. Hub: [[champion]].
- **jud + the wall** (2026-07-05 → 07-06, 6 archived entries) — jud v0/v1 built and graded ([[jud]], [[champion]], [[w42-jud-v1]], [[w42-plateau-probe]]), the 2.38× arena perf pass, value-native endorsed via [[rank-vs-price]].
- **The wall named + the archaeology** (2026-07-06 → 07-10, 5 archived entries) — [[the-wall]] stated precisely (distill-for-what, [[candlewax]] concordance); the ~30-page era backfill + ~173-page staleness audit and anti-rot rules; [[w42-book-second-pass]]; the 162-experiment-page artifact audit (43 corrected); the log's changelog-not-chronicle rotation. Later 07-11+ entries remain below.

---

## [2026-07-11 | bc4eb386 | partnership wall — cumulative record to measurement spine]

**Touched pages:** [[partnership-wall-research]] [[partnership-value]] [[partnership-research-gates]] [[the-wall]] [[consumption-ledger]] [[arena]] [[forge]] [[champion]]
**Added:** [[partnership-failure-atlas-v0]] [[world-sampler-mrv-audit]] [[partnership-decision-record-v1]] [[sources/bc4eb386]]
**Measured/built:** five-way 75,079-action join + 114-source seam inventory; legacy MRV malformed/bias mechanisms; rejected uniform-rejection repair; exact completion-count sampler; replay-verified Arena records with C0 policy and leakage fingerprints.
**Frontier:** CUDA/MPS sampler performance, historical exposure, two-block C0 reproduction, forced causal arms, and information-reactive fixed/shuffled partnerships remain open; no successor architecture selected.

## [2026-07-11 | a2bb0437 | result vocabulary — partnership remains untested]

**Touched pages:** [[partnership-wall-research]] [[partnership-failure-atlas-v0]] [[world-sampler-mrv-audit]] [[partnership-research-gates]] [[sources/a2bb0437]]
**Updated:** archive insufficiency is not a partnership null; three no-flip sampler fixtures are a bounded observation; the confounded `~6.8 Q` estimate stays retired; failed uniform rejection is the genuine negative design result.
**Frontier:** no negative or null result about partnership value has been measured.

## [2026-07-11 | 5f314d2b | partnership research review surface]

**Touched pages:** [[partnership-wall-research]] [[sources/5f314d2b]] [[index]]
**Updated:** one review-first table now distinguishes measured/built/open/untested/designed work; four ordered gates route baseline cleanup → causal runner → first partnership discriminator → architecture selection.
**Frontier:** [[partnership-wall-research]] is the single PR-review entrypoint.

## [2026-07-11 | b89ff635 | docs→wiki consolidation: game-of-42 cluster, engine second pass, forge/burl/lem/gus promotion, entrypoints rewritten]

**Touched pages:** [[texas-42]] [[rules-of-42]] [[suit-algebra-spec]] [[play-phase-algebra]] [[engine]] [[engine-architecture]] [[layer-system]] [[multiplayer-pattern]] [[client-implementation]] [[engine-testing-patterns]] [[intermediate-ai]] [[forge]] [[expected-q-value]] [[the-oracle]] [[gus-qmean-router]] [[router-reality-check]] [[engine-adrs]] (+~30 more: hooks, citations, sha-stamps; waves 2ab1a825, d1f1633d, e2171816, 522779c5, b89ff635)
**Added:** the game-of-42 cluster (rules + algebra + play phase), the six-topic engine reference cluster, [[gus-qmean-router]] (the no-oracle router that works), [[engine-adrs]]; `sources/` gains pi-oracle-bidding {question,answer}, mccfr-exploration, and the book-second-pass reader reports (relocated from docs/)
**Updated:** [[router-reality-check]] corrected (replacement hurts, second opinion helps); [[ls-mixture]] mis-expansion fixed (always the arxiv short/long sense); forge foot-guns/folk-wisdom/training-data doctrine promoted into [[forge]] and [[expected-q-value]]; ~12 stale engine-doc claims corrected against current code while writing the cluster
**Retired:** docs/{adrs,archive,research,wiki-mine} and docs core+theory+rules files (rules-tournament.md unmigrated — erroneous), 36 burl/gus session docs, forge/eq/cpu_deprecated/ (no-legacy violation), SPIKE_REPORT.md, MORNING_DIGEST.md; CLAUDE.md/AGENTS.md/README.md rewritten wiki-first (beads → GitHub issues)

## [2026-07-11 | 4123b2d5 | review repairs — MPS sampler defect + prior-sweep completion + rebalance]

**Touched pages:** [[partnership-wall-research]] [[partnership-research-gates]] [[the-wall]] [[world-sampler-mrv-audit]] [[wiki-entrypoints]] [[burl]] [[forge]] [[sources/4123b2d5]]
**Added:** [[sources/4123b2d5]] — MPS int64-gather defect in the shipped sampler repair, fixed with per-device uniformity regressions.
**Updated:** prior sweep completed ([[w42-champion-selfplay-fixed-point]], [[lamir1-ceiling]], [[strategy-fusion]], [[past-belief-future-direction]], [[pi-opp-head]], Plunge/Splash); clairvoyance decomposition registered as gate 2; partnership reframed as one registered direction on [[the-wall]]; `~6.8 Q` resolved-question entry rewritten as a split; source-digest correction shrunk to a one-line qualifier.
**Frontier:** CUDA benchmark, exposure scan, two-block C0 reproduction, and the clairvoyance bound precede the causal runner.

## [2026-07-11 | c7f74f5c | measurement-ready frontier — infrastructure before path selection]

**Touched pages:** [[partnership-wall-research]] [[partnership-value]] [[partnership-research-gates]] [[the-wall]] [[wiki-entrypoints]] [[index]]
**Added:** [[sources/c7f74f5c]] — review correction and domain synthesis behind the cleaned PR frontier.
**Updated:** Q-mean is restored as bounded positive consumer evidence; natural policy legibility is separated from sparse intentional signaling; wall promotion is separated from the additional fixed-vs-shuffled partnership criterion.
**Frontier:** PR 39 delivers trustworthy measurement infrastructure and an evidence ledger; no next experiment, causal microgame, or successor architecture is selected.

## [2026-07-12 | 1a4482fe | convention-aware blueprint search preserved without selection]

**Touched pages:** [[convention-aware-blueprint-search]] [[partnership-wall-research]] [[partnership-value]] [[sources/1a4482fe]] [[index]]
**Added:** the SPARTA-style blueprint proposal and its separate project-record refinement; neither is promoted to a build.
**Questions opened:** can [[w42-book-second-pass|Winning 42]] conventions seed a shared codebook while a learned policy supplies the complete blueprint?
**Frontier:** the design is IDEATED, unbuilt, and unselected; clairvoyance remains a consumer-specific sensitivity probe rather than a universal bound.

## [2026-07-12 | a6590bf6 | book-seeded coordinated initialization promoted into research trail]

**Touched pages:** [[convention-aware-blueprint-search]] [[partnership-wall-research]] [[partnership-research-gates]] [[sources/a6590bf6]] [[index]]
**Updated:** Winning 42 becomes a visible candidate codebook overlay; convention value is a sender x partner-reader x opponent-reader interaction, not a double-dummy rejection label.
**Questions refined:** can the sparse overlay become a complete calibrated blueprint whose partner gain survives opponent decoding and full-match marks?
**Frontier:** the mechanism is preserved above leaf level but remains IDEATED, unbuilt, unmeasured, and unselected.

## [2026-07-12 | d5816915 | blueprint hypothesis framing rebalanced]

**Touched pages:** [[convention-aware-blueprint-search]] [[partnership-wall-research]] [[partnership-research-gates]] [[sources/d5816915]] [[index]]
**Updated:** the surviving structural case now leads; Winning 42 initialization, existing infrastructure, causal attribution, and four-seat inference receive the same weight as the engineering requirements.
**Corrected:** repeated status caveats no longer imply a negative result; no contrary experiment exists.
**Frontier:** blueprint search remains one candidate among several, with durable research-trail visibility and no editorial presumption against it.

## [2026-07-12 | f6b691da | belief-weighted Jud MCTS preserved and synthesized]

**Touched pages:** [[belief-weighted-jud-mcts]] [[jud]] [[partnership-wall-research]] [[partnership-research-gates]] [[sources/f6b691da]] [[index]]
**Added:** belief particles → information-set MCTS → blueprint policy → Jud realized-value leaf, grounded by JudSearch's `+2.28` gain and the JS2 worlds-sweep boundary.
**Separated:** J0-J4 attributes root belief, adaptive depth, information-set updates, and convention value; determinized and information-set MCTS have distinct promotion gates.
**Frontier:** the idea is a surviving search-consumer hypothesis; Zeb and LAMIR are relevant prior evidence but did not test this combination.

## [2026-07-13 | b28fb55a | continuation-frontier research ingested — lanes selected, MCTS backup semantics corrected]

**Touched pages:** [[research-lane-selection]] [[search-literature-transfer]] [[auction-decoder]] [[belief-weighted-jud-mcts]] [[jud]] [[convention-aware-blueprint-search]] [[the-wall]] [[partnership-wall-research]] [[sources/b28fb55a]] [[index]]
**Added:** [[research-lane-selection]] (the gates' step-3 experiment selection), [[search-literature-transfer]], [[auction-decoder]].
**Updated:** [[belief-weighted-jud-mcts]] backup semantics — two legal forms replace partner-max/opponent-min; actor-relative node identity; J3 gated on calibrated likelihoods. [[jud]] v2 per-move targets split into two consumer-distinct signals.
**Frontier:** Stage 0 closure (CUDA bench, exposure scan, two-block P0/C0) precedes lane grading; Lanes A/B primary.

## [2026-07-13 | research-night | Stage 0 closes; Lane A v0 validates; Lane B armed]

**Touched pages:** [[stage-0-closure]] [[world-sampler-mrv-audit]] [[partnership-wall-research]] [[champion]] [[auction-decoder]] [[auction-decoder-v0]] [[jud-target-granularity]] [[search-literature-transfer]] [[index]]
**Added:** [[stage-0-closure]] (all six arena arms in registered bands; CUDA correctness PASS + throughput-prediction MISS: sampler is launch-bound; exposure 2.51% distributional, 20/200-worst argmax flips), [[auction-decoder-v0]] (instrument validated, causal signature clean), [[jud-target-granularity]] (R1–R6 registered before evaluation).
**Updated:** audit + trail + champion pages forward-linked to the closure; exposure question moved to `questions/resolved.md`; literature citations verified against primary sources.
**Capability:** `--teacher-forced` E[Q] labeling (decision k = recorded play step k; 107,244/107,244 decision coordinates covered on the Lane B corpus).
