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
- **jud + the wall** (2026-07-05 → 07-06, 6 archived entries) — jud v0/v1 built and graded ([[jud]], [[champion]], [[w42-jud-v1]], [[w42-plateau-probe]]), the 2.38× arena perf pass, value-native endorsed via [[rank-vs-price]]. Later 07-06→07-11 entries remain below.

---

## [2026-07-06 | working-tree | the wall, stated precisely — distill-for-what + candlewax concordance]

**Touched pages:** [[candlewax]] [[jud]] [[w42-lens-v1-utility-head-to-head]]

**Updated:** [[candlewax]] — two new sections. (1) *Concordance*: candlewax ≡ bimodal/
multimodal PDF ≡ jud's "melted blob" ≡ mixed-mode geometry, with the founding-era
provenance (report/11, 2026-01-06: −42→+40 swings, 11% stable hands, 53% within-hand
variance; the 85-bin discs rendered 2026-01-24 — ~3.5 months before the name). (2) *The
wall, stated precisely* — Jason, verbatim: "I saw eq, I said sure I could distill it.
but for what purpose? no idea what to do with distilled melted candlewax." Distillation
was never the wall; CONSUMPTION is. Every era is a successive consumer hypothesis
(LLM-as-reasoner → tool surface → Lens utilities → rank-vs-price → jud), and any
"distill X" proposal must first name the consumer and the licensed collapse.
[[jud]] — the blob entry now names the identity with [[candlewax]]. [[w42-lens-v1]] —
the EV-wins verdict now carries the aggregate-vs-per-decision reconciliation note
against [[candlewax]].

**Context:** first surgical edits from the wiki-overhaul mining (5 scout reports in
scratch/wiki-mine/); the full charter (trails, founding-era backfill, page zero) is
pending approval.

**Questions opened:** none new.

## [2026-07-06 | afd4802 | wiki overhaul: archaeology backfill (eras 1-5) + full staleness audit (era 6)]

The seven-month archaeology, landed. Two moves at once: (1) **backfill** — ~30 new pages
reconstructing the project's pre-wiki history, one page per era-question, every date traced
to a git/bead/conversation timestamp and every named thing classified BUILT / IDEATED /
RENAMED; (2) **audit** — a full pass over the existing wiki that reconciled ~173 pages
against the repo, flipping stale `status: active` claims to their true frontier state and
making every correction reachable from the page that carries the stale claim. New front
door: [[the-wall]] → [[the-wall-biography]] → [[consumption-ledger]]. New anti-rot rules
codified in `AGENTS.md` (status-must-be-falsifiable, corrections-reachable-from-error,
questions-not-goals, names-doctrine, dates-trace-to-artifacts, privacy-firewall,
worth-a-bead-is-not-a-resting-state).

**Touched pages (hubs + spine):** [[the-wall]] [[the-wall-biography]] [[consumption-ledger]]
[[web-game]] [[the-oracle]] [[breakthrough-and-oracle]] [[eq-genesis]] [[strategy-fusion]]
[[argmax-q-ceiling]] [[alphazero-under-imperfect-information]] [[the-gestation]]
[[ideated-not-built]] [[zeb]] [[zeb-fleet-ops]] [[lem]] [[burl]] [[gus]] [[forge]]
[[champion]] [[jud]] [[index]]

**Added (~30 new pages):**
- Era backfill topics: [[pre-ml-ai-attempts]], [[multiplayer-lineage]], [[the-book-enters]],
  [[breakthrough-and-oracle]], [[the-oracle]], [[the-analysis-epic]], [[suit-algebra]],
  [[eq-genesis]], [[strategy-fusion]], [[alphazero-under-imperfect-information]],
  [[argmax-q-ceiling]], [[belief-feeding-policy]], [[eval-matrix-bradley-terry]],
  [[the-gestation]], [[ideated-not-built]], plus the front-door pair [[the-wall]] +
  [[consumption-ledger]].
- Entities: [[web-game]] (era-1 founding substrate), [[zeb-fleet-ops]].
- Experiments: [[full-teacher-eq-experiment]], [[gus-drama-atlas]].
- Decisions: [[qval-over-policy-models]], [[vs-random-eval-is-suspect]], [[grok-not-converge]].
- Source digests: [[sources/claude/era1-web-game-prologue]], [[sources/claude/era2-breakthrough-oracle]],
  [[sources/claude/era3-eq-era]], [[sources/claude/era4-zeb-era]], [[sources/claude/era5-gestation]]
  (privacy-curated claude.ai user turns; Texas-42 content only).
- Trail: [[the-wall-biography]] (seven-month capstone).
- Index: 7 previously-unindexed w42 strategy-tag experiments folded into the catalog.

**Updated:** ~173 pages status-reconciled per the era-6 audit; `index.md` catalog synced
(111 status-suffix corrections + 37 new/backfilled entries + a new "Era backfill" Topics
subsection + a claude.ai-digest Sources subsection + the-wall front-door entry);
`questions/open.md` extended (4 items, below).

**Retired / superseded (frontier flips, not deletions):** LEM and its Stage-0 adapter
chain → `complete`/`superseded`; [[burl]] and its lab/chat/microscope/wax-museum surface →
`superseded`; [[zeb]] → `superseded` (belief work carried by Gus/jud); the Winning-42
per-chapter book cluster and its phase-2/3/4 probes → `complete`/`superseded` as the
campaign closed; the LAMIR/router/blunder-detector no-oracle branch → `retired`/`superseded`;
[[candlewax-spike]] and [[selfplay-arena]] → `retired`. The live frontier
([[gus]], [[w42]], [[forge]], [[champion]], [[jud]], [[engine]]) stays `active`.

**Questions opened (raising `afd4802`):**
- `WorldSamplerMRV` sampler bias (~6.8 Q-pts vs enumeration at trick 6) — parked "worth a
  bead," never filed; **needs a GitHub issue** (anti-rot rule 7).
- The never-applied Lens ev-argmax switch — production `select_actions` is still Lens(p_make),
  the worst of four utilities; the one-line ev-argmax fix is open two months on.
- jud v2 cue — does a bigger leaf on per-move targets + opponents-in-rollout close the play
  gap the v1 hand-level MLP could not?
- The era-5 gestation's IDEATED generation — which unbuilt designs are worth resurrecting?

**Lint:** dead backlinks in the new era pages fixed (`[[the-engine]]`→`[[engine]]`,
`[[era-1-web-game-prologue]]`→`[[web-game]]`, `[[era-2-breakthrough-oracle]]`→[[breakthrough-and-oracle]],
`[[the-wall-stated-precisely]]`/`[[distill-for-what]]`→[[the-wall]]/[[candlewax]], phantom
`[[layer-system]]`/`[[mccfr-excursion]]`/`[[walker]]` de-linked or redirected); two private-memory
filename links purged from [[burl-chat-spike]] and [[post-commit-q-and-a]] (privacy firewall);
qualified body links in the new pages converted to bare. No orphans (every new page has an
inbound content link). Flagged-not-fixed: dead `[[plunge]]`, `[[burl-perf-phase1]]`,
`[[topics/spec-decode-acceptance]]`, `[[explore-game-cache-bug]]` in pre-existing modified
pages (missing-page candidates), and the two distinct "~74%" ceilings (era-3 argmax-vs-oracle
tie-structure vs era-4 Zeb vs-random capacity) that no page cross-claims as identical.

## [2026-07-07 | 5e3f3245 | book second pass: what the first extraction missed]

**Touched pages:** [[w42-book-second-pass]] [[w42-book-validation]] [[w42-bookval-v1-wave2-pounce-high-bid]]
**Added:** [[w42-book-second-pass]] — four parallel readers re-read the full OCR text with the
finished campaign as lens. The first pass extracted the book's tactics and missed its
information theory: the auction decoder (ch 6/12 bid→hand posteriors, the {30,31,35,36} bid
lattice, who-bid asymmetry, match-score-conditioned bidding), the action-choice inference
catalog, the signaling conventions (top-unplayed-trump as protocol, donate-highest code,
dump-to-inform, Plunge as legal one-bit signal), reputation-driven overbidding, the
quantified-prior calibration table, and multi-step plans with author-supplied win rates
(strip-the-protector p.92/94, double-ahead-of-off 53/60/33). Nine ranked follow-up
experiments; raw reader reports preserved at `wiki/sources/book-second-pass-2026-07-07/`.
**Updated:** [[w42-book-validation]] trail (frontier section routes to the second pass);
[[w42-bookval-v1-wave2-pounce-high-bid]] gains caveat 0 — the `contradicted` verdict is a
probable information-regime category error (book's clause is an imperfect-information hedge,
"regardless of whether you know who will win the trick," tested under a perfect-information
oracle).
**Questions opened:** does the pounce contradiction dissolve under a belief/PIMC defender
(bid ≥ 35, bidder ≤ 2 offs)? Do the book's bid→hand posteriors hold empirically, and does
head_8 respect the {30,31,35,36} bid lattice? OCR re-scan needed: book pages 181–182 and
185–186 absent, ch 16 four-trump table truncated, worked-hand diagrams are images.
Filed in `questions/open.md`.
**Curation (2026-07-10, PR #35 landing):** provenance pinned to `5e3f3245`; the
"named untested gaps" claim on [[w42-book-second-pass]] §1 re-attributed from jud's
page to the campaign synthesis ([[w42-book-claim-synthesis-and-ai-directions]]),
where the list actually lives; [[jud]] Links section now routes to the second pass
as the book-sourced experiment queue for the auction-first frontier and v2
opponents-in-rollout; opened questions filed into `questions/open.md`.

## [2026-07-07 | working-tree | experiment-page audit: 162 pages validated against primary artifacts, 43 corrected in place]

**Touched pages:** all 162 `experiments/` pages audited via two-pass fan-out (162 auditors, then adversarial re-review of every corrected page); 43 corrected in place, no audit residue left on pages.
**Updated (highest-weight):** [[gus-lamir1-piopp]] (Bug-6 outcome was inverted — the world_assign fix made regret *worse* 2.268→2.350; root cause is scalar V/Q distillation noise flipping argmax, not Q_head depletion-OOD; pivot options replaced with the real four from MORNING4_STATUS @ b42669a), [[gus-q-head-augmentation]] (conclusion rewritten to the sourced diagnosis), [[batch-throughput-bench]] (baseline is 83 tok/s not 43; prompt-cache reuse *was* tested — negative, see [[burl-perf-phase2]]), [[burl-perf-phase0]] (K1 flip was gi=36, not gi=72), [[gus-belief-co-train]] (2×2 regret table disentangled; §20-vs-§21 source discrepancy noted in place), [[gus-belief-calibration-diagnostic]] (receipt quoted verbatim), [[iter5-e1-rank-sweep]] (2048-vs-1024 truncation-ceiling source conflict noted in place), [[iter3-rules-adapter]] (fix landed 17 commits after, not three).
**Trail:** per-page audit ledgers ("page said X; artifact says Y, evidence path") live in git history at `45a7e358` / `29641ded`; live questions surfaced by the audit were already tracked in `questions/open.md` or on their pages.
**Lint:** zero new dead links; pre-existing dead `[[burl-perf-phase1]]`, `[[topics/spec-decode-acceptance]]`, `[[log]]` (burl-perf-phase3), `[[sources/<sha>]]` placeholder (burl-star-run3) remain flagged from the era-6 audit.

## [2026-07-10 | working-tree | log rotated: changelog-not-chronicle rule, digest + archive]

**Touched pages:** [[log]] [[log-archive]] `AGENTS.md`
**Added:** [[log-archive]] — entries 1–145 (2026-04-09 → 06-14) moved verbatim; log.md keeps a phase digest + last ~10 entries.
**Updated:** AGENTS.md log section — entry budget (~5 pointer lines), mechanical rotation trigger (>15 entries), no claim without a link.

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
