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
- **Partnership spine + consolidation** (2026-07-11, 6 archived entries) — the [[partnership-wall-research]] measurement spine (sampler audit, result vocabulary, review surface, four gates), the docs→wiki consolidation (game-of-42 + engine reference clusters), prior-sweep completion and review repairs. Hub: [[partnership-wall-research]]. Later entries remain below.
- **Continuation frontier + consolidation week** (2026-07-12 → 07-14, 10 archived entries) — blueprint-search/coordinated-init/belief-MCTS research preserved ([[convention-aware-blueprint-search]], [[belief-weighted-jud-mcts]]); the research night ([[stage-0-closure]], [[jud-target-granularity]]); wiki schema v2 + routing trails; the champion→[[jud]] name fold; [[count-fate-ledger]] formulated.

---

## [2026-07-18 | worktree-table42 | table42 kept and promoted — data to HF, code to tools/, champion reseated]

**Touched pages:** [[table42]] [[table42-game-night]] [[jud]] [[huggingface-assets]]
**Added:** game night 1's complete evidence mirrored to the HF evidence dataset at pinned tag [`table42-gamenight-1`](https://huggingface.co/datasets/jasonyandell/mk5-run-evidence/tree/table42-gamenight-1/table42) — host run dirs, v1 game logs, h2h rows (`jud_vs_otis`, `jud_vs_gus/n128`; `jud_vs_burl` never wrote CSVs, only its runner survives). Code promoted out of gitignored scratch to `tools/table42/` (host + wait/act seat drivers + probe66/probe66_grid); runs stay in `scratch/table42/run/` per [[run-artifacts-policy]].
**Fixed:** the jud seat now serves **the graded champion** (`margin:wp`(r8) + `lens:ev`) instead of the round-0 loop artifact — the #69 seat fix, serving rule enforced; the host announces trump declarations on the table channel (game night 1's misread-contagion bug); the final-trick winner display bug (the zeb engine already stores the winner in `trick_leader` at terminal).
**Retired:** the Cloudflare halves (v1 worker-authority build and v2's relay page) — humans sit `file:` seats via `wait.py`/`act.py`; no cloud dependency remains (#65 answered: promoted, not retired).
**Framed:** #66 resolution filed on [[jud]] — the 0.68 was the wrongly-seated round-0 head; what the table genuinely exposed is a **field-model error, not dishonesty** (passes priced as jud-shaped weakness). Walker gloss on [[table42]] corrected to [[walt]]'s definition (unbeatable when led).
**Questions opened:** [#80](https://github.com/jasonyandell/mk5-main/issues/80) — host resume (a died host mid-game is a lost game).

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

## [2026-07-16 | conversation | table42 game night 1 — jud seated, humans played, findings filed]

**Touched pages:** [[table42]] [[table42-game-night]] [[playing-from-the-inside]] [[index-topics|index]]
**Added:** `jud:` seat driver (ValueBidder+JudPlay, numbers-as-reasoning) to the v2 host; game night 1 findings on [[table42]] — belief contagion between language-users (trump misread propagated via chat, survived contradicting evidence), jud's field-fragile partner-pass semantics, count-consolidation as live jud-ism, the family vocabulary (rathouse luck, walker taxonomy, bid scale); traditions section on [[table42-game-night]] (shuffle-pause review, announced plans, verify-trump-from-the-view); [[playing-from-the-inside]] — the seat's first-person phenomenology, filed at Jason's insistence. Registered on #66: jud 0.68 vs Jason 0.20.
**Questions opened:** [issue #66](https://github.com/jasonyandell/mk5-main/issues/66) — replay jud's hand-3 auction seat: is P(make)=0.68 honest against a human-containing field?

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

## [2026-07-16 | worktree-table42 | table42: the online table built and live; game night in progress]

**Touched pages:** [[table42]] [[intermediate-ai]]
**Added:** [[table42]] — Cloudflare Worker + D1 around HeadlessRoom/replayActions with seat-token filtered views, seed scrub, access-log audit, gated reveal; browser UI for Jason, CLI seat for Claude, Opus-4.8 jeb partners via headless `claude -p`.
**Updated:** [[intermediate-ai]] — measured: headless bid decisions cost minutes (minimax-to-terminal rollouts per candidate bid), which is why table42's bots are model-brained.

## [2026-07-16 | worktree-table42 | table42 v2: local Python host is the authority; game night playbook]

**Touched pages:** [[table42]] [[table42-game-night]]
**Added:** [[table42-game-night]] — the one-stop playbook: vibe (feel the game, not competition), host commands, persistent agent-teammate seats with logged reasoning, honor system, review flow.
**Updated:** [[table42]] — v2 architecture: Python host on zeb engine + arena auction, Cloudflare demoted to dumb glass (hidden hands never leave the machine), v1 worker-authority retired same day; findings kept.
