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
- **jud + the wall + wiki overhaul** (2026-07-05 → 07-10, ~10 entries, mostly still below) — jud v0/v1 built and graded, [[the-wall]] stated precisely, then the wiki turned on itself: era archaeology backfill, the 162-page experiment audit.

---

## [2026-07-05 | local | jud engineering first cut — value-native endorsed, rank-vs-price mechanism]

**Touched pages:** [[jud]] [[belief-conditioned-self-play]] [[rank-vs-price]] [[pimc]] [[champion]] [[index]]

**Added:** [[rank-vs-price]] — the mechanism resolving the "somehow it was all about
bidding" fragment: PIMC's strategy-fusion optimism is a distribution-shape error that
cancels in play (argmax over siblings; rankings survive common-mode inflation) and
lands whole in bidding (tail mass read cardinally against pass/`race_wp`). One stroke
explains the play-side nulls (#25, #27) and the #26 over-bidder; measured legs:
`optimism_gap.json` (realized 0.52 @ 30 → ~0.19 @ 41 vs oracle 0.64 @ 30), the three
play nulls, and `net:wp`'s frozen-realized-calibration dominance.

**Updated:** [[jud]] — "The engineering, first cut" (Fable 5 session, 2026-07-05; new
session, no memory of `0a708a4e` claimed): value-native **endorsed** for the pricing
path, promoted from optional summit to spine. jud v0 = one added head (V_realized:
info-state → distribution over realized hand margin, categorical CE on realized
outcomes from the #26 arena bridge, MC targets, coverage via ε/forced-bid corpora),
bidder prices via tail mass at the hypothetical-auction root through `MarksToSeven`
(auction-side score-conditioning rides along), play stays `lens:ev`, `pmake_scale`
retires. Factorization law (learn the unknown / compute the exact), the referee
instrument (oracle EV − V_realized EV = price of hidden information), three registered
predictions (v0 ≥ `net:wp` parity; V calibration matches the realized curve; the
fixed point stops over-bidding), and the v1/v2 ladder (search leaves; opponents-in-
rollout signaling). [[belief-conditioned-self-play]] — the open "is value-native
Fable's intent" question split: historical intent stays open (likely permanently);
design question closed by the Fable 5 endorsement; training mechanics now first-cut.
[[pimc]] — rank-vs-price section. [[champion]] — self-consistency section points at
the first cut. Provenance line preserved: the value-native extension was the
2026-06-14 session's, and it was right.

**Evidence rescued:** `champion-one-organ-theory-2026-06-14.md` and
`handoff-2026-06-14-jud-vocab-and-fable-words.md` copied from gitignored scratch into
`champion/evidence/` (they back [[jud]]'s conclusions and the provenance chronicle).

**Questions opened:** none new; jud's open engineering narrows to the v1 search shape
and v2 opponent-model mechanics.

## [2026-07-06 | d678598 | arena perf pass — 2.38× games/sec on MPS, byte-identical]

**Touched pages:** [[arena]]

**Updated:** [[arena]] — dispatch/sync reduction on the oracle decision path
(numpy-assembled state tensors, vectorized order-preserving pool construction,
maskless `scatter_add` void aggregation, MRV loop ~45→~20 kernels/step with dead
per-step syncs removed, per-device table caches, memoized `current_player`).
Byte-identical to baseline on CPU and MPS was the correctness gate. Paired MPS
bench: 0.56 → 1.34 games/s (2.38×), reproduced across two A/B pairs; post-merge
production throughput ~1.34 games/s on a pooled 128-game A/B. Key finding: the
arena is dispatch-bound, not compute-bound — the oracle forward dominates CPU
wall (62%) but shrinks on MPS, where per-tick kernel-dispatch/sync overhead
(~1,500–2,000 launches, ~25–45 syncs) becomes the bottleneck. Full profile:
`docs/arena-perf-2026-07-06.md`. A second pass (constant-batch-width refill,
gate relaxed from byte-identity to distribution-level equivalence by user
decision) is in flight as of 2026-07-06.

**Questions opened:** none new.

## [2026-07-06 | 4080e07 | jud v0 built and graded — value-native bidder reaches net:wp parity]

**Touched pages:** [[w42-jud-v0]] [[jud]] [[champion]] [[rank-vs-price]] [[index]] [[log]]

**Added:** [[w42-jud-v0]] — the closing write-up for jud's first buildable slice
(Champion rung #32, GitHub #32). A value-native bidder prices contracts from a head
(`V_realized`/`champion/margin_net.py`) trained on **realized** 4-seat self-play
outcomes instead of the double-dummy oracle; the bidder (`champion/value_bidder.py`,
CLI `margin:wp`) reads tail mass at the hypothetical-auction root through
`MarksToSeven`; play stays `lens:ev`; `pmake_scale` retires. Graded against the three
registered predictions: **P2 calibration PASS** (ECE 0.046, max |Δ| ≤ 0.029 over 13
thresholds, 6× closer to realized than to oracle, sits 0.09–0.13 below double-dummy);
**P1 round-0 parity MISS** (−1.44 [−1.88, −0.95] — a legible over-bidder that wins
points (+5.66/hand) and loses marks, via a winner's-curse-on-*selection* channel plus a
notrump declaration-level artifact, both independent of double-dummy optimism); **P3
self-play loop PASS** (4 rounds carry the margin −1.44 → −0.31 → +0.24 → +0.24 → +0.22,
CI includes zero from round 2; made-rate 49.9% → 60–64%; notrump artifact dies in one
on-policy round, share 47.5% → 1.5%). Canonical same-seed check: −0.07 [−0.66, +0.49],
**statistical parity** with `net:wp` while winning +7 points/hand. Two methodological
findings: coverage anchoring beats single-variable recipe purity (the recipe fork —
dropping net:wp self-play chunks regressed round 1 to −2.18), and the `MarksToSeven`
pass baseline is a denial-bidding lever that makes over-bidding worse (the A2 sign-catch,
credited to the value-bidder subagent). Evidence at `4080e07`
(`champion/evidence/jud_v0/`). Definitive 512-game same-seed A/B: **−0.01/game
[−0.28, +0.25]**, 258/512 — dead parity (`ab_definitive_512_r4_summary.json`).

**Updated:** [[jud]] — honest status flipped from "not built" to "v0 built and graded";
predictions ledger graded in place (P2 pass / P1 miss→loop-recovered / P3 pass); v1
named as the parity-breaking frontier. [[champion]] — self-consistency section carries
the rung #32 outcome; champion's bidder stays `net:wp` for now (`margin:wp` its
value-native equal on marks, superior on legibility). [[rank-vs-price]] — leg 3
(the pricing mechanism's full test) confirmed at parity, with the winner's-curse-on-
selection rider the mechanism did not originally name.

**Frontier shift:** the value-native pricing path is validated — realized-outcome pricing
dissolves the #26 over-bidder without a tuned knob, and reaches the best hand-tuned
baseline. But it converges *at* parity, not past it (offense share plateaus ~60–67% vs
the predicted selective 50–55%). Whether the residual is the [[pimc]] price of hidden
information or further calibration headroom is v1's question — value at the leaves of
shallow belief-state search in play and defense.

**Questions opened:** what breaks the parity plateau (v1 search shape vs opponent
modeling); reading the referee gap (oracle EV − V_realized EV) as a live convergence
instrument rather than a static meter.

---

## [2026-07-06 | 68fda7b + 0bdd4d5 | the plateau probe: data starvation, not structure]

The jud v0→v1 bridge. [[w42-jud-v0]]'s open question 1 — was the parity plateau the
[[pimc]] price of hidden information (structural) or a data-starved tiny MLP? — run as a
registered prediction and answered.

**Touched pages:** [[w42-plateau-probe]] [[w42-jud-v0]] [[jud]] [[champion]] [[rank-vs-price]]

**Added:** [[w42-plateau-probe]] — the registered-prediction write-up (GitHub #33). The
structural reading was registered pre-run as a falsifiable prediction: scaling on-policy
data would NOT break parity. **Falsified.** Rounds 5–8 at 3× data/round (1000 self-play
games/round vs 300) carried `margin:wp` past `net:wp` — head_8 beats it **+0.38
[+0.09, +0.67]** (reserved seed 7000000, 512 games) and **+0.42 [+0.12, +0.72]** (fresh
seed 9000000), both 287/512 (56.1%), with round 7's 256-game A/B independently excluding
zero. **The first learned bidder to beat the hand-tuned champion on marks.** The plateau
was calibration headroom in a data-starved head, not the hidden-information price. A
registered extension (rounds 9–12) confirmed **saturation**: head_12 at +0.21 [−0.08, +0.47]
/ +0.37 [+0.09, +0.65], inside the registered [+0.2, +0.6] band, indistinguishable from
head_8 — the data-scaling curve flattens at **≈ +0.3–0.4 marks/game** at this net capacity.
The page carries the full r0–r12 round table (`champion/evidence/jud_v0/loop_metrics.json`).
The registered prior was wrong, recorded plainly — a falsified prediction run to its
falsifier is the system working.

**Updated:** [[w42-jud-v0]] — addendum + open question 1 marked ANSWERED (data starvation),
pointing to the probe. [[jud]] — honest status flipped from "reaches parity, does not yet
beat" to "past parity, saturating at ≈+0.3–0.4"; v1 re-described as one net for bid + play,
in build. [[champion]] — the bidder claim flipped: `margin:wp`(head_8) is the first learned
bidder to beat `net:wp`, best-measured bidder is `champion/margin_net_r8.pt`; saturation
noted. [[rank-vs-price]] — leg 3 upgraded from "validated at parity" to "validated and then
dominant"; the winner's-curse-on-selection channel is data-limited, not structural.

**Frontier shift:** the value-native pricing path no longer ties the best hand-tuned
baseline — it beats it. The binding constraint at v0's scale was on-policy data volume, not
the [[pimc]] price of hidden information; that reading is refuted at this scale. Data has
run its course at this net capacity, so the next constraint is capacity or mechanism —
jud v1's premise (one net, bid + play; play-history-conditioned V_realized with 1-ply
argmax-EV play replacing E[Q] n=10 at runtime; then the same self-play-loop method).

**Questions opened:** none new — the probe closed [[w42-jud-v0]]'s open question 1.

## [2026-07-06 | 0d82a97 | jud v1 built: one organ, two consumers (6860a75..0d82a97)]

Three commits building jud v1's machinery: play-decision snapshot emission, the unified
JudNet organ, the judplay consumer, and a graded round-0 head.

**Touched pages:** [[entities/jud]] [[sources/0d82a97]]

**Added:** 1 source digest.

**Frontier established:**
- One net now serves bid and play: info-state (own hand + canonical auction + play
  history) → 43-bin realized-points categorical; bid-time = empty-history play-time,
  byte-identical to margin_net's root encoding (tested train/serve both ways).
- The arena emits per-decision corpora compactly: `HandRecord.plays` (28 seat/domino
  pairs), every decision a prefix, offense and defense rows sharing the hand's Monte
  Carlo label. Works in sequential and fast-batching modes.
- Registry: `jud[:wp][,pass<q>][,model=]` bidder (ValueBidder unchanged) and
  `judplay[:model=]` play (greedy depth-1, argmax E[pts], defenders minimize, one
  forward per tick).
- Round 0 graded honestly (champion/evidence/jud_v1/): combined −6.09 vs
  net:wp+lens:ev, bidder −4.08, play −5.53 (defense the biggest channel);
  judplay beats random +1.80; value sharpens with depth (MAE 8.6 → 3.5).

**Questions opened:**
- Overfitting is the binding constraint (memorizes 200K rows in ~1 epoch at lr 1e-3);
  regularization/data scale for loop rounds.
- Root-row dilution (bid roots are 2/56 of samples; worst ECE slice) — up-weight or
  let the loop close it?
- Does the v1 self-play loop dissolve the round-0 over-bid (93% offense share) the
  way v0's did?

## [2026-07-06 | 3ac03de | jud v1 graded: one organ, bid and play (f550205..3ac03de)]

Four commits carrying jud v1 from build to a graded verdict, every rung registered on
GitHub #33 before its measurement: `f550205` (the organ — one net, play-history
snapshots, the `judplay` consumer), `9d30b25` (the loop grades JP1/JP2/JP3), `e596205`
(judsearch — belief-lift worlds, current-trick rollout, V_realized leaves), and
`3ac03de` (the search-ladder grades JS1/JS2/JS3 + the night verdict).

**Touched pages:** [[experiments/w42-jud-v1]] [[entities/jud]] [[entities/champion]] [[topics/rank-vs-price]]

**Added:** [[experiments/w42-jud-v1]] — the full arc as a registered-prediction ledger.

**Updated:**
- [[entities/jud]] — v1 promoted from "machinery built" to "built and graded"; the
  policy-conditional pricing law added to the vocabulary; v1/v2 ladder bullets and the
  honest status rewritten to the graded verdict.
- [[entities/champion]] — current best player stated: `margin:wp`(head_8)+`lens:ev`
  (+0.38/+0.42 over `net:wp+lens:ev`); rung #33 recorded as built-and-graded, not
  displacing the champion.
- [[topics/rank-vs-price]] — the play half measured: greedy value play is a bad ranker,
  search recovers most (not all) of the gap oracle-free, the oracle's rankings stay
  unbeaten.

**Frontier verdict:** the one-organ unification **holds at the auction and is
mechanism-limited at play**. The bidder survives the fold intact (beats v0's own round-0
bidder); greedy 1-ply value play is a bad move-ranker (the loop moves it zero, JP3
falsified); `judsearch` recovers two-thirds of the play gap oracle-free (−3.44 → −1.16,
JS1 PASS +2.28) but not parity, and neither more worlds (JS2 below band) nor a
better-calibrated head (JS3 falsified) closes the rest. The wall is per-move
discrimination — a 470k MLP on hand-level Monte-Carlo labels cannot out-rank E[Q] n=10's
per-move oracle. The stack went −4.37 → −1.43 oracle-free in one night; the current best
player is unchanged. v2's cue is concrete: a bigger leaf on per-move targets (E[Q]
distilled as bootstrap) plus opponents-in-rollout.

**Questions opened:** none new — v1's play wall is named, and v2's target follows from it.

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
experiments; raw reader reports preserved at `docs/research/book-second-pass-2026-07-07/`.
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
