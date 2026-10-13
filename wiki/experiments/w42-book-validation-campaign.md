---
title: w42 Book Validation Campaign
kind: experiment
first_seen: 2026-05-03
last_updated: 2026-10-12
status: superseded
---

**Superseded as a rollup.** The canonical synthesis of what this campaign found
is [[w42-book-claim-synthesis-and-ai-directions]]; the closure baseline it
started from is [[w42-phase4-final-claim-audit]]. This page remains the
wave-by-wave operational record. Which clock is this: `Wave 0–5` and bookval
`v1`/`v2`/`v3` are this 2026-05-03 campaign (v2/v3 are utility-lens passes, not
versions of the v1 corpus); `phase2/3/4` names the earlier 2026-05-02→03
claim-ledger sweep; the 2026-06+ [[champion-ladder]] rungs are a third,
unrelated clock.

## Mission

Take all 64 [[w42]] ledger rows from "evidence on a slice" to paired
counterfactuals on real auctions or injected late states. Book validation
is the priority. Model improvements are downstream.

The 64 rows cover at least 277 advice sentences in the book unevenly:
chapters 1, 6, 11, 13, 14 and 15 have zero rows, Ch12 has 2, and the book's
worked example hands were never used as test positions (#123). What the
campaign's evidence is worth (E[Q] network labels, points, bid 30, unclustered
CIs, a malformed corpus) is stated once in [[w42-book-claim-synthesis-and-ai-directions#what-the-evidence-is-worth]].

The campaign is structured as five waves of background subagents
orchestrated from the foreground. Each wave's outputs land under
`w42/book_validation_v1/wave<N>/<bead>_<slug>/` plus one wiki experiment
page per agent. Reconciliation is foreground-only: agents do not touch
the central ledger or the synthesis page.

## Status

**Dormant since Wave 5 (2026-06-13); superseded in research attention by
[[jud]] (no closure note was written at the time, no further
waves were planned).** This table was never updated after Wave 4.1 landed;
the Wave 5 row below was added by the era-6 audit (2026-07-06) from the
on-disk artifact, which existed but had never been entered.

| wave | scope | bead | status |
|---|---|---|---|
| 0 | baseline ledger snapshot + agent rules of engagement | t42-snwe | closed |
| 1.1 | distribution-lens action reranker | t42-ybo6 | closed |
| 1.2 | mark-utility transform pass | t42-c6sa | closed |
| 1.3 | hidden-threat impact ranker | t42-c2y9 | closed |
| 1.4 | cross-AI agreement matrix | t42-m2i7 | closed |
| 1.5 | independent ledger audit re-run | t42-1nmm | closed |
| 2.A | state-injection harness | t42-rwdj | closed (commit 19fc675) |
| 2.A.2 | oracle-greedy snapshot mining (5 corpora, 1822 snapshots) | t42-y8b5 | closed (commit 77a8511) |
| 2.A.3 | reentry probe v2 | t42-v9lu | closed (d9f8dbf) — null overall; post-hoc late slice negative; early game (the book's context) n=0 |
| 2.B | bid-aware E[Q] driver smoke | t42-6j3k | closed (0c802d4) |
| 2.B.2 | bid-aware full 50-seed MPS sweep | t42-7eop | closed (720aa83) — 259,618 rows, validation 10/10 PASS; 2 ledger promotions, one on a tautology, one on an aggregate proxy |
| 2.C | void-creation paired contrast (lead) | t42-26j8 | closed (8507b11) — verdict withdrawn pending setter sign check (#122) |
| 2.C.2 | void-creation paired contrast (follow) | t42-z31l | closed (37a5804) — verdict withdrawn pending setter sign check (#122) |
| 2.D | low-trump-trap paired contrast | t42-jysl | closed (29b86b8) — null in the count subgroup (the book's scenario) |
| 2.E | setter-pounce bid=30 | t42-ntbe | closed (79b5b7d) — null; withdrawn pending setter sign check (#122) |
| 2.E.2 | setter-pounce high-bid snapshot probe | t42-8kbh | closed (0b7fb01) — row demoted to `contradicted`; verdict withdrawn pending setter sign check (#122) |
| 2.G | ch02 multi-step bid-only-enough | t42-ey88 | closed — row promoted to `supported` on a test that cannot fail (tautology) |
| 2.H | ch10 mark-multiplier action-level | t42-8na4 | closed (no status change; the row is a scoring definition; the 71% flip rate is an artifact) |
| 3.0 | utility-lens meta-analysis | t42-f2ur | closed — four of its seven inputs are setter-side probes withdrawn pending #122 |
| 4.0 | utility-argmax divergence (architecture-decision gate) | t42-hmjr | closed — EV vs p_make argmax disagrees on 41.2% (CI [36.8%, 45.6%]) of ch05-follow snapshots against an ad hoc 5% gate; two random legal pickers disagree 70.4%; EV's argmax is the void tile on 29.4% |
| 4.1 | Lens v1 utility head-to-head (who wins games?) | t42-4ouu | closed — EV beats the others: round-robin {ev, p_make, cvar_10, robust_q25} × 6 pairings × 1000 hands paired-seed, bid forced to 30; in points all 6 CIs exclude zero (ev +5.42 pts/hand over p_make); in marks only EV's edge survives (+0.110 marks/hand over p_make), the other three pairings are null; Wave 4.0 reading inverted (EV's "third-option" picks are point-winning, not noise); production `select_actions` is Lens(p_make), the lowest-scoring utility in points; switching to ev-argmax is a one-line follow-up |
| 5 | champion teaching battery (book-claim-checking against champion trajectories) | — | closed (`w42/book_validation_v1/wave5/champion_teaching_battery/`, 2026-06-13, [[w42-champion-teaching-battery]]) — the campaign's only Wave 5 probe; runs the champion's lens:ev self-play trajectories through the ch04/ch05 detectors rather than through a planning-aware architecture (MCTS/Lookahead-Lens/book-strategy-player, all unbuilt); backfilled into this table by the era-6 audit (2026-07-06) — the artifact existed on disk but was never entered here |
| 2.F | 84-throwaway snapshot corpus + ch08 probe | t42-wikw | never produced output |

The campaign epic is bead `t42-4zi6`. Detector hygiene follow-ups (filed
during Wave 1.4 reconciliation): `t42-v0m5`, `t42-2yb5`, `t42-btpg`.

## Ledger movement to date

Wave 1 absorbed 2 promotions (Ch 10 timed-marks-advancement-objective and
point-system-skill-signal: both into `context-limited` from
`not-yet-tested` and `underpowered` respectively) and normalized one
non-vocabulary status string in a phase-4 worker artifact.

Wave 2 absorbed 2 net promotions plus 1 promotion-then-demotion:

- `ch02-bid-only-enough`: `not-yet-tested` → `context-limited`
  (Wave 2.B.2) → `supported` (Wave 2.G; 5 step pairs, 85/85 slice
  cells in book direction). The test cannot fail: every bid is scored
  from the same points sample against a rising threshold
  (`forge/bidding/estimator.py:72-100`), so P(make) can only fall as the
  bid rises. The book's auction claim was never tested.
- `ch12-setter-pounce-high-bid-off`: `underpowered` → `context-limited`
  (Wave 2.B.2 aggregate proxy) → `contradicted` (Wave 2.E.2 two-action
  snapshot probe). The probe negates E[Q] for the setter as if it were
  Team 0's view; forge documents it from the acting player's view. The
  verdict is withdrawn pending a sign check (#122).

The Wave 2.E.2 demotion prompted a "promotion guard" rule in
`w42/book_validation_v1/AGENTS.md` (aggregate proxies do not qualify for
promotion). The rule stands on its own logic; its precedent is withdrawn
with the probe.

Status counts after Wave 2.E.2 (as recorded; 20 of the 24 `supported` rows
are arithmetic, rule fixtures or scoring definitions, and no strategic claim
was established; see [[w42-book-claim-synthesis-and-ai-directions#what-the-evidence-is-worth]]):

| status | count |
|---|---:|
| supported | 24 |
| context-limited | 14 |
| underpowered | 19 |
| not-yet-tested | 4 |
| contradicted | 3 |

The campaign's design discipline says we move a row out of
`context-limited` / `underpowered` / `not-yet-tested` only when paired
counterfactual evidence supports it on a named slice with reproducible N
and CI. Wave 1 did not generate that evidence; Wave 1's role was to
extract every signal already in existing artifacts. Wave 2 builds the
infra needed to actually move rows.

## Wave 1 deliverables

Five offline analyses, all over already-existing per-world Q tensors and
detector outputs:

- [[w42-bookval-v1-wave1-distribution-lens-reranker]] — utility-family
  action ranking on one deal; CVaR_10 / robust_q25 agree with EV at 74.4% /
  70.1% (the synthesis quoted 82-83%), p_make at 61.1%, with forced moves
  counted and no chance baseline; EV's argmax never loses in EV terms by
  definition.
- [[w42-bookval-v1-wave1-mark-utility-transform]] — Ch 10 mark utility
  applied to existing branch atlas; 4.3% (12/280) genuine flips on one
  bid=30 deal; `mark_ev` is affine in `p_make` at one-mark multiplier by
  construction; bid-aware
  generator confirmed as a hard prerequisite for further Ch 10 work.
- [[w42-bookval-v1-wave1-hidden-threat-impact-ranker]] — per-decision
  top-K load-bearing tile attribution; the "100% directionally helpful"
  trump-count tiles are 20 partner-held cases from 3 hands; 5-5-in-twos and
  4-4-in-NT identified as load-bearing pseudo-trump in low-pip declarations;
  setter-seat asymmetry, which the book itself states (p.51).
- [[w42-bookval-v1-wave1-cross-ai-agreement]] — EV / Gus / detector /
  dist-lens agreement matrix; detector–EV agreement (43.7%) is at the
  random-legal rate (40.1% on detector-active decisions); setter seats dominate divisive decisions
  (83/100); detector hygiene findings on three Ch 5 / Ch 3 detectors.
- [[w42-bookval-v1-wave1-independent-audit]] — independent rebuild of the
  64-row audit; baseline holds; two Ch 10 ledger absorptions identified.

Synthesis: [[w42-book-claim-synthesis-and-ai-directions]].

## Wave 2 design

[[w42-bookval-v1-wave2-infra-design]] records the build plan. Two infra
pieces:

- **State-Injection Harness** — `GameStateTensor.from_snapshot` plus a
  pipeline driver that lets the forge generator start from arbitrary
  mid-game snapshots. Unblocks reentry preservation, void creation,
  low-trump trap, 35/36 pounce, and 84 throwaway-ladder claims.
- **Bid-Aware E[Q] Driver** — W42-side wrapper that runs the existing
  forge generator across a bid sweep on the same seeds. Forge already
  supports `--bid-values`; the W42 consumer side does not yet exploit it.
  Unblocks Ch 02 bid-only-enough paired tests, Ch 10 mark-multiplier
  divergence (the Wave 1.2 algebraic-identity finding), and Ch 12
  high-bid setter pounce.

Both build agents are running in isolated worktrees and will be reviewed
before merge.

## Operating discipline

- Every agent output cites slice + N + paired/unpaired + claim-ledger
  impact + status. See `w42/book_validation_v1/AGENTS.md`.
- Reconciliation is foreground-only. Agents flag findings; the
  orchestrator decides whether to absorb.
- Detector hygiene findings are filed as their own beads, not folded into
  the data wave that surfaced them. Detectors get refined separately so
  later waves can choose their own label snapshot.
- Chapter pages absorb wave findings in place. Source-backed claims are
  sharpened, not retired.
- Push at session end. Work is not done until pushed.

## Wave 3.0 reconciliation

[[w42-bookval-v2-utility-lens-synthesis]] re-processed all 7 closed
Wave 2 probes through 5 utility lenses. Four of the seven are setter-side
probes withdrawn pending the sign check (#122), so its reading of the
p_make/EV thread, including "high-bid pounce contradicted under all 4
utilities", is withdrawn with them.

- **Schema decision: ADOPT-DEFERRED.** Per-utility status columns were
  never adopted.
- **Status counts unchanged**: supported 24, context-limited 14,
  underpowered 19, not-yet-tested 4, contradicted 3.

## Wave 4.1 — Lens v1 head-to-head (who wins games?)

[[w42-lens-v1-utility-head-to-head]] is the cheap rung-1.5 player —
1-step Q-greedy with utility as a parameter, every hand forced to bid 30.

- In points, all 6 CIs exclude zero and EV beats p_make by **+5.42
  pts/hand** ([+4.03, +6.81]). In marks, EV beats each other utility
  (+0.110 marks/hand over p_make [0.048, 0.168]); the other three pairings
  are null, so there is no total ordering.
- EV's third-option picks score points rather than being noise.
- Sample-sweep at N ∈ {10, 50, 100} preserves the ranking.
- **Production-code follow-up:** `forge.eq.generate.actions.select_actions`
  hardcodes p_make-argmax-with-EV-tiebreak (effectively Lens(p_make)).
  Switching to ev-argmax is a one-line change; filed as `t42-10yj`, now
  #45 (open); not applied.
- **`disaster` follow-up exploration**: a clipped EV that floors all
  sub-threshold samples to Q=−42. Beats p_make by +2.74; loses to ev
  by ~1.5 (CI grazes zero); ties robust_q25.

## Methodology insight — strategy fusion

The campaign concluded that most book claims are multi-step plans, that
single-decision EV cannot see a plan, and that planning-aware probes were
needed. The diagnosis is wrong: a perfect-information evaluator plans
perfectly within each sampled world. Its defect is information-set
inconsistency, one plan per world averaged across worlds the player cannot
tell apart: strategy fusion ([[the-wall]]). Planning-aware probes over the
same oracle inherit it; the remedy was an imperfect-information evaluator,
later used in [[book-study-2026-10-12]]. Detail in
[[w42-book-claim-synthesis-and-ai-directions#methodology-insight-strategy-fusion-not-a-single-decision-blind-spot]].

None of the three proposed planning architectures (MCTS over forge,
Lookahead-Lens, book-strategy player) was built. [[book-strategy-player]]
was designed (2026-05-03/04) but never implemented — superseded before build
by the auction-first [[jud]] redirection ([[champion-design-review]],
2026-06-09). The Wave 5 probe that actually ran, [[w42-champion-teaching-battery]]
(2026-06-13), checks book claims against the champion's own realized
self-play trajectories, scored by the same E[Q] the champion plays by.

## Wave 4.0 — utility-argmax divergence

[[w42-bookval-v3-utility-argmax-divergence]] computed argmax-under-
utility for ALL legal actions on the 500 ch05-follow snapshots.

- EV vs p_make argmax disagrees on 41.2% (CI [36.8%, 45.6%]), above an ad
  hoc 5% gate; two random legal pickers disagree 70.4%, and 44% of
  snapshots have top p_make = 1.0 broken by slot index.
- EV's argmax is the void tile on 29.4% of snapshots; p_make 38.6%;
  CVaR_10 42.6%. The book's follow-void advice is not what the EV oracle
  plays.
- p_make and mark_ev agree on all 500 snapshots, by construction at bid=30.
- **Recommendation:** scope rung-2 utility-tunable searcher; never built.

## Links

[[w42]] | [[w42-book-claim-synthesis-and-ai-directions]] |
[[w42-bookval-v2-utility-lens-synthesis]] |
[[w42-bookval-v1-wave2-infra-design]] |
[[w42-phase4-final-claim-audit]] |
[[w42-phase4-claim-completion-board]] |
[[w42-bookval-v1-wave1-distribution-lens-reranker]] |
[[w42-bookval-v1-wave1-mark-utility-transform]] |
[[w42-bookval-v1-wave1-hidden-threat-impact-ranker]] |
[[w42-bookval-v1-wave1-cross-ai-agreement]] |
[[w42-bookval-v1-wave1-independent-audit]] |
[[the-wall]] | [[book-study-2026-10-12]]
