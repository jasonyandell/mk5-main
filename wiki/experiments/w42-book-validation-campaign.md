---
title: w42 Book Validation Campaign
kind: experiment
first_seen: 2026-05-03
last_updated: 2026-07-06
status: superseded
---

## Mission

Take all 64 [[w42]] ledger rows from "evidence on a slice" to paired
counterfactuals on real auctions or injected late states. Book validation
is the priority. Model improvements are downstream.

The campaign is structured as five waves of background subagents
orchestrated from the foreground. Each wave's outputs land under
`w42/book_validation_v1/wave<N>/<bead>_<slug>/` plus one wiki experiment
page per agent. Reconciliation is foreground-only: agents do not touch
the central ledger or the synthesis page.

## Status

**Dormant since Wave 5 (2026-05-03); superseded in research attention by
[[champion]]/[[jud]] (no closure note was written at the time, no further
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
| 2.A.3 | reentry probe v2 | t42-v9lu | closed (d9f8dbf) — `context-limited`, late game contradicts book |
| 2.B | bid-aware E[Q] driver smoke | t42-6j3k | closed (0c802d4) |
| 2.B.2 | bid-aware full 50-seed MPS sweep | t42-7eop | closed (720aa83) — 259,618 rows, validation 10/10 PASS, **2 ledger promotions** |
| 2.C | void-creation paired contrast (lead) | t42-26j8 | closed (8507b11) — **contradicted** (lead-to-self-void) |
| 2.C.2 | void-creation paired contrast (follow) | t42-z31l | closed (37a5804) — **context-limited** in book direction (follow position) |
| 2.D | low-trump-trap paired contrast | t42-jysl | closed (29b86b8) — `context-limited`, count subgroup symmetric |
| 2.E | setter-pounce bid=30 | t42-ntbe | closed (79b5b7d) — `context-limited`, p_make/EV split |
| 2.E.2 | setter-pounce high-bid snapshot probe | t42-8kbh | closed (0b7fb01) — **`ch12-setter-pounce-high-bid-off` DEMOTED to `contradicted`** |
| 2.G | ch02 multi-step bid-only-enough | t42-ey88 | closed — **`ch02-bid-only-enough` promoted to `supported`** (campaign's first) |
| 2.H | ch10 mark-multiplier action-level | t42-8na4 | closed (no status change; ch10 row already `supported`) |
| 3.0 | utility-lens meta-analysis | t42-f2ur | closed — narrows p_make/EV thread; ch05-void-creation-follow is the only true objective-dependent split; high-bid pounce contradicted under all 4 utilities |
| 4.0 | utility-argmax divergence (architecture-decision gate) | t42-hmjr | closed — **gate TRIPPED**; EV vs p_make argmax disagrees on 41.2% (CI [36.8%, 45.6%]) of ch05-follow snapshots; Wave 3.0 framing inverted (p_make picks void MORE than EV; EV is the outlier preferring third-option discards); rung-2 utility-tunable searcher justified |
| 4.1 | Lens v1 utility head-to-head (who wins games?) | t42-4ouu | closed — **EV wins decisively**: round-robin {ev, p_make, cvar_10, robust_q25} × 6 pairings × 1000 hands paired-seed; total ordering ev > robust_q25 ≳ cvar_10 > p_make, all 6 CIs exclude zero, ev beats p_make by +5.42 pts/hand; Wave 4.0 reading inverted (EV's "third-option" picks are point-winning, not noise); production `select_actions` is Lens(p_make) — the worst utility — switching to ev-argmax is a one-line follow-up |
| 5 | champion teaching battery (book-claim-checking against champion trajectories) | — | closed (`w42/book_validation_v1/wave5/champion_teaching_battery/`, 2026-06-13, [[w42-champion-teaching-battery]]) — the campaign's only Wave 5 probe; runs the champion's lens:ev self-play trajectories through the ch04/ch05 detectors rather than through a planning-aware architecture (MCTS/Lookahead-Lens/book-strategy-player, all unbuilt); backfilled into this table by the era-6 audit (2026-07-06) — the artifact existed on disk but was never entered here |
| 2.C | void-creation snapshot corpus + ch05 probe | t42-26j8 | blocked on 2.A |
| 2.D | low-trump-trap snapshot corpus + ch04 probe | t42-jysl | blocked on 2.A |
| 2.E | pounce-window-high-bid snapshot corpus + ch12 probe | t42-ntbe | blocked on 2.A + 2.B |
| 2.F | 84-throwaway snapshot corpus + ch08 probe | t42-wikw | blocked on 2.A + 2.B |
| 2.G | ch02 bid-only-enough paired bids | t42-ey88 | blocked on 2.B |
| 2.H | ch10 mark-multiplier paired bids | t42-8na4 | blocked on 2.B |

The campaign epic is bead `t42-4zi6`. Detector hygiene follow-ups (filed
during Wave 1.4 reconciliation): `t42-v0m5`, `t42-2yb5`, `t42-btpg`.

## Ledger movement to date

Wave 1 absorbed 2 promotions (Ch 10 timed-marks-advancement-objective and
point-system-skill-signal: both into `context-limited` from
`not-yet-tested` and `underpowered` respectively) and normalized one
non-vocabulary status string in a phase-4 worker artifact.

Wave 2 absorbed 2 net promotions plus 1 promotion-then-demotion:

- `ch02-bid-only-enough`: `not-yet-tested` → `context-limited`
  (Wave 2.B.2) → **`supported`** (Wave 2.G; 5 step pairs, 85/85 slice
  cells in book direction)
- `ch12-setter-pounce-high-bid-off`: `underpowered` → `context-limited`
  (Wave 2.B.2 aggregate proxy) → **`contradicted`** (Wave 2.E.2
  snapshot-level paired probe reversed it)

The Wave 2.G promotion is the campaign's first crossing into
`supported` from a non-supported start. The Wave 2.E.2 demotion is
the first time a Wave-2 promotion was reversed by stronger evidence —
it triggered a new "promotion guard" rule in `AGENTS.md`: aggregate
proxies do not qualify for promotion, only paired same-decision
contrasts on the relevant action shape.

Status counts after Wave 2.E.2:

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
  action ranking; CVaR_10 / robust_q25 agree with EV at 82-83% on the
  branch atlas, while p_make / threshold_mass only at 59%; EV "lies" in
  64.9% of decisions but never beats alternatives in EV terms.
- [[w42-bookval-v1-wave1-mark-utility-transform]] — Ch 10 mark utility
  applied to existing branch atlas; 3.6% genuine flips on bid=30 corpus;
  algebraic identity `mark_ev == p_make` at one-mark multiplier; bid-aware
  generator confirmed as a hard prerequisite for further Ch 10 work.
- [[w42-bookval-v1-wave1-hidden-threat-impact-ranker]] — per-decision
  top-K load-bearing tile attribution; trump-count tiles are 100%
  directionally helpful; 5-5-in-twos and 4-4-in-NT identified as
  load-bearing pseudo-trump in low-pip declarations; setter seats are
  asymmetric in load-bearing-tile category.
- [[w42-bookval-v1-wave1-cross-ai-agreement]] — EV / Gus / detector /
  dist-lens agreement matrix; setter seats dominate divisive decisions
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

## Wave 3.0 reconciliation (revised by Wave 4.0)

[[w42-bookval-v2-utility-lens-synthesis]] re-processed all 7 closed
Wave 2 probes through 5 utility lenses. Initial read narrowed the
emergent p_make/EV thread to one claim. **Wave 4.0 then broadened it.**

- **High-bid pounce hypothesis superseded** (still holds): ch12-setter-
  pounce-high-bid is contradicted under all 4 available utilities. The
  Wave 2.E "p_make split" framing was an EV-only-reporting artefact.
- **Schema decision: ADOPT-DEFERRED.** Per-utility status columns are
  the right shape, but populating them needs probes to record all 5
  utilities. Defer adoption until next 3-5 probes record full coverage.
- **Wave 3.0's "narrowing" was an artefact of paired-contrast statistics**
  (magnitudes on specific action pairs, not what each utility's argmax
  actually picks). Wave 4.0 measured argmax directly and found EV
  disagrees with p_make / mark_ev / CVaR_10 on 41-44% of ch05-follow
  snapshots. The multi-objective story is real, just not in the
  direction Wave 3.0 framed it.
- **Status counts unchanged** (both waves are read-only meta/measurement,
  no new probe verdicts): supported 24, context-limited 14,
  underpowered 19, not-yet-tested 4, contradicted 3.

## Wave 4.1 — Lens v1 head-to-head (who wins games?)

[[w42-lens-v1-utility-head-to-head]] is the cheap rung-1.5 player —
1-step Q-greedy with utility as a parameter. Round-robin in 7 minutes
wall.

- **Total ordering: ev > robust_q25 ≳ cvar_10 > p_make**, all 6 CIs
  exclude zero. EV beats p_make by **+5.42 pts/hand** ([+4.03, +6.81]).
- Wave 4.0's "EV is the outlier" framing is inverted: EV's third-
  option picks are *point-winning*, not noise. The book's void-
  creation advice aligns with the *worst-scoring* utility (p_make) on
  this corpus.
- Sample-sweep at N ∈ {10, 50, 100} confirms ranking is robust to N;
  N=10 is the right operating point (per Zeb's prior finding).
- **Production-code follow-up:** `forge.eq.generate.actions.select_actions`
  hardcodes p_make-argmax-with-EV-tiebreak (effectively Lens(p_make)).
  Switching to ev-argmax is a one-line change predicted to lift the
  E[Q]-vs-Zeb-Large win rate. Filed as `t42-10yj`.
- **`disaster` follow-up exploration**: a clipped EV that floors all
  sub-threshold samples to Q=−42. Beats p_make by +2.74; loses to ev
  by ~1.5; ties robust_q25. Confirms EV is the ceiling for fixed
  pointwise utilities at one-step lookahead.

## Methodology insight — single-decision blind spot

The campaign's deepest finding is structural, not from any single
probe: **most book claims are multi-step plans, but most probes are
single-decision contrasts.** Single-decision EV can be the locally-
best move yet still lose to a planner that sets up future tricks
(the book's bread and butter). This means the 19 underpowered + 14
context-limited claims may be stuck at the wrong abstraction level,
not because the book is wrong. The "EV wins" Lens v1 result is
bounded by the test's abstraction level — Lens v1 measured EV's
individual moves against p_make's individual moves; the book was
never in that contest because the book plays plans.

Wave 5 frontier: planning-aware probes. Three architectures —
MCTS over forge, Lookahead-Lens (K=2-3 step), or book-strategy player
(hand-coded multi-step policies). For the book validation use case,
the third is most direct (tests specific claims rather than "is
planning generically good?"); for broader model-design questions,
the second is the cheap general-purpose tool. Detail in
[[w42-book-claim-synthesis-and-ai-directions#methodology-insight-the-single-decision-blind-spot]].

None of the three were built as planning architectures. [[book-strategy-player]]
was designed (2026-05-03/04) but never implemented — superseded before build
by the auction-first [[champion]]/[[jud]] redirection ([[champion-design-review]],
2026-06-09). The Wave 5 probe that actually ran, [[w42-champion-teaching-battery]]
(2026-06-13), answered the book-validation use case a different way: it checks
book claims against the champion's own realized self-play trajectories,
bypassing the planning-architecture question entirely.

## Wave 4.0 — architecture-decision gate

[[w42-bookval-v3-utility-argmax-divergence]] computed argmax-under-
utility for ALL legal actions on the 500 ch05-follow snapshots.

- **Verdict: gate TRIPPED.** EV vs p_make argmax disagrees on 41.2%
  (CI [36.8%, 45.6%]) — order of magnitude above the 5% gate.
- **Direction inverts Wave 3.0's framing.** p_make picks void MORE
  often than EV (38.6% vs 29.4%); CVaR_10 picks void most aggressively
  (42.6%). EV is the outlier — it more often selects a third-option
  discard (37.6% "neither" rate). The book's void advice aligns with
  *risk-aware* utilities, not with mean-EV.
- **Empirical confirmation of the bid=30 mm=1 affine identity at
  argmax level:** p_make and mark_ev agree on every single one of 500
  snapshots (0/500 disagreement).
- **Recommendation:** scope rung-2 utility-tunable searcher (named
  decision pending — see [[w42-bookval-v3-utility-argmax-divergence]]).
  Whether to actually build is a separate prioritization decision.

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
[[w42-bookval-v1-wave1-independent-audit]]
