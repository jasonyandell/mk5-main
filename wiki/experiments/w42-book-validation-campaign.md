---
title: w42 Book Validation Campaign
kind: experiment
first_seen: local-2026-05-03
last_updated: local-2026-05-03
status: active
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

## Status (live)

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
| 2.E.2 | setter-pounce high-bid | t42-8kbh | filed, blocked on 2.B.2 |
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

Wave 2 (so far) absorbed 2 more promotions: `ch02-bid-only-enough`
(`not-yet-tested` -> `context-limited`) and
`ch12-setter-pounce-high-bid-off` (`underpowered` -> `context-limited`),
both backed by paired same-hand counterfactual evidence from
[[w42-bookval-v1-wave2-bid-aware-atlas]] (n=14,000 paired decisions).

No claim has been demoted. No claim has crossed into `supported` from
`context-limited` or weaker.

Status counts after Wave 2.B.2:

| status | count |
|---|---:|
| supported | 23 |
| context-limited | 16 |
| underpowered | 19 |
| not-yet-tested | 4 |
| contradicted | 2 |

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

## Links

[[w42]] | [[w42-book-claim-synthesis-and-ai-directions]] |
[[w42-bookval-v1-wave2-infra-design]] |
[[w42-phase4-final-claim-audit]] |
[[w42-phase4-claim-completion-board]] |
[[w42-bookval-v1-wave1-distribution-lens-reranker]] |
[[w42-bookval-v1-wave1-mark-utility-transform]] |
[[w42-bookval-v1-wave1-hidden-threat-impact-ranker]] |
[[w42-bookval-v1-wave1-cross-ai-agreement]] |
[[w42-bookval-v1-wave1-independent-audit]]
