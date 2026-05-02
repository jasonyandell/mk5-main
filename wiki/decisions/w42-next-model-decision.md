---
title: w42 Next Model Decision
kind: decision
first_seen: local-2026-05-02
last_updated: local-2026-05-02
status: active
---

## Decision

The next w42 model step is a **targeted v2 strategy-tag probe with direct
claim-regime detectors and distribution-aware labels where available**, not a
broader rich-tag kitchen sink and not a Gus architecture change.

The first target should be one high-value tactical regime with direct labels:

1. setter pounce and count-to-set windows;
2. 84 last-trick weapon preservation;
3. auction bid-margin / bid-only-enough counterfactuals.

The preferred first target is setter pounce if the detector can be implemented
from existing public state plus contract context. It is the clearest gap exposed
by [[w42-setter-defense-claim-validation]]: the current proxy evidence is noisy
because the direct labels are missing.

The refinement from the E[Q] PDF visualizer is that the target should not be
trained or reported only against scalar expected value. Some positions are
branch-shaped: a mean may sit between a safe shelf and a disaster tail. The
strategy problem is then mitigation, preservation, and branch recognition, not
just average-value maximization.

A second refinement is belief impact magnitude. The generated E[Q] artifacts can
record sampled worlds, hidden domino ownership, and outcome branches together.
That allows offline reports to ask which unseen dominoes and holders explain the
large shelves or middle lumps in the PDF. Live play still cannot use hidden truth,
but w42 can use these attributions as belief-quality targets and as diagnostics
for whether a model is worrying about the right unseen threats.

## Evidence Used

The decision rests on four survey findings:

- [[w42-multi-seed-larger-eval-replication]] shows that v0/rich strategy tags
  improve small-model regret over raw public state across five seeds.
- [[w42-strategy-tag-family-ablations]] shows no rich tag family was decisive in
  the single-seed family-drop matrix, so adding more broad tags is not the next
  best move.
- [[w42-setter-defense-claim-validation]] and [[w42-84-claim-validation]] show
  that the most interesting tactical claims are blocked by missing direct
  detectors, not by lack of a larger model.
- [[w42-final-empirical-strategy-report]] separates supported substrate claims
  from underpowered tactical claims, so the next model should be claim-led.
- E[Q] PDF visual inspection shows that mean E[Q] can hide threshold cliffs,
  high-variance shelves, and disaster tails that are directly relevant to book
  strategy concepts such as bracing, preserving stoppers, and mitigating bad
  branches.
- Generated sampled-world artifacts can connect distribution modes back to
  specific hidden domino ownerships, creating an impact-weighted belief target
  that is unavailable to human players but legal as offline supervision and
  evaluation.

## Alternatives Considered

| option | decision | reason |
|---|---|---|
| train longer raw/v0/rich runs immediately | defer | useful as confirmation, but does not answer which book claims are real |
| add every easy detector and rerun | reject for now | the rich-over-v0 signal is modest; indiscriminate features risk hiding the mechanism |
| optimize only scalar E[Q] / mean regret | reject for targeted tactical work | scalar EV is useful but hides branch shape, threshold mass, and tail-risk mitigation opportunities |
| evaluate beliefs only by average ownership calibration | reject for tactical belief work | some unseen dominoes matter far more than others; belief quality should be weighted by outcome impact |
| promote w42 into Gus training now | reject | evidence is not strong enough, and w42's charter keeps Gus core paths out of scope |
| publish HF checkpoint/dataset now | defer | [[w42-hugging-face-artifact-publishing]] says artifacts are not mature enough |
| build direct claim-regime detectors, then train/evaluate | choose | most aligned with the survey's gaps and the user's desire to dig scientifically |

## Recommended Work Package

Create a new bead series for a single targeted regime:

- implement direct public-state/report labels for the chosen regime;
- add deterministic fixture tests and anti-leakage checks;
- build a held-out slice for that regime;
- attach E[Q] PDF or sampled-world distribution features where available:
  threshold mass, variance, quantiles, lower-tail risk, and branch/shelf labels;
- attach hidden-domino threat attribution where saved worlds allow it: for each
  branch or shelf, identify the unseen domino holdings most associated with that
  outcome shift;
- train raw/v0/rich/direct-detector/distribution-aware variants with W&B
  per-epoch series;
- report aggregate regret, tail regret, distribution calibration, threshold-mass
  errors, belief-impact calibration, and claim-specific bucket metrics;
- only then consider claim-ledger movement.

The model can stay small and cheap. The value is in better labels, sharper
slices, branch-aware reports, and honest comparisons.

[[w42-phase2-decision-table]] is the first bridge artifact for this package. It
turns the E[Q] PDF visualizer sample into 140 decision-state rows and 346
legal-action rows with public seat/role context, actor hand/action facts,
distribution-aware labels, and explicit blank columns for bid margin and
hidden-domino ownership. It is ready for schema review and slice analysis, not a
claim-ledger verdict or a training dataset.

[[w42-powered-branch-atlas-v1]] is the first powered follow-up. It uses saved
joint worlds rather than collapsed visualizer PDFs, fills the hidden-threat
label surface on a two-game N=1000 pilot, and logs W&B dashboard series over
processed decisions. Its result is still a report/label artifact, not a model
decision or claim-ledger verdict, but it proves the branch-aware/hidden-impact
measurement loop is now executable.

## Risks

- Direct detectors may require state not present in current Gus corpora.
- Some claims may need generated games with special contracts or late-hand
  states, especially 84.
- Distribution labels may require heavier E[Q] generation than scalar N=10
  labels, so the first probe should keep the slice small and inspectable.
- Hidden-domino attribution must remain an offline label/eval target; using
  hidden truth directly as a live feature would violate the legal public-state
  boundary.
- The branch-atlas v1 pilot uses fixed `bid_value=30`; real bid-margin analysis
  still needs auction metadata and should not infer "bid only enough" claims from
  this slice.
- A better model may exploit tags as shortcuts without learning the intended
  strategy mechanism.
- W&B curves can look exciting while claim evidence remains proxy-only.

## Provenance

No new training, W&B run, HF upload, or claim-ledger update occurred for this
decision memo.

| field | value |
|---|---|
| bead | `t42-csw6.29` |
| commands | `bd show t42-csw6.29 --json`; source report inspection; `git rev-parse HEAD` |
| configs | not applicable |
| data inputs | [[w42-final-empirical-strategy-report]] and cited source reports |
| commit SHA at decision time | `326fc5092d31e94b67fa5b548e6596b6d5e1d3d2` |
| random seeds | not applicable |
| W&B links | not applicable |
| HF links | not applicable |
| claim-ledger impact | no claim-ledger change |

## Links

[[w42]] | [[w42-final-empirical-strategy-report]] |
[[w42-setter-defense-claim-validation]] | [[w42-84-claim-validation]] |
[[w42-strategy-tag-family-ablations]]
