---
title: w42 Next Model Decision
kind: decision
first_seen: local-2026-05-02
last_updated: local-2026-05-02
status: active
---

## Decision

The next w42 model step is a **targeted v2 strategy-tag probe with direct
claim-regime detectors**, not a broader rich-tag kitchen sink and not a Gus
architecture change.

The first target should be one high-value tactical regime with direct labels:

1. setter pounce and count-to-set windows;
2. 84 last-trick weapon preservation;
3. auction bid-margin / bid-only-enough counterfactuals.

The preferred first target is setter pounce if the detector can be implemented
from existing public state plus contract context. It is the clearest gap exposed
by [[w42-setter-defense-claim-validation]]: the current proxy evidence is noisy
because the direct labels are missing.

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

## Alternatives Considered

| option | decision | reason |
|---|---|---|
| train longer raw/v0/rich runs immediately | defer | useful as confirmation, but does not answer which book claims are real |
| add every easy detector and rerun | reject for now | the rich-over-v0 signal is modest; indiscriminate features risk hiding the mechanism |
| promote w42 into Gus training now | reject | evidence is not strong enough, and w42's charter keeps Gus core paths out of scope |
| publish HF checkpoint/dataset now | defer | [[w42-hugging-face-artifact-publishing]] says artifacts are not mature enough |
| build direct claim-regime detectors, then train/evaluate | choose | most aligned with the survey's gaps and the user's desire to dig scientifically |

## Recommended Work Package

Create a new bead series for a single targeted regime:

- implement direct public-state/report labels for the chosen regime;
- add deterministic fixture tests and anti-leakage checks;
- build a held-out slice for that regime;
- train raw/v0/rich/direct-detector variants with W&B per-epoch series;
- report aggregate regret, tail regret, and claim-specific bucket metrics;
- only then consider claim-ledger movement.

The model can stay small and cheap. The value is in better labels, sharper
slices, and honest comparisons.

## Risks

- Direct detectors may require state not present in current Gus corpora.
- Some claims may need generated games with special contracts or late-hand
  states, especially 84.
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
