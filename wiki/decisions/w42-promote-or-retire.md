---
title: w42 Promote Or Retire
kind: decision
first_seen: local-2026-05-02
last_updated: local-2026-05-02
status: active
---

## Decision

Keep w42 as an active research workstream. Do not promote it into Gus core
training, Burl behavior, or durable production infrastructure yet. Do not retire
it.

w42 has earned continuation because the initial survey found repeatable signal:
strategy tags improve small-model learning over raw public state, and the report
suite exposes concrete next digs. It has not earned promotion because most
tactical book claims remain underpowered, context-limited, or missing direct
detectors.

## Promotion Boundary

Promote now:

- wiki/report conventions for claim-ledger discipline;
- W&B naming and series logging standard;
- deterministic odds/rules/scoring checks as reference evidence;
- the idea that strategy tags are worth testing.

Keep in scratch/research:

- tiny raw/v0/rich model checkpoints;
- rich-tag feature maps;
- claim-validation scripts and CSVs;
- all unproven tactical recommendations.

Do not promote:

- w42 model architecture into Gus;
- book advice as hard-coded policy rules;
- HF artifacts;
- Burl tool or harness behavior.

## Cleanup And Follow-Up

The current scratch artifacts should remain because they are useful provenance.
Cleanup should wait until a later promotion bead names a durable code home.

The next phase should open new, narrower beads from
[[w42-next-model-decision]], starting with one direct detector/regime work
package. The initial survey epic can close once this decision is committed and
the child beads are closed.

## Provenance

No new training, W&B run, HF upload, or claim-ledger update occurred for this
decision.

| field | value |
|---|---|
| bead | `t42-csw6.30` |
| commands | `bd show t42-csw6.30 --json`; source report inspection; `git rev-parse HEAD` |
| configs | not applicable |
| data inputs | [[w42-final-empirical-strategy-report]], [[w42-next-model-decision]], and cited source reports |
| commit SHA at decision time | `326fc5092d31e94b67fa5b548e6596b6d5e1d3d2` |
| random seeds | not applicable |
| W&B links | not applicable |
| HF links | not applicable |
| claim-ledger impact | no claim-ledger change |

## Links

[[w42]] | [[w42-final-empirical-strategy-report]] |
[[w42-next-model-decision]] | [[w42-hugging-face-artifact-publishing]]
