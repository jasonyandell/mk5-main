---
title: w42 Promote Or Retire
kind: decision
first_seen: local-2026-05-02
last_updated: afd4802
status: complete
---

**Complete: fulfilled and superseded.** This charter decision correctly kept
w42 alive as research in May 2026; its recommended next phase (open narrower
beads from [[w42-next-model-decision]]) was executed and the workstream then
evolved two further generations past this decision's scope — the Champion
ladder (2026-06-09 → 06-14) and the [[jud]] stack ([[w42-jud-v1]], graded
2026-07-06) — which now sits closer to [[gus]] (the belief head) and is the
project's leading best-player candidate. The "don't promote w42 model
architecture into Gus" boundary this decision drew was never explicitly
re-litigated for that line — open, not resolved.

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

Keep in w42 research:

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

The tracked w42 artifacts now live under the top-level `w42/` directory. Local
W&B run directories, caches, and other generated machine-state remain
non-durable.

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
