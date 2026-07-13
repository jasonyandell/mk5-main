---
title: w42 Hugging Face Artifact Publishing
kind: experiment
first_seen: local-2026-05-02
last_updated: local-2026-05-02
status: active
---

## Summary

[[w42]] does not yet publish a Hugging Face dataset, model checkpoint, or artifact
repo. The initial survey produced useful local scripts, w42 tables, wiki
reports, and W&B runs, but the artifacts are still research-lab outputs rather
than stable public packages.

This is an explicit non-publish decision, not a tooling failure.

## Decision

HF artifact publishing is deferred until at least one of these becomes true:

- a detector dataset has a stable schema, versioned card, and leakage boundary
- a model checkpoint is selected by a multi-seed or larger-run decision, not only
  by a pilot
- an artifact bundle is useful outside this repo without local Gus corpora,
  repo-local w42 paths, or wiki context
- a later promotion decision names a public/exportable artifact home beyond the
  top-level `w42/` research directory

The current durable external surface is W&B, not HF:

- project: `https://wandb.ai/jasonyandell-forge42/w42`
- run comparison: [[w42-wandb-run-comparison-dashboard]]
- series standard: [[w42-wandb-series-logging-standard]]
- five-seed replication group: `w42-csw6-31-multi-seed-larger-eval`

## Rationale

The survey outputs are valuable but not HF-ready:

- The raw/v0/rich models are small w42 research probes, not selected checkpoints.
- The richer tags improved learning, but rich-over-v0 is still modest and not a
  promotion decision.
- Several claim-validation reports use exact enumeration or deterministic
  scoring checks, but their CSV/JSON outputs are tied to local script paths.
- The strongest model evidence uses existing Gus corpus chunks whose card,
  license, split policy, and public packaging are not yet written for HF.
- Several report-only buckets intentionally say "underpowered" or
  "not-yet-tested"; publishing them as datasets now would make the frontier look
  more settled than it is.

## Artifact Triage

| candidate | current home | HF status | reason |
|---|---|---|---|
| raw/v0/rich tiny checkpoints | `w42/*baseline*`; replication output | defer | useful probes, not promoted models |
| v0/v1 detector schemas | `w42/strategy_tags_v0*`; `w42/strategy_tags_v1_map/` | defer | schemas need public dataset-card work before HF |
| claim-validation CSV/JSON tables | `w42/*claim_validation*/` | defer | good local evidence, but report-linked and narrow |
| W&B run metadata | W&B project | not applicable | W&B is already the right external notebook |
| final report / decision memo | wiki | not applicable | repo wiki is the durable narrative surface |

## Provenance

No new training, model evaluation, data generation, W&B run, or HF upload occurred
for this bead.

| field | value |
|---|---|
| bead | `t42-csw6.27` |
| commands | `bd show t42-csw6.27 --json`; wiki/report inspection; `git rev-parse HEAD` |
| configs | not applicable |
| data inputs | existing w42 wiki pages and artifact inventory |
| commit SHA at decision time | `326fc5092d31e94b67fa5b548e6596b6d5e1d3d2` |
| random seeds | not applicable |
| W&B links | not applicable |
| HF links | not applicable |
| claim-ledger impact | no claim-ledger change |

## Links

[[w42]] | [[w42-lab-infrastructure]] |
[[w42-wandb-run-comparison-dashboard]] |
[[w42-multi-seed-larger-eval-replication]]
