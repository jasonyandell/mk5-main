---
title: "Gus population depth campaign: immutable completion evidence"
kind: source
first_seen: 2026-10-18
last_updated: 2026-10-18
status: complete
---

## Source bundle

Immutable receipts behind [[gus-population-depth-2026-10-18]],
[[gus-tiny-gpt]] and [[gus-population-iteration]]. Findings and their scope live
on the experiment page; this digest identifies the evidence.

| Artifact | Pin |
|---|---|
| Preregistered protocol and trainer | Git `4a8596fc` |
| Runtime run-ID/provenance repair | Git `deebd4d0` |
| Final publication implementation and model-card wording | Git `a9cf6c5c` |
| Selected model | HF model `e6b88165cc017594f09f62d55c2d28ea55b450d3` |
| Final formation, chart and confirmation publication | HF dataset `bcf9e609560f2b7fa05693f4723cf63752217594` |
| Supplemental independent checks and source scripts | HF dataset `59f2e4dbadd6a4514dfc22abb9bcc82205a43b26` |
| Selected source checkpoint SHA256 | `ee5be88a5e545f17b374287a064166f3ab338fd8c4eda526e4b835c012f10ae1` |

[Registered protocol at 4a8596fc](https://github.com/jasonyandell/mk5-main/blob/4a8596fc/wiki/experiments/gus-population-depth-2026-10-18.md)
contains the material-improvement anchor, separate retained maximum and identity
initialization eligibility before the first outcome. These rules did not change
to accommodate the result.

## Evidence map

- [Selected model release](https://huggingface.co/jasonyandell/gus/tree/e6b88165cc017594f09f62d55c2d28ea55b450d3): all exported tensors, exact input/output
  schema and example, checksums, model card, provenance and confirmation report.
- [Final panel and preselected checkpoint](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/blob/59f2e4dbadd6a4514dfc22abb9bcc82205a43b26/lem/gus-depth-campaign-2026-10-18/final.json).
- [Every completed learner input/output and replay audit](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/blob/59f2e4dbadd6a4514dfc22abb9bcc82205a43b26/lem/gus-depth-campaign-2026-10-18/audit.json).
- [Independent schedule, growth, source and confirmation audit](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/blob/59f2e4dbadd6a4514dfc22abb9bcc82205a43b26/lem/gus-depth-campaign-2026-10-18/completion-checks.json).
- [Anonymous object metadata and immutable-tag verification](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/blob/59f2e4dbadd6a4514dfc22abb9bcc82205a43b26/lem/gus-depth-campaign-2026-10-18/remote-archive-verification.json).
- [W&B summaries read back against raw smoke reports](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/blob/59f2e4dbadd6a4514dfc22abb9bcc82205a43b26/lem/gus-depth-campaign-2026-10-18/wandb-smoke-verification.json).
- [Runnable supplemental verification sources and logs](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/59f2e4dbadd6a4514dfc22abb9bcc82205a43b26/lem/gus-depth-campaign-2026-10-18/verification).
- [Complete raw formation](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/59f2e4dbadd6a4514dfc22abb9bcc82205a43b26/lem/gus-depth-campaign-2026-10-18): each `gNNN/` holds eight optimizer
  checkpoints, inputs/outputs, full replays, development and smoke reports,
  frozen population hashes and generation publication receipts.

The archive pin retains earlier generation releases and growth snapshots.
Generated artifacts are CC0. The nanoGPT-derived implementation retains its MIT
attribution. Independent object-hash verification is distinct from downloading
all raw bytes; the release receipts state exactly what was downloaded.
