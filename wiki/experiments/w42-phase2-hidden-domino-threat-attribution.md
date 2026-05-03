---
title: w42 Phase 2 Hidden Domino Threat Attribution
kind: experiment
first_seen: local-2026-05-02
last_updated: local-2026-05-02
status: active
---

## Summary

[[w42]] Phase 2 needs a way to explain branch-shaped E[Q] PDFs: which unseen
domino ownerships create the high shelves, middle lumps, and disaster tails?
This page defines the attribution schema and leakage boundary for saved
joint-world artifacts. The first empirical implementation is now
[[w42-powered-branch-atlas-v1]], which generated two N=1000 schema-v2 games and
produced 1026 hidden-threat rows from real `world_hands` and `q_per_world`
tensors. [[w42-branch-atlas-scaled-v0]] expands the same analyzer to all ten
declarations for one seed, producing 5955 hidden-threat rows.
[[w42-hidden-threat-legacy-mining]] then scales the same hidden-impact idea over
the 100 legacy Gus chunks, summarizing 280000 decisions and 748305 legal actions
without exporting a full action table.

## Method

The required source artifact is a `torch.save` payload from [[forge]] E[Q]
generation with `--save-joint-worlds`. Each decision must retain:

- `world_hands`: `[M, 3, 7]`, sampled opponent hands relative to the acting
  player;
- `q_per_world`: `[M, 7]`, oracle Q values for each action slot before the PDF
  collapse;
- `legal_mask`, `player`, and `hands` so rows can be tied back to legal visible
  actions.

The original analyzer in
`w42/hidden_domino_threat_attribution/analyze_joint_world_threats.py` groups
worlds by `(hidden_domino, holder)` for each decision/action and compares the
holder-conditioned Q distribution against the unconditional distribution.
[[w42-powered-branch-atlas-v1]] extends that design into the reusable branch
atlas at `w42/branch_atlas_v1/build_branch_atlas.py`, adding public decision
context, distribution-shape features, W&B progress series, and action/decision
tables beside the hidden-threat rows.

The analyzer now computes high-shelf/threshold mass from recorded `bid_value`
when it exists. The current powered artifacts still use fixed `bid_value=30`;
true bid-margin analysis still requires real auction metadata.

The output row grain is:

`game, decision, action, hidden_domino, relative_holder, absolute_holder`.

Each row records conditioned mass, mean-Q delta, disaster-tail mass delta,
high-shelf mass delta, and a simple top-k ranking score:

`abs(mean_q_delta) + 10 * (abs(tail_low_mass_delta) + abs(shelf_high_mass_delta))`.

## Leakage Boundary

Hidden ownership is an offline label and evaluation target only. It must not be
used as a live feature by Gus, Burl, or any gameplay selector.

The legal online version is belief-shaped: a model may infer a probability that
seat `s` holds domino `d` from public state. The hidden attribution label says
how much that belief error matters for the outcome distribution. That turns
average ownership calibration into impact-weighted belief calibration without
leaking the table truth into live play.

Aggregate E[Q] PDFs alone are insufficient for this report. They can show shelves
and tails, but once `compute_eq_pdf` collapses the worlds, the ownership labels
that explain those modes are gone. The artifact must preserve `world_hands` and
`q_per_world`, and posterior-weighted runs need per-world weights before weighted
attribution is exact.

## Proposed Metrics

| metric | definition | use |
|---|---|---|
| threshold-mass delta by hidden holder | conditioned shelf/tail mass minus baseline shelf/tail mass for `(domino, holder)` | identify unseen holdings that explain make/set cliffs |
| outcome mean delta by hidden holder | conditioned mean Q minus baseline mean Q | rank hidden holdings by expected outcome impact |
| top-k hidden holdings by absolute impact | largest impact rows per decision/action | create inspectable labels for reports and trace review |
| impact-weighted belief calibration | belief error weighted by mean/tail/shelf impact | evaluate whether beliefs focus on tactically important uncertainty |

These metrics are diagnostic. They do not validate a book claim by themselves,
but they can identify the hidden threats that a pounce, 84 preservation, count
donation, or bid-margin detector should care about.

## Artifacts

| artifact | path | status |
|---|---|---|
| manifest | `w42/hidden_domino_threat_attribution/manifest.json` | schema/design artifact |
| summary | `w42/hidden_domino_threat_attribution/summary.json` | schema/design artifact |
| schema proposal | `w42/hidden_domino_threat_attribution/metrics_schema_proposal.json` | schema/design artifact |
| original analyzer | `w42/hidden_domino_threat_attribution/analyze_joint_world_threats.py` | standalone attribution analyzer |
| example rows | `w42/hidden_domino_threat_attribution/example_rows.jsonl` | illustrative only |
| walkthrough | `w42/hidden_domino_threat_attribution/example_walkthrough.md` | illustrative only |
| powered implementation | `w42/branch_atlas_v1/build_branch_atlas.py` | empirical branch/threat atlas |
| powered threat rows | `w42/branch_atlas_v1/hidden_threat_rows.csv` | 1026 real rows |
| scaled threat rows | `w42/branch_atlas_scaled_v0/hidden_threat_rows.csv` | 5955 real rows |
| legacy mining pass | `w42/hidden_threat_legacy_mining/summary.json` | full-corpus compact diagnostics |

## Run Recipe

Standalone attribution sample:

```bash
python -u -m forge.eq.generate \
  --start-seed 9420 \
  --n-games 1 \
  --samples 1000 \
  --save-joint-worlds \
  --schema v2 \
  --bid-value 30 \
  --device mps \
  -o w42/hidden_domino_threat_attribution/tiny_joint_world_sample.pt
python w42/hidden_domino_threat_attribution/analyze_joint_world_threats.py \
  w42/hidden_domino_threat_attribution/tiny_joint_world_sample.pt \
  --output-dir w42/hidden_domino_threat_attribution \
  --top-k 10
```

Real metric pass:

- use at least `M=100` before trusting top-k holder impacts;
- keep a one-game sample for inspection, but run more seeds for calibration;
- prefer bid-aware thresholds once `bid_value` is consistently available;
- record posterior weights if posterior sampling is used.

## Provenance

| field | value |
|---|---|
| bead | `t42-5m82.3` |
| worktree | `.claude/worktrees/w42-phase2-hidden-threat` |
| branch | `w42/phase2-hidden-threat` |
| design-time commit | `343a9f4c45244889ea9c1eaa8edacde7e2a69920` |
| generation status | schema design superseded by [[w42-powered-branch-atlas-v1]] and [[w42-branch-atlas-scaled-v0]] empirical runs |
| W&B links | `https://wandb.ai/jasonyandell-forge42/w42/runs/44z1kl9j`; `https://wandb.ai/jasonyandell-forge42/w42/runs/7fwi2zwn`; `https://wandb.ai/jasonyandell-forge42/w42/runs/74vfet6o` |
| HF links | not applicable |
| claim-ledger impact | no central claim status change |

## Next Steps

- Preserve per-world posterior weights for posterior runs.
- Feed [[w42-hidden-threat-legacy-mining]] mitigation labels into the next
  targeted w42 regime: setter pounce, 84 weapon preservation, or auction
  bid-margin counterfactuals.

## Links

[[w42]] | [[w42-final-empirical-strategy-report]] |
[[w42-next-model-decision]] | [[joint-world-tensor]] |
[[gus-joint-world-tire-kick]] | [[w42-powered-branch-atlas-v1]] |
[[w42-branch-atlas-scaled-v0]] | [[w42-hidden-threat-legacy-mining]]
