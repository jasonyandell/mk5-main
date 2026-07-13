---
title: w42 Phase 2 Distribution-Aware EV Report
kind: experiment
first_seen: 2026-05-02
last_updated: 2026-07-06
status: superseded
---

**Superseded** by [[w42-powered-branch-atlas-v1]], which recomputes the same
distribution family from saved `q_per_world` tensors rather than collapsed
JSONL PDFs, then by [[w42-branch-atlas-scaled-v0]] and
[[w42-hidden-threat-legacy-mining]] at greater scale.

## Summary

[[w42]] phase 2 treats E[Q] PDFs as tactical evidence rather than collapsing
every legal action to scalar mean E[Q]. The report artifacts in
`w42/distribution_aware_ev_report/` compute per-action distribution features
from the `eq_pdf_v3_sample.jsonl` visualizer sample:

- mean and std;
- make/set threshold mass;
- lower-tail mass;
- q10/q25/q50/q75/q90;
- approximate lower-tail CVaR;
- branch entropy, local peak count, and shelf gap;
- flags for close-mean high-variance actions and scalar EV omissions.

This is report and feature design, not a production selector. No model was
trained, W&B is not applicable, and the claim ledger does not move.

## Data Slice

The source slice is the small E[Q] PDF browser sample described in
[[eq-browser-visualizers]]. It contains 5 games, 140 decisions with PDF data,
346 legal-action PDF rows, and 1,000 sampled worlds per PDF row. Each PDF has 85
integer bins from `-42` to `+42`.

Thresholds follow the visualizer/exporter semantics:

- offense make mass: `E[Q] >= 18`;
- defense set mass: `E[Q] > -18`;
- lower-tail mass: `E[Q] <= -18`;
- lower-tail CVaR: approximate expectation over the lowest 10% of PDF mass.

The sample is enough for visualizer-backed walkthroughs and schema design, but
not enough for final empirical claims.

## Walkthrough

The clearest walkthrough is `sample_002_decl_2`, move 16, active player 1 in
twos. Scalar EV prefers `6-2`: mean `-1.13`, set mass `0.687`, lower-tail mass
`0.313`, std `16.06`, and shelf gap `38`. The distribution-aware view prefers
attention to `4-4`: mean `-4.28`, set mass `0.731`, lower-tail mass `0.269`,
std `15.49`, and the same shelf gap `38`.

The scalar mean says `6-2` is better by about 3.15 E[Q]. The threshold view says
`4-4` produces about 4.4 points more set/make-threshold mass. The tail view says
`4-4` cuts lower-tail mass by about 4.4 points. That is exactly the omission
phase 2 is meant to expose: the choice is not just "higher mean"; it is a
branch-shaped tactical state where threshold probability and bad-tail exposure
move the other way.

## Feature Schema

The report's candidate feature schema separates offline labels from live inputs.
All current distribution features are cheap once an offline E[Q] PDF tensor or
visualizer JSONL exists, but too expensive for live inference unless a small
model learns to predict them from legal public state.

Cheap offline labels:

- `make_mass`, `set_mass`, `lower_tail_mass`;
- quantiles and `cvar_low_10`;
- `std`, `branch_entropy`, `branch_peak_count`, `shelf_gap`;
- `high_variance_close_mean`;
- `scalar_ev_lying_by_omission`.

Not live-cheap as direct features:

- any feature requiring sampled-world PDFs at decision time;
- branch/shelf labels that require hidden-world outcome branches;
- future hidden-domino threat attribution labels, unless converted into a legal
  public-state or learned-belief predictor.

## Findings

The small visualizer sample produced 68 scalar-omission flags and 87
high-variance-close-mean flags across 346 legal-action rows. Mean std across PDF
rows was `16.9081`; max std was `31.0460`. Mean lower-tail mass was `0.2937`,
with some actions entirely in the lower tail.

These counts should be read as instrumentation smoke evidence, not project-level
rates. Their value is that the existing visualizer data already supports the
feature vocabulary needed by [[w42-next-model-decision]]: future claim-led probes
can train or report against threshold mass, bad-tail mass, and branch shape
instead of only scalar mean regret.

[[w42-powered-branch-atlas-v1]] is the powered follow-up for this report. It
recomputes the same distribution family from saved `q_per_world` tensors rather
than collapsed JSONL PDFs, then adds hidden-holder impact rows from the matching
`world_hands` tensor.

[[w42-branch-atlas-scaled-v0]] expands the powered follow-up to all ten
declarations for one seed and adds bid-aware threshold plumbing. Its fixed
`bid_value=30` means distribution rates remain a declaration-coverage pilot, not
real auction strategy evidence.

[[w42-hidden-threat-legacy-mining]] is the full legacy-corpus follow-up for the
same feature family. It shows that close-mean action choices can reduce
lower-tail mass and hidden-downside score without materially changing scalar
mean, while broader safest-tail choices often trade mean and threshold mass for
tail safety.

## Artifacts

| artifact | path | purpose |
|---|---|---|
| script | `w42/distribution_aware_ev_report/build_distribution_aware_ev_report.py` | reads visualizer JSONL and writes report artifacts |
| action features | `w42/distribution_aware_ev_report/action_distribution_features.csv` | one legal-action row per PDF |
| examples | `w42/distribution_aware_ev_report/example_decisions.json` | flagged scalar-omission decisions |
| schema | `w42/distribution_aware_ev_report/candidate_feature_schema.json` | candidate feature contract and cost notes |
| summary | `w42/distribution_aware_ev_report/summary.json` | compact result summary |
| manifest | `w42/distribution_aware_ev_report/manifest.json` | provenance and command |

## Caveats

- This uses the current N=1000 visualizer sample, not a long run.
- The source JSONL existed in the main checkout rather than this isolated
  worktree; the derived artifacts record the absolute source path and SHA-256.
- The branch indicators are simple PDF-shape heuristics, not a tactical theorem.
- Public-state safety belongs to the future predictor or detector. The offline
  PDF-derived labels are legal as training/evaluation targets but not live direct
  inputs.
- No W&B run, HF artifact, model checkpoint, production selector, or
  claim-ledger update is part of this bead.

## Provenance

| field | value |
|---|---|
| bead | `t42-5m82.4` |
| command | `python w42/distribution_aware_ev_report/build_distribution_aware_ev_report.py --input /Users/jason/code/mk5-main/forge/analysis/results/data/eq_pdf_v3_sample.jsonl` |
| source data | `/Users/jason/code/mk5-main/forge/analysis/results/data/eq_pdf_v3_sample.jsonl` |
| W&B links | not applicable |
| claim-ledger impact | no claim-ledger change |

## Links

[[w42]] | [[eq-browser-visualizers]] | [[w42-next-model-decision]] |
[[w42-powered-branch-atlas-v1]] | [[w42-branch-atlas-scaled-v0]] |
[[w42-hidden-threat-legacy-mining]]
