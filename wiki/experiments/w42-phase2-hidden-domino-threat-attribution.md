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
This bead defines the first attribution schema and script for saved
joint-world artifacts.

No empirical attribution table was produced in this worktree. The tiny
generation command was attempted, but the local Python environment was missing
`lightning`, which is required by `forge.eq.oracle.Stage1Oracle`. The result is
therefore a concrete, runnable design plus illustrative rows, not a claim about
any real seed.

## Method

The required source artifact is a `torch.save` payload from [[forge]] E[Q]
generation with `--save-joint-worlds`. Each decision must retain:

- `world_hands`: `[M, 3, 7]`, sampled opponent hands relative to the acting
  player;
- `q_per_world`: `[M, 7]`, oracle Q values for each action slot before the PDF
  collapse;
- `legal_mask`, `player`, and `hands` so rows can be tied back to legal visible
  actions.

The analyzer in `w42/hidden_domino_threat_attribution/analyze_joint_world_threats.py`
groups worlds by `(hidden_domino, holder)` for each decision/action and compares
the holder-conditioned Q distribution against the unconditional distribution.
The first-pass thresholds are `Q <= -18` for disaster-tail mass and `Q >= 18`
for high-shelf mass, matching the bid-30 cliff used in the E[Q] PDF and action
selection surfaces.

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
| manifest | `w42/hidden_domino_threat_attribution/manifest.json` | created |
| summary | `w42/hidden_domino_threat_attribution/summary.json` | created |
| schema proposal | `w42/hidden_domino_threat_attribution/metrics_schema_proposal.json` | created |
| analyzer | `w42/hidden_domino_threat_attribution/analyze_joint_world_threats.py` | created |
| example rows | `w42/hidden_domino_threat_attribution/example_rows.jsonl` | illustrative only |
| walkthrough | `w42/hidden_domino_threat_attribution/example_walkthrough.md` | illustrative only |

## Attempted Tiny Run

Command attempted:

```bash
python -u -m forge.eq.generate \
  --start-seed 9200 \
  --n-games 1 \
  --samples 10 \
  --save-joint-worlds \
  --device mps \
  -o w42/hidden_domino_threat_attribution/tiny_joint_world_sample.pt
```

Failure:

```text
ModuleNotFoundError: No module named 'lightning'
```

The failure occurred while importing `forge.eq.oracle.Stage1Oracle`, before any
sampled-world artifact was written. `forge/requirements.txt` lists
`lightning>=2.0`, so the command recipe remains the expected small reproduction
path after dependencies are installed.

## Run Recipe

Tiny inspectable smoke sample:

```bash
python -m pip install -r forge/requirements.txt
python -u -m forge.eq.generate \
  --start-seed 9200 \
  --n-games 1 \
  --samples 10 \
  --save-joint-worlds \
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
| generation status | attempted, blocked by missing `lightning` dependency |
| W&B links | not applicable |
| HF links | not applicable |
| claim-ledger impact | no central claim status change |

## Next Steps

- Install the forge requirements or run in the normal forge environment, then
  execute the tiny recipe above.
- Add bid-aware make/set thresholds from `bid_value`.
- Preserve per-world posterior weights for posterior runs.
- Feed the resulting top-k rows into the next targeted w42 regime: setter pounce,
  84 weapon preservation, or auction bid-margin counterfactuals.

## Links

[[w42]] | [[w42-final-empirical-strategy-report]] |
[[w42-next-model-decision]] | [[joint-world-tensor]] |
[[gus-joint-world-tire-kick]]
