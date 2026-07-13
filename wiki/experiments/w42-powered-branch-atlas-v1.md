---
title: w42 Powered Branch Atlas v1
kind: experiment
first_seen: 2026-05-02
last_updated: 2026-07-06
status: superseded
---

**Superseded** three days later in scope by [[w42-branch-atlas-scaled-v0]]
(10 games, all declarations, bid-aware threshold plumbing), and in corpus
size by the 100-file [[w42-hidden-threat-legacy-mining]].

## Summary

[[w42]] now has its first powered branch-atlas pass from saved joint-world E[Q]
data. The artifact lives in `w42/branch_atlas_v1/` and joins public decision
context, per-action distribution shape, and offline hidden-domino ownership
impact in one inspectable table.

This is the first empirical bridge from the visual "candlewax" insight to
hidden-threat labels. It does not train a model and does not validate a book
claim. It proves the loop can generate joint-world games with N=1000 sampled worlds per decision, keep hidden
owner/Q outcomes, log a multi-point W&B dashboard, and produce rows that say
which unseen domino holders most explain the high shelf, disaster tail, or
large branch spread.

## Data Slice

| field | value |
|---|---:|
| source games | 2 |
| seeds | 9420, 9421 |
| declarations | blanks, ones |
| samples per decision | 1000 |
| decisions processed | 56 |
| legal-action rows | 134 |
| hidden-threat rows | 1026 |
| bid value | fixed schema-v2 value `30` |

The source tensor is
`w42/branch_atlas_v1/eq_pdf_s9420-9421_joint_1000s_v2.pt`.
It was generated with `forge.eq.generate --save-joint-worlds --schema v2`, so
each decision retains `world_hands` and `q_per_world` instead of only the
collapsed PDF.

The fixed `bid_value=30` is useful for threshold labels but is not a real auction
margin. Bid-margin claims remain blocked on real auction/bid-margin metadata.

## Artifacts

| artifact | path | grain |
|---|---|---|
| builder | `w42/branch_atlas_v1/build_branch_atlas.py` | reproducible script |
| source tensor | `w42/branch_atlas_v1/eq_pdf_s9420-9421_joint_1000s_v2.pt` | generated E[Q] games |
| decision table | `w42/branch_atlas_v1/decision_states.csv` | one row per decision |
| action table | `w42/branch_atlas_v1/decision_actions.csv` | one row per legal action |
| threat rows | `w42/branch_atlas_v1/hidden_threat_rows.csv` | top hidden-holder impacts per action |
| examples | `w42/branch_atlas_v1/examples.json` | inspectable branch cases |
| manifest | `w42/branch_atlas_v1/manifest.json` | schema/provenance |
| summary | `w42/branch_atlas_v1/summary.json` | compact metrics |

W&B run:
`https://wandb.ai/jasonyandell-forge42/w42/runs/44z1kl9j`.

The W&B run is a dashboard-style run: it logs repeated points over
`progress/decisions_processed`, including action rows, threat rows, scalar
omission action rate, multi-peak action rate, high-std action rate, hidden-impact
scores, and joint-world coverage. It is not a single final-score frame.

## Findings

The two-game pilot is too small for project-level rates, but it is strong enough
to validate the measurement surface.

| finding | value |
|---|---:|
| joint-world decision coverage | 100% |
| scalar-EV omission decisions | 25 / 56 |
| decisions with multi-peak PDF action | 46 / 56 |
| action rows tagged high std | 84 / 134 |
| action rows tagged large lower tail | 79 / 134 |
| action rows tagged hidden-threat large impact | 121 / 134 |
| max hidden-impact score | 42.84 |
| mean top hidden-impact score | 17.69 |

The actual selected action was the top-mean action in 48 of 56 decisions, the
top threshold-mass action in 51 of 56, and the safest lower-tail action in 45 of
56. That means the policy mostly follows the scalar/threshold selector, but the
atlas still preserves where branch shape and hidden threat magnitude disagree.

## Hidden-Threat Label

For each legal action, the atlas groups sampled worlds by `(hidden_domino,
holder)` and compares the holder-conditioned Q distribution to the baseline
action distribution. Each top-k row records:

- conditioned world mass;
- mean-Q delta;
- lower-tail mass delta for `Q <= -18`;
- high-shelf mass delta for `Q >= 18`;
- std delta;
- an absolute impact score plus downside/upside scores.

Example shape: one bidder-opening action on seed 9420 marks hidden `5-5` with
the bidder's partner as a high-impact upside driver. The baseline action mean is
about `13.98`; conditioning on that ownership lifts it to about `26.70`, drops
lower-tail mass by about `0.10`, and raises high-shelf mass by about `0.25`.

This is the belief-impact label the project wanted: not just "who might hold a
domino," but "how much does that hidden ownership move the outcome branch?"

## Leakage Boundary

Hidden ownership and `q_per_world` outcomes are offline labels and diagnostics.
They may supervise a learned belief-quality or threat-attention target, but they
must not be live policy inputs.

Online-safe inputs in the atlas include public score/trick state, seat role,
actor hand facts, and candidate action facts. Offline labels include E[Q]
distribution shape, hidden-owner impact, future outcome branches, and
scalar-omission flags.

## Scientific Status

This is a powered pilot, not a conclusion. It validates:

- joint-world generation works in the local forge environment;
- hidden threat attribution is no longer only a schema proposal;
- W&B can serve as a live progress dashboard for w42 report experiments;
- branch-shaped EV states can be archived with both distribution and hidden
  ownership impact columns.

It does not validate any book strategy claim, beat E[Q] N=10, train w42, publish
HF artifacts, or move the claim ledger.

## Commands

```bash
python -m forge.eq.generate \
  --start-seed 9420 \
  --n-games 2 \
  --samples 1000 \
  --save-joint-worlds \
  --schema v2 \
  --bid-value 30 \
  --device mps \
  -o w42/branch_atlas_v1/eq_pdf_s9420-9421_joint_1000s_v2.pt

python w42/branch_atlas_v1/build_branch_atlas.py \
  w42/branch_atlas_v1/eq_pdf_s9420-9421_joint_1000s_v2.pt \
  --output-dir w42/branch_atlas_v1 \
  --top-k 8 \
  --examples 12 \
  --log-every-decisions 4 \
  --wandb-mode online \
  --wandb-name t42-gc7m-branch-atlas-v1-s9420-9421-n1000
```

Validation:

```bash
python -m py_compile w42/branch_atlas_v1/build_branch_atlas.py
python - <<'PY'
import csv, json
from pathlib import Path
base = Path("w42/branch_atlas_v1")
summary = json.loads((base / "summary.json").read_text())
assert summary["coverage"]["processed_joint_world_decisions"] == 56
assert summary["rows"]["action_rows"] == 134
assert summary["rows"]["hidden_threat_rows"] == 1026
for name in ["decision_states.csv", "decision_actions.csv", "hidden_threat_rows.csv"]:
    with (base / name).open(newline="") as f:
        assert next(csv.reader(f))
PY
```

## Next Steps

The next useful step is not a larger conclusion. It is a larger and more varied
slice: more seeds, more declarations, real auction/bid-margin metadata, and
branch-atlas summaries by role and strategy detector. After that, w42 can train a
small belief/threat probe that predicts impact-weighted hidden-threat labels from
legal public state and learned beliefs.

[[w42-branch-atlas-scaled-v0]] completes the first part of that next step: one
seed across all declarations, with bid-aware threshold plumbing. It still does
not supply real auction metadata or bid margin.

## Links

[[w42]] | [[w42-next-model-decision]] |
[[w42-phase2-decision-table]] |
[[w42-phase2-hidden-domino-threat-attribution]] |
[[w42-phase2-distribution-aware-ev-report]] |
[[w42-branch-atlas-scaled-v0]] |
[[gus-joint-world-tire-kick]]
