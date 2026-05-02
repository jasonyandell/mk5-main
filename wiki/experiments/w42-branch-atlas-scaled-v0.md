---
title: w42 Branch Atlas Scaled v0
kind: experiment
first_seen: local-2026-05-02
last_updated: local-2026-05-02
status: active
---

## Summary

[[w42]] scaled the powered branch-atlas loop from the first two-game pilot to an
all-declarations N=1000 slice. The artifact lives in
`w42/branch_atlas_scaled_v0/` and uses the generalized
`w42/branch_atlas_v1/build_branch_atlas.py` analyzer with its own bead,
experiment, and W&B metadata.

This run also fixes a provenance gap: [[forge]] E[Q] generation now uses
`bid_value` for p_make selection thresholds instead of only recording it, and
the atlas computes threshold labels from each recorded bid value. The scaled v0
data still uses a fixed `bid_value=30`, so it does not test real auction
metadata, bid margin, or "bid only enough" claims.

## Data Slice

| field | value |
|---|---:|
| bead | `t42-uptc` |
| source games | 10 |
| seed | 9430 |
| declarations | all 10 declarations |
| samples per decision | 1000 |
| decisions processed | 280 |
| legal-action rows | 773 |
| hidden-threat rows | 5955 |
| bid value | fixed recorded `30` |

The source tensor is
`w42/branch_atlas_scaled_v0/eq_pdf_s9430_d10_bid30_joint_1000s_v2.pt`.

W&B run:
`https://wandb.ai/jasonyandell-forge42/w42/runs/7fwi2zwn`.

The W&B run logs repeated points over `progress/decisions_processed` rather than
only a final summary. This makes progress, failure, and stabilization visible for
branch rates, hidden-threat rows, coverage, and wall time.

## Findings

The slice is larger and more declaration-diverse than
[[w42-powered-branch-atlas-v1]], but it is still a small one-seed report. It
should be read as a stronger measurement surface, not as statistical closure.

| finding | value |
|---|---:|
| joint-world decision coverage | 100% |
| scalar-EV omission decisions | 147 / 280 |
| decisions with multi-peak PDF action | 230 / 280 |
| action rows tagged high std | 457 / 773 |
| action rows tagged large lower tail | 392 / 773 |
| action rows tagged hidden-threat large impact | 642 / 773 |
| max hidden-impact score | 56.82 |
| mean top hidden-impact score | 18.46 |

The actual selected action was the top-mean action in 251 of 280 decisions, the
top threshold-mass action in 234 of 280, and the safest lower-tail action in 216
of 280. The selector remains mostly aligned with scalar/threshold preferences,
but many decisions retain branch-shape or hidden-threat interpretations that a
single scalar cannot explain.

## Bid-Aware Thresholds

The generator and atlas now share bid-aware threshold semantics:

- offense at bid `B` uses `Q >= 2B - 42`;
- defense uses the matching set threshold, preserving the historical bid-30
  convention of `Q >= -17`;
- bid `84` is treated as a take-all contract target of 42 points for threshold
  purposes.

This makes future bid-stress or auction-aware artifacts possible. The current
scaled v0 slice deliberately holds bid fixed at 30 to isolate declaration
diversity and keep interpretation clean.

## Scientific Status

This run validates a bigger branch-atlas loop and a bid-aware plumbing fix. It
does not validate a book claim, train w42, beat E[Q] N=10, publish an HF
artifact, or move the claim ledger.

Hidden ownership and `q_per_world` outcomes remain offline labels. They may
supervise belief quality, hidden-threat attention, or mitigation reports, but
they must not become live hidden-truth features.

## Commands

```bash
python -m forge.eq.generate \
  --start-seed 9430 \
  --n-games 10 \
  --n-decl-per-seed 10 \
  --samples 1000 \
  --save-joint-worlds \
  --schema v2 \
  --bid-value 30 \
  --device mps \
  -o w42/branch_atlas_scaled_v0/eq_pdf_s9430_d10_bid30_joint_1000s_v2.pt

python w42/branch_atlas_v1/build_branch_atlas.py \
  w42/branch_atlas_scaled_v0/eq_pdf_s9430_d10_bid30_joint_1000s_v2.pt \
  --output-dir w42/branch_atlas_scaled_v0 \
  --bead-id t42-uptc \
  --experiment w42-branch-atlas-scaled-v0 \
  --top-k 8 \
  --examples 16 \
  --log-every-decisions 20 \
  --wandb-mode online \
  --wandb-group w42-branch-atlas-scaled-v0 \
  --wandb-name t42-uptc-branch-atlas-scaled-v0-s9430-d10-n1000
```

Validation:

```bash
python -m py_compile \
  forge/eq/generate/actions.py \
  forge/eq/generate/cli.py \
  forge/eq/generate/pipeline.py \
  w42/branch_atlas_v1/build_branch_atlas.py

python -m pytest forge/eq/test_select_actions.py -q
```

## Links

[[w42]] | [[w42-powered-branch-atlas-v1]] |
[[w42-next-model-decision]] |
[[w42-phase2-hidden-domino-threat-attribution]] |
[[w42-phase2-distribution-aware-ev-report]]
