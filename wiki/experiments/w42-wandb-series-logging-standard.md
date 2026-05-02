---
title: w42 W&B Series Logging Standard
kind: experiment
first_seen: local-2026-05-02
last_updated: local-2026-05-02
status: active
---

## Summary

[[w42]] treats W&B as a lab notebook, not only as a final-score registry. Runs
with epochs, variants, seeds, bootstrap checkpoints, corpus chunks, or repeated
evaluation checkpoints should log multiple points on a meaningful axis. Failed
or splatted runs remain useful evidence when W&B initialized before the failure.

The standard exists because early w42 work produced good final summaries but
under-exposed run trajectories. A run with 800 iterations should normally show
multiple W&B points unless the script is deterministic and one-shot.

## Standard

Every W&B-enabled w42 script follows these rules:

- Long iterative runs log multiple points, not only a final summary.
- Each point includes a human-readable numeric axis metric in addition to W&B's
  internal `_step`.
- Training uses `epoch`.
- Ablations use `variant/index`.
- Claim-spec sweeps use `claim/index` or `spec/index`.
- Bootstrap/statistical loops use checkpointed axes such as
  `bootstrap/samples_seen` when the iteration count is large.
- Corpus generation or chunked evals use `chunk/index`, `seed/index`, or an
  equivalent named axis.
- Final summaries still record best/final values and status fields.
- Uncaught post-init failures still set `status=failed`,
  `failure/type`, `failure/message`, and `failure/traceback_tail`.

The shared helper `w42/wandb_utils.py` provides
`WandbRun.log_series_point(axis=..., value=..., metrics=..., step=...)` so
future scripts can make the axis explicit without repeating W&B boilerplate.

## Script Inventory

| script | W&B behavior after this bead | series status |
|---|---|---|
| `w42/raw_public_state_baseline.py` | live/default W&B; logs train/eval metrics once per epoch with `epoch` and final/best groups | series-ready |
| `w42/v0_strategy_tags_baseline.py` | live/default W&B; logs train/eval metrics once per epoch with `epoch` and final/best groups | series-ready |
| `w42/rich_tag_many_signal_probe.py` | live/default W&B; logs rich-model train/eval metrics once per epoch with `epoch`, then final/bucket deltas | series-ready |
| `w42/strategy_tag_family_ablations.py` | live/default W&B; now logs a generic per-variant series on `variant/index` plus variant-specific metric names | series-ready |
| `w42/setter_defense_claim_validation/analyze_setter_defense.py` | live/default W&B; now logs one point per claim on `claim/index` plus claim-specific metric names | series-ready for claim sweep; bootstrap remains aggregate |
| `w42/eighty_four_claim_validation/validate_84_claims.py` | live/default W&B; logs one final deterministic/static validation summary | final-only, acceptable one-shot |
| `w42/doubles_no_trump_claim_validation/validate_doubles_no_trump.py` | live/default W&B; logs one final deterministic ruleset/static validation summary | final-only, acceptable one-shot |
| `w42/scoring_objective_drift_claim_validation/validate_scoring_objective_drift.py` | live/default W&B; logs one final deterministic scoring-transform summary | final-only, acceptable one-shot |
| `w42/style_partnership_concept_buckets/build_style_partnership_report.py` | live/default W&B; logs a synthesized bucket table/report summary | report-only; add bucket/spec series if expanded |
| `w42/partner_support_claim_validation/analyze_partner_support.py` | no W&B instrumentation; local summary records `wandb_links: not applicable` | report-only backlog if rerun |
| `w42/bidding_risk_budget_claim_validation/validate_bidding_risk_budget.py` | no W&B match found in the current script scan | report-only backlog if rerun |
| `w42/bidder_sequencing_claim_validation/analyze.py` | no W&B match found in the current script scan | report-only backlog if rerun |
| `w42/odds_ruleset_claim_validation/validate_odds_ruleset.py` | no W&B instrumentation; deterministic local ruleset summary | final-only, acceptable one-shot |
| `w42/detector_tests.py` | no W&B instrumentation; local detector/unit-style report | final-only, acceptable one-shot |
| `w42/data_adapter_smoke.py` | no W&B instrumentation; local adapter shape smoke | final-only, acceptable one-shot |
| `w42/strategy_tags_v0.py` | no W&B instrumentation; feature-map generation/reporting | final-only, acceptable one-shot |

## Smoke

The smoke script `w42/wandb_series_smoke.py` logs synthetic epoch and
bootstrap checkpoint series without loading Gus corpora or training a model.

Command:

```bash
python w42/wandb_series_smoke.py \
  --epochs 3 \
  --bootstrap-checkpoints 3 \
  --bootstrap-samples 300 \
  --wandb-mode auto \
  --wandb-name t42-csw6.32-wandb-series-smoke-s0000-4ff779d
```

Result:

- W&B run:
  `https://wandb.ai/jasonyandell-forge42/w42/runs/xq7q9bar`
- W&B run id: `xq7q9bar`
- Points logged: 6
- Series metrics visible: `epoch`, `train/loss`, `eval/mean_regret`,
  `bootstrap/samples_seen`, `bootstrap/checkpoint`, `bootstrap/ci_width`, and
  `series/global_step`
- Local manifest: `w42/wandb_series_smoke/run.json`
- HF links: `not applicable`
- Claim-ledger impact: `no claim-ledger change`

## Completion Notes

This bead changes instrumentation only. It does not rerun any expensive
experiment, move any claim status, publish HF artifacts, or change model
training conclusions.

The current W&B project remains
`https://wandb.ai/jasonyandell-forge42/w42`.

## Links

[[w42]] | [[w42-lab-infrastructure]] |
[[w42-wandb-run-comparison-dashboard]]
