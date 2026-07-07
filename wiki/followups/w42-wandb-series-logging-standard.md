# Follow-ups: w42-wandb-series-logging-standard

Reviewed against code on 2026-07-07.

## Corrections

- Page said later scripts live at `w42/bookval_v1/` and `w42/jud_v1/`; those paths do not exist — actual locations are `w42/book_validation_v1/` and `champion/`/`arena/` (jud) (evidence: `ls w42/`, `find . -iname "*jud*"`).
- Page cited `w42/branch_atlas_v1/` as an example of later scripts not logging to W&B; that script imports `init_wandb` from `w42/wandb_utils.py` and records W&B status in its manifest (evidence: w42/branch_atlas_v1/build_branch_atlas.py:34).

## Verified as accurate

- Smoke result: 6 points, run id `xq7q9bar`, entity/project `jasonyandell-forge42/w42`, manifest at `w42/wandb_series_smoke/run.json` — all match the manifest.
- `WandbRun.log_series_point(axis=..., value=..., metrics=..., step=...)` exists in `w42/wandb_utils.py` with that signature.
- Axis spot-checks: `variant/index` in `w42/strategy_tag_family_ablations.py`, `claim/index` in `w42/setter_defense_claim_validation/analyze_setter_defense.py`.

## Not verifiable locally

- The live W&B run page (wandb.ai) — external service, not checked.
