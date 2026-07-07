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

## Review (second pass, 2026-07-07)

- Verified — corrections stand.
- Re-derived both corrections independently: `ls w42/` confirms `book_validation_v1/` exists and `bookval_v1/`/`jud_v1/` do not; jud artifacts live in `champion/` (jud_net*.pt, jud_net.py) and `arena/` (test_jud_search.py). `w42/branch_atlas_v1/build_branch_atlas.py:34` imports `add_wandb_args, init_wandb` from `w42.wandb_utils` and records `"wandb": wb.status()` in its manifest (line 1191), so it correctly stands as the exception.
- Also confirmed the "do not log to W&B at all" claim survives the one grep hit in `w42/book_validation_v1/`: `wave5/probe_champion_teaching_battery.py:950` passes `wandb_status=None` — it references the field but never initializes W&B.
- "Verified as accurate" items re-checked: smoke manifest `w42/wandb_series_smoke/run.json` (6 points, run id `xq7q9bar`, entity/project `jasonyandell-forge42/w42`); `log_series_point(*, axis, value, metrics, step=None)` at `w42/wandb_utils.py:120`; `variant/index` at `w42/strategy_tag_family_ablations.py:596`; `claim/index` at `w42/setter_defense_claim_validation/analyze_setter_defense.py:505`. All match.
