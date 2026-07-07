## Corrections

- Page's Code Surface table omitted `w42/claim_analysis/registry.py` (the built-in Gus tactical `ClaimSpec` registry, source of the "6 registered claim specs"); added a row (evidence: w42/claim_analysis/registry.py).

## Follow-ups

- The branch-atlas smoke command shows `--wandb-name t42-0b4l.2-branch-atlas-smoke`, but the recorded run name is `t42-0b4l.2-branch-atlas-smoke-v2` (w42/claim_analysis_smoke/branch_atlas_scaled_v0_wandb/summary.json) — cosmetic only; run id rj3j0jsz matches.
- All headline metrics (773/280/36/14/6 branch-atlas; 3903/3428/6/4 Gus) verified exactly against both smoke summary.json artifacts.
- The "eight progress points" W&B trajectory claim lives only in W&B (run rj3j0jsz) — unverifiable from the repo, but plausible given `progress/rows_processed` logging in harness.py.
