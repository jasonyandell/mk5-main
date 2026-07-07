## Corrections

- Page's Code Surface table omitted `w42/claim_analysis/registry.py`; added a row. The registry holds three spec families of 6 each — `GUS_TACTICAL_SPECS`, `BRANCH_ATLAS_SPECS`, `CHAMPION_SPECS` — selected per source kind via `specs_for_source_kind`. Each smoke run's "6 registered claim specs" is the family matching its source kind (branch-atlas smoke registers `BRANCH_ATLAS_SPECS`, per its `claim_specs.csv`; Gus smoke registers `GUS_TACTICAL_SPECS`) (evidence: w42/claim_analysis/registry.py, w42/claim_analysis_smoke/branch_atlas_scaled_v0_wandb/claim_specs.csv).

## Follow-ups

- The branch-atlas smoke command shows `--wandb-name t42-0b4l.2-branch-atlas-smoke`, but the recorded run name is `t42-0b4l.2-branch-atlas-smoke-v2` (w42/claim_analysis_smoke/branch_atlas_scaled_v0_wandb/summary.json) — cosmetic only; run id rj3j0jsz matches.
- All headline metrics (773/280/36/14/6 branch-atlas; 3903/3428/6/4 Gus) verified exactly against both smoke summary.json artifacts.
- The "eight progress points" claim is derivable from the repo, not just plausible: `log_row_progress` (w42/claim_analysis/harness.py:484-501) logs every `chunk_size` rows plus the final row, so 773 rows at `--progress-chunk-size 100` yields exactly 8 points (100..700, 773). The W&B-side rendering (run rj3j0jsz) remains unverified from the repo.

## Review (second pass, 2026-07-07)

- Amended the registry row the first-pass auditor added to the Code Surface table: registry.py is not a "Gus tactical `ClaimSpec`" registry — it holds three families of 6 specs each (`GUS_TACTICAL_SPECS`, `BRANCH_ATLAS_SPECS`, `CHAMPION_SPECS`) dispatched by `specs_for_source_kind` (evidence: w42/claim_analysis/registry.py; w42/claim_analysis/run_existing_rows.py:16,70).
- Corrected the followup's matching claim: the branch-atlas smoke's "6 registered claim specs" are `BRANCH_ATLAS_SPECS`, not Gus tactical specs (evidence: w42/claim_analysis_smoke/branch_atlas_scaled_v0_wandb/claim_specs.csv lists distribution/hidden-threat/position claim ids).
- Upgraded the "eight progress points" note from "plausible" to derivable: harness.py:484-501 logs per 100-row chunk plus final row → exactly 8 points for 773 rows.
- Re-verified and kept: the wandb-name mismatch note (summary.json records `t42-0b4l.2-branch-atlas-smoke-v2`, run id rj3j0jsz matches the page URL) and the exact-match headline metrics (773/280/36/14/6 and 3903/3428/6/4 against both summary.json artifacts).
