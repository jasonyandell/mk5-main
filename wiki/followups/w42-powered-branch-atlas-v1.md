# w42-powered-branch-atlas-v1 — audit 2026-07-07

## Corrections

- Page said "the loop can generate N=1000 joint-world games"; artifact says 2 source games with 1000 sampled joint worlds per decision (evidence: w42/branch_atlas_v1/summary.json `sample_counts_by_decision`, manifest `input_games: 2`, `n_samples: 1000`). Phrasing fixed in place.

Everything else verified: all artifact paths exist; Data Slice counts (56 decisions, 134 action rows, 1026 threat rows, seeds 9420/9421, decls blanks/ones), Findings table (25/56 omission, 46/56 multi-peak, 84/79/121 action tags, max 42.84, mean 17.69, 48/51/45 selector counts), and the 5-5 bidder-partner example (13.98 → 26.70, tail −0.101, shelf +0.248) all match summary.json / hidden_threat_rows.csv exactly.

## Follow-ups

- W&B run 44z1kl9j is external; local wandb dir path exists in summary but run contents not verified from repo.
- A cheap next probe: the multi-peak action-row tag count is 119 (summary) but the page only reports the decision-level 46/56 — could surface the action-level rate too.

## Review (second pass, 2026-07-07)

- Verified — corrections stand. Re-derived the sole page edit from `w42/branch_atlas_v1/manifest.json` (`input_games: 2`, `n_samples: 1000`) and `summary.json` (`sample_counts_by_decision: {"1000": 56}`); the original "N=1000 joint-world games" phrasing was wrong, the replacement is right, and no adjacent content was damaged. Independently re-checked every "everything else verified" number against `summary.json` and `hidden_threat_rows.csv` row 1 (5-5/bidder_partner: 13.9786 → 26.6973, tail −0.1012, shelf +0.2482) — all exact. Both follow-up suggestions confirmed sensible (local `wandb/` mirror exists; action-level `multi_peak_pdf: 119` is real and unreported on the page).
