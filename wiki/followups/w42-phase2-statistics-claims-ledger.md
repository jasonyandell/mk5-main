# Follow-ups: w42-phase2-statistics-claims-ledger

Reviewed against code and artifacts on 2026-07-07.

## Corrections

- Page said generation commit was `343a9f4c45244889ea9c1eaa8edacde7e2a69920`; the current artifact records `git_sha: e55a6f89de6e599f795db87c30c29d82b8883313` for the final rebuild (evidence: `w42/statistics_claims_ledger/summary.json`).

## Verified clean

- 64 rows and status counts (23/2/12/21/6) match `summary.json` exactly.
- Void frequencies (42.314 / 46.982 / 10.349 / 0.355), 161/210 = 76.667%, +4.1 Q setter-pounce gap, and 90.0% stopper-ownership figure all appear in `claims.csv`.
- Generator path, `--check` flag, all nine w42 artifact input directories, and bead `t42-5m82.1` verified.

## Follow-ups

- The W&B run link (`.../runs/zm3jdrnj`) was not verifiable from the repo; assumed correct.

## Review (second pass, 2026-07-07)

- Verified — corrections stand. `w42/statistics_claims_ledger/summary.json` records `git_sha: e55a6f89de6e599f795db87c30c29d82b8883313`; the original `343a9f4c` is commit "Plan w42 phase 2 research beads", which predates the ledger artifact, so the page was wrong and the replacement is right. claims.csv has exactly 64 data rows, all cited figures present (42.314/46.982/10.349/0.355, 161/210 = 76.667%, mean_delta=4.10 Q pounce, 90.0, zm3jdrnj), status counts match summary.json. The edit touched only the one provenance cell.
