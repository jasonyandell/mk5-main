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
