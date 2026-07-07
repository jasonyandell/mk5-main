Reviewed against code on 2026-07-07 — no issues found.

- Headline numbers (65.4 / 62.1 / 61.8) and per-decision hints are sourced from the commit message of 5a4c9b9; no results JSON/log artifact exists in-repo, so they're commit-attested rather than artifact-verified.
- `gus/eval/eval_pimc.py` confirms the mechanism: pimc-q uses the single corpus-saved `world_assignment` per decision (K=1, code comment calls it an approximation of K-averaged corpus PIMC — `--k-corpus-cap 200` exists but is unused in the current path), pimc-belief samples K=50 worlds from belief_head and argmaxes mean Q.
- Cheap next probe: actually wire up the K-capped corpus-world averaging (the dangling `--k-corpus-cap`) to test whether pimc-q at K=50 oracle worlds closes the gap to direct — distinguishes "sampler quality" from "single-step PIMC is inherently redundant".
