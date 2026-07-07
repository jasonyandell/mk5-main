## Corrections

- Page said cost was "~$0.26/iter; total ~$25 for 15 iters"; the ~$0.26 figure was a 5-example smoke-test iteration, not a production 200/300-example iteration — real per-iter cost is ~$1.67 given ~$25 total (evidence: lem/OVERVIEW.md lines 235, 301, 320).

## Follow-ups

- Per-iteration pass table, pools (3148 seeds 0–199 / 7409 seeds 0–499), loss trend, adapter count, and plateau quotes all match lem/OVERVIEW.md at efad16e/908773a exactly.
- HuggingFace adapters (`star-iter0`–`iter14`) and the `jasonyandell-forge42/lem-star` wandb project are external artifacts — names match repo docs but were not fetched/verified.
- The page title says "15 Iterations" while the filename is star-10-iterations; harmless, but a redirect note could prevent confusion.

## Review (second pass, 2026-07-07)

- Verified — corrections stand. lem/OVERVIEW.md:235 shows ~$0.26 was a 5-example smoke iteration (151s on B200, line 259); production cost was ~$15 for iters 0–9 (line 301) and ~$25 total after 15 iters (line 320), so ~$1.67/iter average is correct. Table, pools, losses, adapter names, and plateau quotes re-checked against lem/OVERVIEW.md — all match; the edit damaged nothing nearby.
