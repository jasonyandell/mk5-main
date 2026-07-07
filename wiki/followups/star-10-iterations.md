## Corrections

- Page said cost was "~$0.26/iter; total ~$25 for 15 iters"; the ~$0.26 figure was a 5-example smoke-test iteration, not a production 200/300-example iteration — real per-iter cost is ~$1.67 given ~$25 total (evidence: lem/OVERVIEW.md lines 235, 301, 320).

## Follow-ups

- Per-iteration pass table, pools (3148 seeds 0–199 / 7409 seeds 0–499), loss trend, adapter count, and plateau quotes all match lem/OVERVIEW.md at efad16e/908773a exactly.
- HuggingFace adapters (`star-iter0`–`iter14`) and the `jasonyandell-forge42/lem-star` wandb project are external artifacts — names match repo docs but were not fetched/verified.
- The page title says "15 Iterations" while the filename is star-10-iterations; harmless, but a redirect note could prevent confusion.
