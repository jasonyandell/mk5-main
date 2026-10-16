## [2026-10-16 | book | Seed-varied Zeb seats, and #117 re-registered against a seed-varied field (#190)]

- `arena.seed_varied.SeedVaried`: a Zeb seat that plays each hand at a planner seed picked from a pool by the deal (keyed as a set of hands; FNV plus splitmix64's mixer, because FNV's low bits alone reached 4 of 16 seeds). It plays the same seed in every arm and both halves, and at a pool of one it is the plain player byte for byte. Its `.walt` is a pooled planner, so RegularsPlay and BookPlay get the deal's seed too.
- [[seed-varied-2026-10-16]] is pre-registered: #117's K, U and Za against a seed-varied field, primary K read against an honest A/A; 352 batches.
