---
title: Run Artifacts Policy — claims in git, measurements ephemeral
kind: decision
first_seen: 2026-07-13
last_updated: 2026-07-13
status: active
---

Git holds **claims and the means to reproduce them, not the measurements
themselves**. Adopted 2026-07-13 during PR review of the research night, on
Jason's reviewer instinct that arena run data does not belong in git.

## The three tiers

1. **The wiki experiment page** (committed) — the numbers that matter, the
   registered bands, the verdicts, and the exact reproduction command with
   its code sha. This is the scientific record; a result that is not written
   here does not exist.
2. **A curated evidence bundle** (committed, small) — only for load-bearing
   claims: `summary.json` per run, plus `per_game.csv` exactly where a
   paired-CI claim depends on game-level rows. Lives in the area's
   `evidence/` directory (`champion/evidence/<experiment>/`,
   `w42/world_sampler_audit/`, …). Promotion is a deliberate act, not an
   accumulation.
3. **Everything else** (never committed) — `arena/results/` is gitignored;
   raw run dirs, per-hand CSVs, decision-record JSONLs, corpora, and trained
   heads are ephemeral and regenerable from seeds + shas.

## Reproducibility caveat, stated once

Arena runs are byte-deterministic only on the same device (MPS and CUDA RNG
paths differ); on other hardware, reproduction is statistical, which is
sufficient because every promoted claim carries a CI. This is why tier 2
keeps per-game rows for paired claims instead of keeping nothing.

## Boundary

Pre-policy run dirs remain tracked under `arena/results/` (gitignore does not
untrack them); they await a small sweep that checks which wiki pages
reference them before removal. New runs never enter git.

## Links

[[stage-0-closure]] [[jud-target-granularity]] [[partnership-research-gates]]
[[arena]] [[beads-to-gh-issues]]
