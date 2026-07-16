---
title: Run Artifacts Policy — claims in git, measurements ephemeral
kind: decision
first_seen: 2026-07-13
last_updated: 2026-07-15
status: active
---

Git holds **claims and the means to reproduce them, not the measurements
themselves**. Adopted 2026-07-13 during PR review of the research night, on
Jason's reviewer instinct that arena run data does not belong in git.
**Amended 2026-07-15** at Jason's direction after otis night 2 shipped ~7k
rows of per-game CSVs, a 487-line premium ledger, and a `.pt` head through
the old tier-2 loophole: game-level rows no longer enter git at all.

## The three tiers

1. **The wiki experiment page** (committed) — the numbers that matter, the
   registered bands, the verdicts, and the exact reproduction command with
   its code sha. This is the scientific record; a result that is not written
   here does not exist.
2. **A curated receipts bundle** (committed, small) — only for load-bearing
   claims, and only *aggregates*: `summary.json` per run, pooled-CI `.txt`
   receipts, grade files, plots, `*.metrics.json`. Lives in the area's
   `evidence/` directory (`champion/evidence/<experiment>/`,
   `w42/world_sampler_audit/`, …). Promotion is a deliberate act, not an
   accumulation. **Row-level data is never tier 2** — no `per_game.csv`, no
   `.jsonl` ledgers, no `.pt` heads, regardless of how load-bearing the
   claim is.
3. **Everything else** (never committed) — raw run dirs, per-game/per-hand
   CSVs, decision-record JSONLs, corpora, and trained heads. Enforced by
   `.gitignore` (`arena/results/`, `champion/evidence/**/per_game.csv`,
   `champion/evidence/**/*.jsonl`, `otis/models/*.pt`). Durable home for
   anything a claim may need re-examined: a HuggingFace dataset
   ([[huggingface-assets]]), uploaded via the `scripts/hf_publish/` pattern;
   otherwise regenerable from seeds + shas.

## Reproducibility caveat, stated once

Arena runs are byte-deterministic only on the same device (MPS and CUDA RNG
paths differ); on other hardware, reproduction is statistical, which is
sufficient because every promoted claim carries a CI. When a paired-CI claim
depends on game-level rows, those rows go to the HF evidence dataset and the
experiment page links them — they do not come back into git.

## Boundary

Closed 2026-07-15: everything under `arena/results/` (including the
pre-policy subdirs cited by [[champion-ladder]]) plus `otis/models/*.pt` was
mirrored to the public HF evidence dataset at tag `otis-night2-2026-07-15`
and untracked; the ladder's citations now point at the pinned HF URLs.
Moving data needs no custom tooling — the stock `hf` CLI upload/download
commands are documented at [[huggingface-assets]] § Moving evidence data.
New runs never enter git.

## Adding a new data-producing area

When a new area starts writing run outputs, its data directories get
gitignore entries **in the same commit that creates them** — before the
first run, not after review catches tracked CSVs.

## Links

[[stage-0-closure]] [[jud-target-granularity]] [[partnership-research-gates]]
[[arena]] [[beads-to-gh-issues]]
