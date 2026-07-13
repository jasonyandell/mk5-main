---
title: Eval matrix — Bradley-Terry Elo tournament infra
kind: topic
first_seen: 2026-02-09
last_updated: 2026-02-09
status: complete
---

## What it is

By 2026-02-09 the project had several named players ([[zeb]] at multiple checkpoint
sizes, `random`, `heuristic`, `eq:n=10/50/100/500`) and a per-pair eval CLI that could
compare any two, but no way to fold all pairwise results into one ranking, and no way to
run a full round-robin without it taking most of a day serially. Three same-day commits
(all 2026-02-09, all `forge/modal_app.py` and `forge/zeb/eval/`) built that ranking layer:

- **`4f4220f`** (16:17) — `eval_matrix_remote` (all-pairs matchups on a Modal T4 GPU) and
  `eval_matrix_entry` (local CLI entrypoint). Also fixed a real bug: `_get_cached_model`
  was silently dropping its `weights_name` kwarg, which would have made every
  named-checkpoint eval secretly resolve to the same default weights.
- **`3a77bb6`** (17:07) — `compute_elo_ratings()` / `format_elo()`: a Bradley-Terry
  maximum-likelihood fit over a pairwise win-count matrix (`scipy.optimize.minimize`,
  Nelder-Mead), anchored so a chosen reference player (default [[expected-q-value|E[Q]]])
  sits at Elo 1600 and all others scale relative to it (standard `400/ln(10)` Elo
  convention). 77 lines of unit tests. W&B logging wired in, ratings keyed by training
  step.
- **`477ff00`** (19:13) — reworked the matrix runner to fan out `C(n,2)` pairs to separate
  T4 instances via Modal `starmap` (parallel, not serial), with progressive W&B logging —
  ratings recomputed and pushed after each pair finishes. A same-day follow-up,
  `f5cbdc1`, fixed a `starmap`-ordering bug in the progressive-logging path.

`forge/zeb/OVERVIEW.md` §"Modal Eval Matrix (GPU Tournaments)" is the canonical usage doc,
still present on disk.

## What it found

A same-day calibration snapshot (2026-02-09), explicitly labeled **noisy** in the doc
itself (N=100 games, not the full 5,000-game tournament):

| Player | Elo | Note |
|---|---|---|
| `eq:n=100` | 1600 | anchor |
| `zeb-large-belief` | **1579** | step 1660, ~1M training games — "already neck-and-neck with eq:n=100 (50% win rate head-to-head)" |
| `eq:n=500` | 1561 | |
| `random` | 1386 | floor |

Throughput by bottleneck player (100 games each, steady state): random/heuristic
~2000+ g/s, Zeb (single forward pass) ~48 g/s, `eq:n=100` ~11-13 g/s, `eq:n=500` ~3.0-3.5
g/s. Cost/time estimate for a full 8-player, 56-matchup, 5,000-game tournament: ~8.5
hours / ~$5 serially, collapsed to ~30 minutes wall-clock by parallelizing across T4s.

**No results artifact for the full 5,000-game tournament itself was located** in the repo
history — the doc frames the big run as a next step ("will pin down the precise
ranking"), not a completed one. The only concrete Elo numbers on the record are the N=100
calibration snapshot above; whether the full tournament ever ran is asserted, unverified.

## Reading the 1579-vs-1600 number correctly

This is the era's cleanest apples-to-apples comparison of [[zeb]] against
[[expected-q-value|E[Q]]] — both on the same Elo scale, both anchored to the same
reference. **Neck-and-neck is the honest read, not "beat."** The [[w42-jud-v1|jud v1]]
verdict (`afd4802`, 2026-07-06) later confirms that no Zeb-descended policy has ever been
recorded winning against E[Q] n=10 at pure play, in this window or since. See [[zeb]] and
[[vs-random-eval-is-suspect]].

## Terminal status

BUILT. All three commits' artifacts (`compute_elo_ratings`, `format_elo`,
`eval_matrix_entry`, `eval_matrix_remote`) exist in the repo today; no deletion found. No
wiki page records a subsequent parking or retirement of the tool specifically — its fate
after Feb 2026 is tied to [[zeb]]'s (parked as [[burl]]'s belief primitive 2026-04-18,
superseded by [[gus]] 2026-04-23), but the eval-matrix code itself was not part of either
event. Ranking infrastructure, not a consumption mechanism — it answers "which player is
stronger," not "what should a player do with E[Q]'s output." It made the era's other,
load-bearing finding ([[full-teacher-eq-experiment]]) measurable, not itself a finding
about [[candlewax|the wall]].

See [[zeb]] · [[full-teacher-eq-experiment]] · [[era4-zeb-era|conversation digest]].
