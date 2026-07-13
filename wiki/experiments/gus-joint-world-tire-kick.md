---
title: Gus Joint-World Tire-Kick
kind: experiment
first_seen: 2026-04-20
last_updated: 2026-04-20
status: active
---

## Summary

Single-game validation of the joint-world tensor infrastructure on eval seed 900000.
Confirms the (world_hands, q_per_world) signal is real and sample-size requirements
are feasible on Apple Silicon. (commit message @ 31e10ef)

## Setup

- **Seed**: 900000 (held-out eval set)
- **Sample sizes**: M ∈ {10, 100, 2400-adaptive}
- **Platform**: MPS (M5 Max, no CUDA)
- **Goal**: Validate that belief-Q correlations converge and that the joint tensor
  carries a distillable signal before committing to a 100-game corpus generation

## Results

| M | Finding |
|---|---|
| 10 | r=0.79 for top belief-Q correlations — **sampling-noise artifact** |
| 100 | Correlations begin to stabilize |
| ~100-500 | Convergence frontier for top-10 (domino, seat) correlations |
| Converged | r≈0.2-0.4 for strongest belief-Q relationships |
| Any M | Q-std is position-intrinsic and stable (~23 at decision 0) |
| Adaptive SEM<0.5 | ~10s/game on MPS; file ~6 MB/game |

(commit message @ 31e10ef)

## Key conclusions

1. **M=10 is insufficient** for belief-Q correlation estimation — r=0.79 is inflated by
   noise, not signal. M≥100 required.
2. **The signal is real but moderate.** Converged r≈0.2-0.4 means meaningful but not
   overwhelming correlation — learnable by a distilled student head.
3. **Q-std is a position property, not a sample-size artifact.** Stable at ~23 across
   all M values at decision 0. This is intrinsic game state uncertainty.
4. **Adaptive SEM<0.5 is practical.** 10s/game on MPS means a 100-game corpus takes
   ~17 minutes locally. The plan to generate on M5 Max is validated.

## Infrastructure validated

- `forge/eq/generate` with `--save-joint-worlds --adaptive` flag produces correct tensors
- MPS fallback path works (no CUDA required)
- File size (~6 MB/game) is manageable; 100-game corpus ≈ 600 MB

## Next

v0 training: belief-only head on 100-game corpus (seeds 0-99), evaluated on held-out
(seeds 900000-900099). Success bar: top-1 > 50% (beats [[zeb]]'s 39% hidden-only baseline).

## Related pages

[[gus]] · [[joint-world-tensor]] · [[zeb]] · [[forge]] · [[sources/31e10ef]]
