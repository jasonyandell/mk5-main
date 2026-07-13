---
title: Joint-World Tensor
kind: entity
first_seen: 2026-04-20
last_updated: 2026-04-20
status: active
---

## What it is

The joint-world tensor is a per-decision data artifact saved by `forge/eq/generate/` when
`--save-joint-worlds` is passed. It captures `(world_hands, q_per_world)` — the sampled
hidden-hand realizations and the oracle's Q value for each legal action in each world.
(commit message @ 31e10ef)

## Schema

```python
# Per decision in DecisionRecordGPU:
world_hands   # [M, 3, 7]  — opponent hands per sampled world
q_per_world   # [M, 7]     — oracle Q per action per world

# Per game in GameRecordGPU:
hands         # full initial deal (all 4 seats) → belief_head truth target
```

Both fixed-sampling (M constant) and adaptive-convergence (SEM < threshold) paths are
supported. Adaptive+posterior path still skipped at this frontier. MPS fallback added so
tire-kicking runs locally on Apple Silicon. (commit message @ 31e10ef)

**This is schema v1.** Schema v2 (`dcd9365`) added `oracle_softmax_per_seat [4, 7]`,
`legal_mask_per_seat [4, 7]`, and `voids_per_seat [4, 3, 8]` to support [[pi-opp-head]]
training — see that page for the v2 fields. Those fields were never folded back into this
page's schema block above; v1 loaders remain backwards compatible with v2 corpora.

## Tire-kick validation (seed 900000)

| M | Finding |
|---|---|
| 10 | r=0.79 belief-Q correlations were sampling-noise artifacts |
| 100–500 | Top (domino, seat) correlations converge; settle at r≈0.2–0.4 |
| Adaptive (SEM<0.5) | ~10s/game on MPS; file ~6 MB/game; Q-std stable (~23 at decision 0) |

Q-std is position-intrinsic and stable across sample sizes — it measures genuine
decision-point volatility, not estimation noise. (commit message @ 31e10ef)

## Why it matters for LAMIR

[[gus]]'s world-conditioned Q head (`Q_head`) is trained on cached `q_per_world` values.
At inference, when an opponent plays a card and the belief distribution updates, the
look-ahead tree can re-weight: `Σ_m w_m(new_belief) · Q_m` — using the cached world Q
values rather than re-querying the [[forge]] oracle. This is the piece that makes
LAMIR-style continual-resolving look-ahead cheap enough to run at game time.
See [[topics/lamir1]]. (gus/BUILD_PLAN.md @ 31e10ef)

## Generation command

```bash
python -u -m forge.eq.generate \
    --start-seed 0 --n-games 100 \
    --adaptive \
    --min-samples 100 --max-samples 50000 \
    --sem-threshold 0.5 --batch-size 200 \
    --save-joint-worlds \
    -o gus/data/corpus_100.pt
```

Expected: ~10s/game on MPS, ~17 min for 100 games, ~600 MB file.

## Links

[[gus]] [[forge]] [[lamir1]] [[pi-opp-head]] [[student-distillation]]
