---
title: Gus Q-mean router (belief-sampled second opinion, no oracle)
kind: experiment
first_seen: 3ad63f9f
last_updated: pending-this-ingest
status: active
---

## Summary

A no-oracle router over {direct π, belief-sampled Q-mean} that works: routing
only ~5-7% of decisions cuts the eval blunder tail from 8 to ~4 per seed and
mean [[regret-eval]] regret from 0.551 to ~0.42-0.43, with the learned router's
5% cutoff introducing **zero new blunders** across five sampled-world seeds
(gus/analysis/qmean_router_findings.md @ 233b7dc5;
also recorded as gus/PRACTICALITIES.md §23 @ 233b7dc5). Adapter: `gus/adapters/v3_consistency_10000g.pt`;
eval: `gus/data/corpus_eval_20.pt` (20 held-out games, 560 decisions); 2026-04-23.

This refines [[router-reality-check]]: every non-oracle *replacement* hurts, but a
belief-sampled Q-mean *second opinion* — consulted only where direct π is uncertain
and the two disagree — helps. It is the [[detect-and-route]] shape with no oracle
fallback required.

## The finding

`lamir1-K` — averaging `Q_head` over K belief-sampled worlds — had been dismissed
because `gus/eval/eval_lamir1.py`'s verdict block only evaluated `lamir1-argmax`
(which does lose: 0.646 vs direct 0.551). The K-sampled variant beats direct π on
mean regret and, on direct-π blunders (regret ≥ 8), roughly halves regret
(11.42 → 5.94 at K=200).

| policy | mean regret | bot-match |
|---|---:|---:|
| direct π_me | 0.551 | 76.1% |
| Q-mean K=200 (corrected RNG, seed 42) | 0.517 | 75.9% |
| Q-mean K=500 (corrected RNG, seed 42) | 0.500 | 75.7% |

K=200 was stable across sampled-world seeds {0, 1, 2, 42, 99}: regret 0.476-0.520.
Pure Q-mean is a modest, repeatable improvement — not a replacement; it damages
some easy direct-π wins.

## Router results

Hand gate (`direct != Q-mean` ∧ `pi_peak < 0.5`, eval-tuned): mean regret 0.424
(range 0.405-0.460) across five seeds, routing 6.25% of decisions.

Learned router, trained on a **separate** 100-game corpus (`corpus_train_100.pt`),
target = "direct π would blunder (regret ≥ 8) and Q-mean would not", evaluated on
`corpus_eval_20.pt` across five seeds:

| policy | regret | blunders/seed | routed | new blunders |
|---|---:|---:|---:|---:|
| direct π_me | 0.551 | 8.00 | — | — |
| pure Q-mean | 0.489 | 7.60 | — | — |
| learned route top 5% | 0.434 | 4.40 | 5.00% | 0.00 |
| learned route top 7% | 0.440 | 4.20 | 6.96% | 0.20 |

Top learned features: `qmean_minus_direct_q`, `pi_margin`, `qmean_margin`,
`pi_peak` — the router uses disagreement *shape* between direct π and the
sampled-world Q estimate, not just low confidence. Compare [[blunder-detector]],
which was a pure confidence detector.

## Headroom and the standing conclusion

An oracle selector over direct π plus a pool of sampled-world Q aggregations
(mean, quantiles, confidence bounds, per-world argmax vote;
`gus/eval/eval_qworld_variants.py`) reaches ~2.6 blunders/seed (regret 0.178 at
K=100) — but a first learned multi-candidate selector over that pool did **not**
beat the Q-mean router (still bottomed at 4 blunders on seed 0 vs the pool
oracle's 3). Standing conclusion:

> Candidate generation is not exhausted — candidate selection is now the hard
> problem.

## The RNG fix

`sample_worlds()` was reseeding from `rng.initial_seed()` on every call, so the
caller's RNG state never advanced across batches. Fixed in
`gus/model/sample_worlds.py`. The corrected sampler makes pure Q-mean more modest
(0.505 → 0.517 at seed 42) but the router result survives. The hand-size sanity
check passed: inferred opponent hand sizes from `belief_mask` matched true
remaining hand sizes on all 560 eval decisions.

## Reproduction

```bash
# baseline + Q-mean K comparison
python -u gus/eval/eval_lamir1.py \
  --adapter gus/adapters/v3_consistency_10000g.pt \
  --eval gus/data/corpus_eval_20.pt \
  --k 200 --device cpu --seed 42

# train-to-eval learned router
python -u gus/eval/learn_qmean_router.py \
  --adapter gus/adapters/v3_consistency_10000g.pt \
  --train gus/data/corpus_train_100.pt \
  --eval gus/data/corpus_eval_20.pt \
  --k 100 --eval-seeds 0,1,2,42,99 --device cpu

# aggregation-variant pool + oracle headroom
python -u gus/eval/eval_qworld_variants.py \
  --adapter gus/adapters/v3_consistency_10000g.pt \
  --eval gus/data/corpus_eval_20.pt \
  --k 100 --seeds 0,1,2,42,99 --device cpu
```

Concrete example (seed 900009, decl 9, decision 12): direct π picks 6-0 with a
split, low-confidence distribution (peak 0.38) for regret 18.44; Q-mean K=200
picks the oracle-best 3-2 for regret 0.00.

## Caveats

- 20 games / 560 decisions; only 8 direct blunders to measure against.
- The `pi_peak < 0.5` hand gate was tuned on this eval; the learned router is
  the convincing version.
- `sample_worlds` is Python-looped; K=200 is analysis-grade, not product-grade.
- The `eval_lamir1.py` verdict block still reports only `lamir1-argmax`
  (misleading for K-sampled runs) as of this finding.

## Links

[[gus]] [[router-reality-check]] [[detect-and-route]] [[blunder-detector]]
[[regret-eval]] [[belief-co-train]]
