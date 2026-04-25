# Q-Mean Fallback Finding

**Date:** 2026-04-23
**Adapter:** `gus/adapters/v3_consistency_10000g.pt`
**Eval:** `gus/data/corpus_eval_20.pt` (20 held-out games, 560 decisions)
**Status:** Strong enough to investigate. Not yet a promoted result.

**Update after harness cleanup:** `sample_worlds()` was reseeding from
`rng.initial_seed()` on every call, so the caller's RNG state did not advance
across batches. Fixed in `gus/model/sample_worlds.py`. The corrected sampler
makes pure Q-mean more modest, but the router result survives.

---

## TL;DR

The previous "LAMIR-1 does not beat direct π_me" conclusion missed a useful
variant: averaging `Q_head` over many belief-sampled worlds (`lamir1-K`) does
beat direct π on mean regret in this eval, even though bot-match barely moves.

Most importantly: on direct π blunders, Q-mean cuts regret roughly in half.

| policy | mean regret | bot-match |
|---|---:|---:|
| direct π_me | 0.551 | 76.1% |
| Q-mean, K=100, seed 42 | 0.505 | 75.5% |
| Q-mean, K=200, seed 42, corrected RNG | 0.517 | 75.9% |
| Q-mean, K=500, seed 42, corrected RNG | 0.500 | 75.7% |

Direct blunders (`regret >= 8`) are the headline:

| slice | n | direct regret | Q-mean K=200 regret |
|---|---:|---:|---:|
| direct blunders >=8 | 8 | 11.42 | 5.94 mean (5.28-8.55 range) |
| direct big misses >=4 | 29 | 7.11 | 3.87 mean (3.64-4.49 range) |

This is exactly the shape a deployable fallback should have: it does not
dramatically improve label agreement, but it reduces the cost of the worst
mistakes.

---

## How This Got Lost

`gus/eval/eval_lamir1.py` reports three policies:

- `direct`
- `lamir1-argmax`
- `lamir1-K`

But the final verdict block only evaluates `lamir1-argmax`, not `lamir1-K`.
So a run can show `lamir1-K` beating direct π and still end with:

```text
BELOW SUCCESS BAR -- delta vs baseline: +0.0954
```

That verdict is true for `lamir1-argmax`, but false for the K-sampled policy.
The script should be updated to report the best tested mode, or at least give
`lamir1-K` its own verdict.

---

## Reproduction Commands

Baseline comparison:

```bash
python -u gus/eval/eval_lamir1.py \
  --adapter gus/adapters/v3_consistency_10000g.pt \
  --eval gus/data/corpus_eval_20.pt \
  --k 200 \
  --device cpu \
  --seed 42
```

Observed after the RNG fix:

```text
direct       regret=0.5510  bot-match=76.071%
lamir1-K     regret=0.5173  bot-match=75.893%
```

The hand-size sanity check passed: inferred opponent hand sizes from
`belief_mask` matched true remaining hand sizes for all 560 eval decisions.
So the Q-mean result is not coming from impossible sampled hand sizes.

---

## Seed Sweep

`K=200` with the corrected sampler was stable across sampled-world RNG seeds:

| seed | Q-mean regret | Q-mean bot-match |
|---:|---:|---:|
| 0 | 0.520 | 75.4% |
| 1 | 0.476 | 75.9% |
| 2 | 0.509 | 75.2% |
| 42 | 0.517 | 75.9% |
| 99 | 0.506 | 75.4% |

Direct π baseline for the same eval is `0.551`. Pure Q-mean is now best
understood as a modest but repeatable improvement, not a dramatic replacement.
At K=500, seed 42 improves further to `0.500`, so the K-asymptote is still
worth measuring.

This is a small eval, so this is not proof. But the effect is not a single
lucky sample.

---

## Router Shape

Pure Q-mean helps regret but sometimes damages easy direct-π wins. The more
interesting shape is a fallback:

1. Run direct π.
2. Run Q-mean on belief-sampled worlds.
3. If direct π and Q-mean agree, commit.
4. If they disagree, route to Q-mean only when direct π is uncertain.

The simplest gate was `pi_peak < 0.5`.

Across five sampled-world seeds (`K=200`, corrected sampler):

| seed | pure Q-mean regret | router regret | routed fraction |
|---:|---:|---:|---:|
| 0 | 0.520 | 0.431 | 6.4% |
| 1 | 0.476 | 0.413 | 6.1% |
| 2 | 0.509 | 0.413 | 6.1% |
| 42 | 0.517 | 0.460 | 6.2% |
| 99 | 0.506 | 0.405 | 6.4% |

Mean router regret: `0.424` (range `0.405-0.460`) while routing only `6.25%`
of decisions.

The threshold was chosen during this exploratory pass, so treat it as
eval-tuned. But the direction is important: routing only ~6-7% of decisions
gets regret near `0.41`, far below the direct `0.551` baseline and below the
previous oracle-fallback target band.

Other candidate gates from the corrected seed sweep:

| gate | routed | regret |
|---|---:|---:|
| direct != Q-mean and `pi_peak < 0.5` | 6.25% | 0.424 mean |
| direct != Q-mean and `pi_entropy > 1.0` | 6.50% | 0.454 mean |
| direct != Q-mean and either condition | 7.11% | 0.429 mean |

---

## Train-to-Eval Learned Router

The hand gate above is eval-tuned, so the next check trained a small router on
`corpus_train_100.pt` and evaluated on `corpus_eval_20.pt`.

Command:

```bash
python -u gus/eval/learn_qmean_router.py \
  --adapter gus/adapters/v3_consistency_10000g.pt \
  --train gus/data/corpus_train_100.pt \
  --eval gus/data/corpus_eval_20.pt \
  --k 100 \
  --eval-seeds 0,1,2,42,99 \
  --device cpu
```

Target: direct π would blunder (`regret >= 8`) and Q-mean would not.

Result across five sampled-world seeds:

| policy | regret | blunders/seed | big misses/seed | bot-match | routed | fixed blunders | new blunders |
|---|---:|---:|---:|---:|---:|---:|---:|
| direct π_me | 0.551 | 8.00 | 29.00 | 76.07% | - | - | - |
| pure Q-mean | 0.489 | 7.60 | 22.60 | 75.89% | - | - | - |
| learned route top 5% | 0.434 | 4.40 | 20.80 | 76.82% | 5.00% | 3.60 | 0.00 |
| learned route top 7% | 0.440 | 4.20 | 21.20 | 76.64% | 6.96% | 4.00 | 0.20 |

This is the cleanest version of the finding so far:

- The router was trained on a separate 100-game corpus.
- It routes only 5-7% of decisions.
- It cuts eval blunders from 8 to roughly 4.
- The 5% cutoff introduced zero new blunders across the five sampled-world
  seeds.

Top learned features were `qmean_minus_direct_q`, `pi_margin`,
`qmean_margin`, and `pi_peak`, so the router is not just a plain low-confidence
detector. It is using disagreement shape between direct π and the sampled-world
Q estimate.

---

## How Much Lower Can This Family Go?

Q-mean is only one aggregation over sampled-world Q values. A follow-up script
tested mean, quantiles, lower/upper confidence bounds, and per-world argmax
vote:

```bash
python -u gus/eval/eval_qworld_variants.py \
  --adapter gus/adapters/v3_consistency_10000g.pt \
  --eval gus/data/corpus_eval_20.pt \
  --k 100 \
  --seeds 0,1,2,42,99 \
  --device cpu
```

The individual variants are not good replacements. They fix extra direct
blunders but introduce many new ones. However, an oracle selector over direct π
plus the variant pool shows real headroom:

| K | oracle-pool regret | oracle-pool blunders/seed | direct blunders/seed | fixed blunders/seed |
|---:|---:|---:|---:|---:|
| 100 | 0.178 | 2.60 | 8.00 | 5.40 |
| 200 | 0.185 | 2.60 | 8.00 | 5.40 |

So the sampled-world Q family can theoretically reduce the 20-game eval from
8 direct blunders to about 2-3. The current learned Q-mean router captures most
of the safe/easy gain, but not all of the available headroom.

A first learned multi-candidate router over the whole variant pool did **not**
beat the Q-mean router. With a regret-delta objective on seed 0, its best
learned-prefix cutoff still bottomed out at 4 blunders, while the oracle pool
for that seed reached 3. That makes the current bottleneck explicit:

> Candidate generation has more headroom than Q-mean alone, but candidate
> selection is now the hard problem.

---

## What Q-Mean Fixes

On the decisions where direct π and Q-mean disagree, pure Q-mean is a mixed
second opinion. The router is valuable because it learns when to listen to
that second opinion.

Exploratory pre-fix seed-42 diagnostics were:

| subset | n | direct regret | Q-mean regret |
|---|---:|---:|---:|
| Q-mean better | 37 | 3.93 | 0.16 |
| Q-mean worse | 35 | 0.06 | 2.67 |
| about tied | 16 | 0.35 | 0.34 |

The exact counts should be recomputed with the corrected sampler before being
treated as final, but the qualitative pattern still matches the corrected
router sweep: pure Q-mean is not universally better. It is a high-leverage
second opinion.

That is why the router matters.

---

## Concrete Example

Rendered hand:

```text
scratch/gus_game900009_viz.md
```

Seed `900009`, declaration `9`, decision `12`:

- Direct π picks slot `6` (`6-0`)
- Oracle best is slot `1` (`3-2`)
- Direct regret: `18.44`
- Q-mean K=200 picks slot `1`
- Q-mean regret: `0.00`

The brainstate is exactly what a fallback should notice:

```text
Student π_me: 3-2:0.13  5-3:0.21  5-4:0.29  6-0:0.38
V_head estimate: -1.15
Oracle E[Q] at π pick: -9.99
Oracle best: +8.45
```

Direct π is split and low-confidence. Q-mean corrects the blunder.

---

## Relationship To Earlier Conclusions

This does not fully refute the LAMIR-1 caution. `lamir1-argmax` still loses:

```text
direct          0.551
lamir1-argmax   0.646
```

The original root-cause story also still matters: distilled scalar values are
noisy, and Q-head rollouts can damage easy decisions.

But the stronger conclusion should be revised:

> Direct π is still the best default policy, but belief-sampled Q-mean is a
> useful fallback signal for uncertain direct-π decisions and materially cuts
> the blunder tail.

This is closer to the detect-and-route architecture than to full LAMIR:

```text
direct π fast path
  -> if confident: commit
  -> if uncertain and Q-mean disagrees: route to Q-mean
```

No oracle fallback required.

---

## Caveats

- Eval is only 20 games / 560 decisions.
- The `pi_peak < 0.5` threshold was found on this eval; it needs a separate
  validation split or cross-validation over train chunks.
- The train-to-eval learned router is more convincing than the hand gate, but
  it still has only 8 eval blunders to measure against.
- `sample_worlds` is Python-looped and CPU-ish. K=200 is fine for analysis,
  but a product path would want vectorization or a smaller K/gate.
- Q-mean still introduces regressions. Pure replacement is not the right
  shape.
- The multi-aggregator pool has lower oracle blunder count, but the first
  learned selector did not realize that extra headroom.
- The current `eval_lamir1.py` verdict is misleading for K-sampled runs.

---

## Next Experiments

1. **Patch the eval script verdict.**
   Report direct, argmax-world, K-world, and best mode. Add disagreement
   diagnostics for `lamir1-K`, not just `lamir1-argmax`.

2. **Promote the router eval to a script.**
   Evaluate gates:
   - `pi_peak < t`
   - `pi_entropy > t`
   - direct/Q-mean disagreement
   - Q-mean margin
   - V/π consistency using deployable model features only

3. **Cross-validate the gate.**
   Learn thresholds on train chunks, evaluate on `corpus_eval_20.pt`.
   Avoid tuning on the 560 held-out decisions.

4. **Improve the second-opinion selector.**
   The variant-pool oracle reaches ~2.6 blunders/seed, while the learned
   Q-mean router reaches ~4.2-4.4. Inspect the extra fixable blunders and train
   a selector that models both rescue value and new-blunder risk.

5. **Sweep K for the router, not just pure Q-mean.**
   The pure policy improves from K=100 to K=500, but the router might only
   need K=50 or K=100 if it routes few decisions. The variant-pool oracle did
   not improve from K=100 to K=200 on this eval.

6. **Inspect fixed and broken cases.**
   The top Q-mean fixes and regressions are both informative. They should
   become a small qualitative set for the reasoning/trajectory work.

---

## Side Finding: Real-Drama Filter Needs `n_legal_actions >= 2`

The corrected drama atlas (`drama_atlas_v2.parquet`) marks many forced moves
as real drama because `marginal_eq_gap` is `0` for single-legal decisions.

Counts:

| set | decisions |
|---|---:|
| raw real-drama flags | 67,605 |
| single-legal / forced inside raw flags | 31,421 |
| filtered with `n_legal_actions >= 2` | 36,184 |

This does not affect the Q-mean finding directly, but it matters for the
next reasoning-as-product pass. Any "real drama" view should exclude forced
moves.
