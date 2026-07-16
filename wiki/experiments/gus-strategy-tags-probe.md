---
title: Gus Strategy Tags Probe
kind: experiment
first_seen: 2026-04-30
last_updated: 2026-07-15
status: superseded
---

## Summary

Explicit, human-legible public-state strategy tags were promoted from a scratch idea
into a durable Gus probe. The tags do **not** beat `E[Q] N=10`, but they materially
improve a tiny supervised public-state policy over the same architecture without tags.

The strongest result so far: on 28k early-decision examples, strategy tags reduce mean
regret from **2.012 → 1.181 Q-points** versus the base tiny model. `E[Q] N=10` remains
the boss at **0.167 Q-points** on the same eval slice.

## Motivation

The Winning 42 strategy-book harvest produced a concrete hypothesis: book concepts
should become detector features and concept buckets, not hard-coded strategy rules.
This probe asks whether cheap strategy tags buy sample efficiency over the raw public
hand/play sequence.

## Code

- `gus/model/strategy_features.py`
- `gus/eval/strategy_probe.py`

`JointWorldFullDataset` can emit these opt-in fields:

- `strategy_features`: 68 global public-state features
- `strategy_action_features`: 7 × 32 action-local features

The main Gus training scripts keep the default `include_strategy_features=False`; the probe
sets it true so ordinary training does not pay the extra feature-compute cost.

## Feature Shape

Global tags include:

- declaration, phase, trick position
- current hand shape, trump/called-suit/double/count summaries
- legal-action summary
- played/live count summary
- void summaries
- visible pip coverage
- current trick pressure and current winner relation
- own pip coverage and unseen count-by-pip estimates

Action tags include:

- legal/called/trump/double/count identity
- rank and pip identity
- whether the action follows/beats the current trick
- point-dump and trump-in flags
- live and higher-live tiles in the action's suit
- live count pressure in the action's suit
- pip coverage and count risk
- whether the tile is protected by a same-pip double in hand
- count donation to partner/opponent-currently-winning flags

These are intentionally cheap public-state features, not oracle features.

## Runs

### 100-game full-decision probe

Command shape:

```bash
python -m gus.eval.strategy_probe \
  --train-limit 4096 --eval-limit 2048 --epochs 8 --batch-size 256 \
  --d-model 96 --n-heads 4 --n-layers 2 --ff-dim 192 --random-eq-worlds
```

Actual corpus: [corpus_train_100.pt](https://huggingface.co/datasets/jasonyandell/texas-42-joint-world-corpus/blob/main/corpus_train_100.pt) / [corpus_eval_20.pt](https://huggingface.co/datasets/jasonyandell/texas-42-joint-world-corpus/blob/main/corpus_eval_20.pt), so `train=2800`, `eval=560`. Bulk corpora live on HF ([[huggingface-assets]]).

| policy | mean regret | match | near-tie |
|---|---:|---:|---:|
| `E[Q] N=10` | 0.191 | 86.25% | 92.68% |
| base tiny model | 2.687 | 60.18% | 68.21% |
| global-tags only | 2.478 | 61.43% | 69.11% |

Interpretation: global tags alone had weak but positive signal.

### 10k early-decision probe

Command shape:

```bash
python -m gus.eval.strategy_probe \
  --train gus/data/corpus_v2_train_0-9_d0-9.pt ... gus/data/corpus_v2_train_90-99_d0-9.pt \
  --eval gus/data/corpus_v2_eval.pt \
  --train-limit 10000 --eval-limit 2048 --epochs 8 --batch-size 256 \
  --d-model 96 --n-heads 4 --n-layers 2 --ff-dim 192 --random-eq-worlds
```

Actual corpus: `train=10000`, `eval=560`.

| policy | mean regret | match | near-tie |
|---|---:|---:|---:|
| `E[Q] N=10` | 0.167 | 87.86% | 93.04% |
| base tiny model | 2.225 | 63.04% | 70.00% |
| strategy tags | 1.626 | 68.04% | 76.61% |

Delta: strategy tags improve over base by **−0.599 Q-points**.

### 28k early-decision probe

Command shape:

```bash
python -m gus.eval.strategy_probe \
  --train gus/data/corpus_v2_train_0-9_d0-9.pt ... gus/data/corpus_v2_train_90-99_d0-9.pt \
  --eval gus/data/corpus_v2_eval.pt \
  --train-limit 0 --eval-limit 2048 --epochs 12 --batch-size 256 \
  --d-model 96 --n-heads 4 --n-layers 2 --ff-dim 192 --random-eq-worlds
```

Actual corpus: `train=28000`, `eval=560`.

| policy | mean regret | match | near-tie |
|---|---:|---:|---:|
| `E[Q] N=10` | 0.167 | 87.86% | 93.04% |
| base tiny model | 2.012 | 65.36% | 72.86% |
| strategy tags | **1.181** | 69.82% | 79.64% |

Delta:

- Strategy vs base: **−0.831 Q-points**
- Strategy vs `E[Q] N=10`: **+1.014 Q-points**

## Interpretation

1. Strategy tags contain real policy signal.
2. Action-local tags matter much more than global tags.
3. More cheap signals helped only slightly beyond the first action-tag jump; the main unlock was letting the model attend to action-level facts.
4. `E[Q] N=10` is still a very strong small-compute baseline on this eval slice.
5. The next useful question is not just aggregate regret, but concept buckets: count-dump risk, trump pressure, off-risk/protection, donation windows, pounce windows, and walker/endgame states.

## Next (executed as w42)

- Add concept-bucket reporting to `gus/eval/strategy_probe.py`.
- Train/eval on full-decision chunks, not only d0-9 early-decision v2 shards.
- Add deeper book-derived tags: off protection, partner donation, setter pounce, effective walker, 84 preservation.
- Run multiple random `E[Q] N=10` samples per decision to estimate the boss's variance.

This entire "Next" list was executed the following day as a top-level workstream: `a417295`,
promoted explicitly in `2d3e9d7` ("Promote w42 to top-level workstream"). This page is the
**origin probe** for [[w42]] — the seed the whole wave/phase structure grew from — and was
unlinked from it until this correction. Superseded by the w42 strategy-tag program; go to
[[w42]] for the current frontier.

## Links

[[gus]] · [[student-distillation]] · [[regret-eval]] · [[gus-probe]] · [[gus-v3-consistency-full-run]] · [[w42]]
