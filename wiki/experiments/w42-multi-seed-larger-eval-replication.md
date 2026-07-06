---
title: w42 Multi-Seed Larger-Eval Replication
kind: experiment
first_seen: local-2026-05-02
last_updated: afd4802
status: complete
---

## Summary

[[w42]] has a five-seed replication of the raw, v0 strategy-tag, and rich-tag
small-model comparison on a larger held-out slice than the original baseline
reports.

- script: `w42/multi_seed_larger_eval_replication.py`
- aggregate artifacts:
  `w42/multi_seed_larger_eval_replication/pilot_2seed_e2800/`
- W&B group: `w42-csw6-31-multi-seed-larger-eval`
- feature sets: raw public state, raw plus v0 tags, raw plus v0 plus rich tags
- seeds: `42, 43, 44, 45, 46`
- eval slice: 2,800 decisions, 5x the earlier 560-row eval

The replication keeps the conclusion conservative. The raw-to-v0 and raw-to-rich
gains held across five seeds on this slice. The rich surface also beat v0 by a
modest paired mean-regret delta, but this is still a small research model with
four-epoch training, not a book-claim verdict or a promotion decision.

## Key Question

Do the small-slice findings from [[w42-raw-public-state-baseline]],
[[w42-v0-strategy-tags-baseline]], and [[w42-rich-tag-many-signal-probe]] survive
more random seeds and a materially larger held-out eval set?

## Method

| field | value |
|---|---|
| report owner bead | `t42-csw6.31` |
| evidence mode | multi-seed w42 model comparison with forge/Gus oracle labels |
| ruleset / score mode | existing [[gus]] corpus semantics |
| train corpus | `gus/data/corpus_train_chunk_0-99.pt`; `gus/data/corpus_train_chunk_100-199.pt` |
| eval corpus | `gus/data/corpus_train_chunk_9000-9099.pt` |
| train rows | 5,600 decisions |
| eval rows | 2,800 decisions |
| seeds | `42, 43, 44, 45, 46` |
| epochs / batch size | 4 / 256 |
| device | MPS |
| baseline policy | `E[Q] N=10` first worlds |
| W&B mode | online |
| HF artifact | not applicable |

The eval file name contains `train_chunk` because it comes from the existing Gus
generated-corpus inventory, but the replication used disjoint explicit paths for
training and held-out evaluation. No fresh game generation occurred.

The first two seed groups ran before Worker B's W&B-series commit landed on the
shared branch and record commit `4ff779d`. The remaining seed groups and the
aggregate summary record commit `86bdae6`. The final script lands in the worker
commit that adds this report.

## W&B Series

Every model/seed run logged per-epoch W&B series points for loss, regret, match,
near-tie rate, tail-regret rate, epoch seconds, and eval `n`, plus final/best
summary metrics. This directly addresses the requirement that longer iterative
runs expose trajectories rather than only final summaries.

Primary W&B runs:

| feature | seed | W&B run |
|---|---:|---|
| raw | 42 | `https://wandb.ai/jasonyandell-forge42/w42/runs/s0ldo98h` |
| v0 | 42 | `https://wandb.ai/jasonyandell-forge42/w42/runs/pcocleie` |
| rich | 42 | `https://wandb.ai/jasonyandell-forge42/w42/runs/m10m51xh` |
| raw | 43 | `https://wandb.ai/jasonyandell-forge42/w42/runs/nihgd7yq` |
| v0 | 43 | `https://wandb.ai/jasonyandell-forge42/w42/runs/e9iqizbw` |
| rich | 43 | `https://wandb.ai/jasonyandell-forge42/w42/runs/w7rt339c` |
| raw | 44 | `https://wandb.ai/jasonyandell-forge42/w42/runs/ve330jnv` |
| v0 | 44 | `https://wandb.ai/jasonyandell-forge42/w42/runs/fdh7il8b` |
| rich | 44 | `https://wandb.ai/jasonyandell-forge42/w42/runs/qyrrqo6o` |
| raw | 45 | `https://wandb.ai/jasonyandell-forge42/w42/runs/0q5kpe4z` |
| v0 | 45 | `https://wandb.ai/jasonyandell-forge42/w42/runs/w84cta35` |
| rich | 45 | `https://wandb.ai/jasonyandell-forge42/w42/runs/2u9lyndd` |
| raw | 46 | `https://wandb.ai/jasonyandell-forge42/w42/runs/yy60h0wz` |
| v0 | 46 | `https://wandb.ai/jasonyandell-forge42/w42/runs/kbq1smyl` |
| rich | 46 | `https://wandb.ai/jasonyandell-forge42/w42/runs/mfedph7d` |

## Results

`E[Q] N=10` on this 2,800-row eval slice reached 0.175 mean regret, 87.00%
match, and 92.89% near-tie rate. The small models remain far behind that boss
baseline.

Best-epoch metrics by feature:

| feature | seeds | mean regret | 95% CI half-width | match | tail `>=5` |
|---|---:|---:|---:|---:|---:|
| raw | 5 | 1.967 | 0.044 | 59.94% | 13.05% |
| v0 tags | 5 | 1.606 | 0.063 | 64.20% | 10.79% |
| rich tags | 5 | 1.515 | 0.050 | 64.89% | 10.38% |

Paired best-regret deltas, where negative is better for the left feature set:

| comparison | pairs | mean delta | 95% CI half-width | reading |
|---|---:|---:|---:|---|
| v0 minus raw | 5 | -0.361 | 0.064 | raw-to-v0 gain held |
| rich minus raw | 5 | -0.452 | 0.064 | raw-to-rich gain held |
| rich minus v0 | 5 | -0.091 | 0.086 | modest rich-over-v0 signal, still small-model evidence |

The larger-eval replication shifted the rich-vs-v0 result from "tiny single-seed
edge" to "modest five-seed edge on this slice." It did not change the larger
fact that `E[Q] N=10` remains much stronger.

## Interpretation

The v0/rich feature direction remains worth pursuing. The most responsible
reading is:

- v0 strategy tags improve the tiny model over raw public state on this
  replicated slice.
- rich tags improve over raw and show a modest additional gain over v0.
- the rich-over-v0 effect is not large enough to justify promotion by itself.
- no Winning 42 book claim moves status from this run.
- longer training, more held-out chunks, and regime-specific detector slices are
  still needed before final report language hardens.

This page should inform [[w42-strategy-tag-family-ablations]] and the eventual
final report, but should not replace claim-specific enumeration, oracle rollout,
or Burl trace review.

## Artifacts

| artifact | path | durable? |
|---|---|---|
| script | `w42/multi_seed_larger_eval_replication.py` | w42 |
| aggregate summary | `w42/multi_seed_larger_eval_replication/pilot_2seed_e2800/summary.json` | w42 |
| per-seed table | `w42/multi_seed_larger_eval_replication/pilot_2seed_e2800/per_seed_metrics.csv` | w42 |
| feature summary | `w42/multi_seed_larger_eval_replication/pilot_2seed_e2800/feature_summary.csv` | w42 |
| paired deltas | `w42/multi_seed_larger_eval_replication/pilot_2seed_e2800/paired_deltas.csv` | w42 |
| per-run manifests | `w42/multi_seed_larger_eval_replication/pilot_2seed_e2800/{raw,v0,rich}_s*/manifest.json` | w42 |
| W&B group | `w42-csw6-31-multi-seed-larger-eval` | yes |
| HF dataset/model/artifact | not applicable | not applicable |

The output directory name contains `pilot_2seed_e2800` because the work began as
a two-seed pilot and then extended in place to five seeds with `--skip-existing`.

## Exact Commands

Implementation check:

```bash
python -m py_compile w42/multi_seed_larger_eval_replication.py
```

Initial two-seed pilot:

```bash
python w42/multi_seed_larger_eval_replication.py \
  --seeds 42 43 \
  --train gus/data/corpus_train_chunk_0-99.pt gus/data/corpus_train_chunk_100-199.pt \
  --eval gus/data/corpus_train_chunk_9000-9099.pt \
  --train-limit 5600 \
  --eval-limit 2800 \
  --epochs 4 \
  --batch-size 256 \
  --output-dir w42/multi_seed_larger_eval_replication/pilot_2seed_e2800 \
  --wandb-mode online
```

Extension to the preferred five-seed run:

```bash
python w42/multi_seed_larger_eval_replication.py \
  --seeds 42 43 44 45 46 \
  --train gus/data/corpus_train_chunk_0-99.pt gus/data/corpus_train_chunk_100-199.pt \
  --eval gus/data/corpus_train_chunk_9000-9099.pt \
  --train-limit 5600 \
  --eval-limit 2800 \
  --epochs 4 \
  --batch-size 256 \
  --output-dir w42/multi_seed_larger_eval_replication/pilot_2seed_e2800 \
  --wandb-mode online \
  --skip-existing
```

Fresh one-shot equivalent:

```bash
python w42/multi_seed_larger_eval_replication.py \
  --seeds 42 43 44 45 46 \
  --train gus/data/corpus_train_chunk_0-99.pt gus/data/corpus_train_chunk_100-199.pt \
  --eval gus/data/corpus_train_chunk_9000-9099.pt \
  --train-limit 5600 \
  --eval-limit 2800 \
  --epochs 4 \
  --batch-size 256 \
  --output-dir w42/multi_seed_larger_eval_replication/five_seed_e2800 \
  --wandb-mode online
```

## Claim-Ledger Impact

no claim-ledger change

This run compares model feature sets. It supports continuing the strategy-tag
modeling direction, but it does not test any specific Winning 42 claim with
enumeration, oracle counterfactuals, calibrated belief checks, or Burl trace
review.

## Next Checks

- Repeat with 8 epochs to match the original baseline horizon.
- Evaluate on more held-out chunks or a dedicated eval corpus with a clearer
  manifest name than `train_chunk_9000-9099`.
- Add aggregate W&B summary runs or tables once the W&B standard page settles
  the preferred dashboard shape.
- Use regime-specific detector slices before using rich-over-v0 as claim-ledger
  evidence.

## Links

[[w42]] | [[w42-lab-infrastructure]] | [[w42-wandb-series-logging-standard]] |
[[w42-raw-public-state-baseline]] | [[w42-v0-strategy-tags-baseline]] |
[[w42-rich-tag-many-signal-probe]] | [[w42-strategy-tag-family-ablations]] |
[[winning42-strategy-measurement]]
