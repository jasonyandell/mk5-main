---
title: "Gus: modern learning and recoverable continuations"
kind: experiment
first_seen: 2026-10-18
last_updated: 2026-10-18
status: complete
---

## Question

Does a standard cosine learning-rate cooldown improve the trained
[[gus-tiny-gpt]] splat model's validation loss relative to the same number of
constant-rate updates? Can either trajectory stop during an update and resume
without changing its subsequent optimization?

Jason requested a complete reread of Cammarata, applicable improvements to the
existing net, interruptible/resumable training, and development of the
`gus-training` skill. This controlled continuation is separate from the original
[[gus-splat-pretraining-2026-10-18]] specimen and from [[gus-rlvr-silas-2026-10-18]].

## What the article actually changes

[Cammarata's full essay](https://already-clean.goodfire.com/essay.md), including
its appendices, attributes some earlier failures to weak observation tools and
some to learning. Its explicit architecture examples are residual connections
and LayerNorm; its model-training regularization recipe is not specified.
Regularization of optimized visualizations is a separate matter. The essay also
advocates watching native mechanisms develop, testing both poles and tracing
their connections. It does not establish that RLVR or any particular learning
schedule makes neurons cleaner.

Gus already uses pre-LN residual blocks, GELU, AdamW with matrix weight decay
0.1, and gradient clipping. These are retained. No convolutional structure,
SwiGLU, dropout, sparsity penalty, SAE or extra information is added. The model
still has 185,652 parameters and the same two output heads. This applies the
article's learning-first stance without inventing a missing architectural fix.

Cosine decay is an ordinary optimization choice from
[nanoGPT's trainer](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/train.py),
not a prescription from Cammarata. The
[cosine-annealing literature](https://arxiv.org/abs/1608.03983) motivates gradually
reducing step size; this continuation uses no warm restart and no fresh warmup,
because its AdamW moments already contain 195,000 updates. Better forecasts,
stronger play and cleaner native mechanisms remain separate empirical claims.

## Registered matched continuation

`plan.json` was written before either optimizer run. Both arms start from the
same best splat checkpoint at step 195,000, retaining all parameters, optimizer
moments and the next sampler state. Each performs exactly 2,000 **new** updates;
the old 200,000-update experiment is neither reopened nor overwritten.

- Constant arm: learning rate 0.001 throughout.
- Cosine arm: 0.001 at local update 1, smoothly reaching 0.0001 at update 2,000.
- Both: all seven original caches; batch 64, two layers/heads, width 64,
  matrix decay 0.1, AdamW betas 0.9/0.95, gradient clip 1, identical minibatches.
- Full validation every 500 steps and the early 250-step formation point.
  Local initialization and all formation points preserve full recovery state.
- Primary comparison: **final-model full-validation joint loss**. Within-arm
  best checkpoints are also retained, including the inherited initial model.
  The already-used test split is a diagnostic, never a new confirmation set.
- One matched pair, no tuning after the result, no H2H or strength promotion.
  No native-unit interpretation score is optimized or claimed.
- Each accelerator invocation goes through `cpu-slot`, outer cap 300 seconds,
  cooperative stop budget 240 seconds. An unfinished invocation resumes the
  same remaining update budget.

## Recovery contract

`lem/observer/splat/recovery.py` provides `StopRequest`, `capture_rng` and
`restore_rng`, usable by another trainer. SIGINT and SIGTERM set a request flag;
they do not interrupt mutation of parameters. The trainer finishes the current
optimizer update and any scheduled validation, then atomically replaces
`latest.pt`, retains a named pause checkpoint and writes `status.json`.
`--stop-file PATH` and `--stop-after-seconds N` request the same boundary.
The stop file remains until its owner removes it. SIGKILL cannot be handled.

Snapshots preserve model, AdamW moments, sampler, Python/NumPy/Torch randomness,
the active MPS/CUDA randomness, fixed schedule, step, selection history and W&B
identity. `--resume` loads that full state. `--fork-from FULL_CHECKPOINT` creates
a separately named continuation and records the parent hash; exported inference
weights alone are insufficient. A changed cosine horizon, task, model, data,
optimizer recipe or W&B identity is refused. Constant-rate runs retain their
existing explicit budget-extension capability; an extension is additional
training, not recovery of an interrupted fixed budget.

The validation sidecar is recoverable even when the process stops at local step
zero. A clean stop writes no completion report. Completed identical resumes
return the existing report. A service resume uses the saved W&B ID, while
random state is restored after tracking setup so setup cannot perturb training.

The `gus-training` skill documents these boundaries and retains `clean-neuron`
as the interpretation companion. A named unit still requires held-out examples
and matched causal tests under [[clean-neurons-42]], rather than an attractive
activation map or a change in loss.

## Completed comparison

Both registered arms completed 2,000 additional updates. Their initial model,
optimizer and sampler exactly match the source checkpoint; their final sampler
states also match. The original source checkpoint remains byte-identical.
The constant arm's best is its inherited initialization; the cosine arm's best
is local step 2,000. The primary final-model comparison is:

| Full validation measure | Inherited model | Constant final | Cosine final |
|---|---:|---:|---:|
| Joint loss, nats | 3.69056 | 3.70075 | 3.55979 |
| Raw all-play tile accuracy | 50.761% | 50.546% | 52.569% |
| Consequential make-probability MAE | 10.026 pp | 10.279 pp | 8.765 pp |
| Consequential histogram KL, nats | 0.36669 | 0.37157 | 0.30080 |
| Consequential teacher make/set regret | 3.012 pp | 3.040 pp | 2.703 pp |

This single matched continuation supports the cooldown as a forecasting
improvement for this trained net and corpus. The reused test diagnostic also
improves: cosine tile accuracy 52.529%, make-probability MAE 8.771 pp and teacher
regret 2.732 pp. Its regret remains above G0's 2.413 pp from the earlier paired
diagnostic. There is no H2H, promotion or native-unit interpretation claim.

Training plus validation/test took 73.93 seconds for the constant arm and
96.10 seconds for cosine, excluding setup and tracking sync; these are
descriptive times. All 16 focused splat/recovery tests pass, including real
SIGINT/SIGTERM delivered during an optimizer update, a stop file, exact
resumed CPU weights/moments/randomness/schedule, step-zero recovery, and W&B
identity through the tracking interface. Cross-device bitwise identity and
remote-service outage recovery are not established by those tests.

Live records: [constant continuation](https://forge.coreweave.com/wandb/jasonyandell-forge42/lem-observer/runs/ea987429)
and [cosine continuation](https://forge.coreweave.com/wandb/jasonyandell-forge42/lem-observer/runs/8e215fd5).
The evidence package contains the preregistration, source hashes, every local
formation checkpoint, initial/final measurements, comparison audit and tests.
The source article remains at its publisher; it is not relicensed as CC0 data.
