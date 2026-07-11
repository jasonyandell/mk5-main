# E[Q] Training Pipeline

Knowledge lives in the wiki — see
[wiki/topics/expected-q-value.md](../../wiki/topics/expected-q-value.md) for what E[Q]
is, its history, and the consumption foot-guns (points-not-logits, backtracking-not-
rejection sampling, Q-model-vs-logit choice, adaptive "drunken master" sampling).
This file is the CLI and component reference.

## 🚨 CUDA/WSL: CPU = FULL STOP 🚨

If you are running in WSL and `torch.cuda.is_available()` is **False** (or you see
CUDA init errors like `cudaGetDeviceCount`), this is almost always a WSL/CUDA/driver
problem and requires human intervention. Do **not** fall back to CPU when validating
performance — CPU timings are not a proxy for GPU performance. Fix CUDA first.

## Quick start

```bash
# Basic usage (GPU, gap-filling, per-seed files)
python -m forge.cli.generate_eq_continuous \
    --checkpoint forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt

# Recommended for training data: adaptive sampling + posterior weighting
python -m forge.cli.generate_eq_continuous \
    --checkpoint model.ckpt \
    --adaptive --posterior --posterior-window 4

# Preview / inspect
python -m forge.cli.generate_eq_continuous --checkpoint model.ckpt --dry-run
python -m forge.eq.viewer data/eq-games/train/seed_00000042.pt
```

Output: per-seed `.pt` files in `data/eq-games/{train,val,test}/`, routed by
`seed % 1000` (same 90/5/5 split as Stage 1). Atomic saves, Ctrl+C safe, gap-filling.

## CLI reference

### Core options

| Option | Description | Default |
|--------|-------------|---------|
| `--checkpoint PATH` | **REQUIRED**: Stage 1 Q-value model checkpoint | - |
| `--start-seed N` | Start gap-filling from this seed | 0 |
| `--batch-size N` | GPU batch size (games per batch) | 32 |
| `--n-samples N` | Worlds sampled per decision | 50 |
| `--device cuda\|cpu` | Device to use | cuda |
| `--output-dir PATH` | Output directory | data/eq-games |
| `--dry-run` | Show missing seeds without generating | - |
| `--limit N` | Stop after generating N games | None |

### Adaptive sampling (recommended for training data)

Samples in batches until max(SEM) over legal actions falls below the threshold.

| Option | Description | Default |
|--------|-------------|---------|
| `--adaptive` | Enable adaptive convergence-based sampling | disabled |
| `--min-samples N` | Minimum samples before checking convergence | 5000 |
| `--max-samples N` | Maximum samples (hard cap) | 2000000 |
| `--adaptive-batch-size N` | Samples per iteration | 1000 |
| `--sem-threshold F` | SEM threshold (Q-value points) | 0.1 |

### Posterior weighting (optional, improves mid-game E[Q])

| Option | Description | Default |
|--------|-------------|---------|
| `--posterior` | Enable posterior weighting | disabled |
| `--posterior-window K` | Sliding window size K | 4 |
| `--posterior-tau T` | Temperature for softmax weighting | 0.1 |
| `--posterior-mix α` | Uniform mix coefficient | 0.1 |

### Exploration policies (optional)

| Option | Description | Default |
|--------|-------------|---------|
| `--exploration POLICY` | `greedy`, `epsilon_greedy`, `boltzmann` | greedy |
| `--epsilon ε` | Epsilon for epsilon_greedy | 0.1 |
| `--temperature T` | Temperature for boltzmann | 2.0 |

## File structure

```
forge/eq/
├── generate/                # ★ GPU PIPELINE PACKAGE ★
│   ├── pipeline.py          # generate_eq_games_gpu() main orchestrator
│   ├── types.py             # PosteriorConfig, AdaptiveConfig, records
│   ├── sampling.py          # sample_worlds_batched, infer_voids_batched
│   ├── tokenization.py      # tokenize_batched
│   ├── actions.py           # select_actions, record_decisions
│   ├── posterior.py         # compute_posterior_weighted_eq
│   ├── adaptive.py          # sample_until_convergence
│   ├── enumeration.py       # enumerate_or_sample_worlds
│   ├── eq_compute.py        # compute_eq_with_counts, compute_eq_pdf
│   ├── model.py             # query_model
│   ├── deals.py             # build_hypothetical_deals
│   └── cli.py               # CLI entry point
├── collate.py               # GPU records → training format
├── game_tensor.py           # GameStateTensor for GPU
├── sampling_mrv_gpu.py      # MRV world sampler on GPU
├── tokenize_gpu.py          # GPUTokenizer
├── voids.py                 # Void inference from play history
├── sampling.py              # CPU backtracking sampler
├── oracle.py                # Stage1Oracle wrapper (async CUDA streams — ASYNC_PIPELINE.md)
├── game.py                  # GameState tracker
├── transcript_tokenize.py   # Stage 2 tokenizer (public info only)
├── viewer.py                # Interactive inspector (press 'd' for debug mode)
└── test_*.py                # Unit tests
```

## Dataset format

Each training example is one decision point. Key fields: `transcript_tokens` (36, 8),
`e_q_mean` (7,) — **E[Q] in POINTS, roughly [−42, +42], NOT logits, do NOT softmax** —
`e_q_var` (7,), `legal_mask` (7,), `action_taken`, `u_mean`/`u_max`, `ess`/`max_w`
(posterior diagnostics), `n_samples`, `converged`. Full spec:
`docs/EQ_STAGE2_TRAINING.md`.

## Testing and validation

```bash
python -m pytest forge/eq/ -v --timeout=60          # All unit tests (< 30s)

python scripts/validate_eq_computation.py <dataset.pt>   # E[Q] matches fresh oracle queries
python scripts/analyze_eq_predictions.py <dataset.pt>    # Predictions vs actual outcomes
```
