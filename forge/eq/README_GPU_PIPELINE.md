# GPU-Native E[Q] Generation Pipeline

## Overview

The GPU-native E[Q] pipeline generates training data by playing games using a Stage 1 oracle to estimate E[Q] (expected Q-value) for each legal action. The pipeline runs entirely on GPU for both game state management and model inference.

## Architecture

```
Core Loop (per decision):
──────────────────────────────────────────────────────────────────────────────
    GameStateTensor          N games running in parallel
           ↓
    WorldSamplerMRV          Sample M opponent hands per game
           ↓                 (constrained by voids, played dominoes)
    Build hypothetical       Combine known hand + sampled opponents
           ↓                 → [N, M, 4, 7] full deals
    GPUTokenizer             Tokenize current decision
           ↓                 → [N×M, 32, 12] tokens
    Model forward            Query oracle for Q-values
           ↓                 → [N×M, 7] Q per action
    E[Q] aggregation         Uniform or posterior-weighted (see below)
           ↓                 → [N, 7] expected Q per action
    Action selection         Greedy, softmax, or exploration policy
           ↓
    apply_actions            Advance game state
           └──→ repeat until all games complete
```

One Python loop iteration per game **step**, not per decision.

## Key Features

### Aggregation Modes

**1. UNIFORM (default):**
```python
generate_eq_games_gpu(model, hands, decl_ids, n_samples=50)
# E[Q] = mean over M samples. Fast, no extra model calls.
```

**2. POSTERIOR-WEIGHTED:**
```python
from forge.eq.generate import PosteriorConfig

generate_eq_games_gpu(
    model, hands, decl_ids,
    posterior_config=PosteriorConfig(
        enabled=True,
        window_k=4,    # Past K steps for weighting
        tau=0.1,       # Softmax temperature
        uniform_mix=0.1,  # Robustness mixture
    )
)
```

### Adaptive Sampling

Instead of fixed sample count, adaptively sample until E[Q] estimates converge:

```python
from forge.eq.generate import AdaptiveConfig

generate_eq_games_gpu(
    model, hands, decl_ids,
    adaptive_config=AdaptiveConfig(
        enabled=True,
        min_samples=50,     # Minimum before checking convergence
        max_samples=2000,   # Hard cap
        batch_size=50,      # Samples per iteration
        sem_threshold=0.5,  # Stop when max(SEM) < this (Q-value points)
    )
)
```

**Convergence criterion**: max(SEM) over legal actions < sem_threshold, where SEM = σ / √n.

**Benefits**:
- Early stopping for low-variance decisions (saves compute)
- More samples for high-variance decisions (improves accuracy)

**Output includes**:
- `n_samples`: Actual samples used (varies per decision)
- `converged`: True if SEM < threshold, False if hit max_samples

Uses Bayesian inference to reweight samples based on past play. Samples consistent with observed opponent actions get higher weight.

Additional steps when posterior enabled:
- Reconstruct past K states (what board looked like K steps ago)
- Reconstruct historical hands (undo plays to get hands at step k)
- Tokenize past steps [N×M×K, 32, 12] (second tokenization)
- Model forward for past Q-values (second oracle call)
- Compute legal masks at each past step
- Bayesian weighting: P(world|observed actions) ∝ P(actions|world)

Output includes ESS (effective sample size) diagnostics:
- ESS << M indicates strong filtering (opponent play is informative)
- ESS ≈ M indicates weak filtering (opponent play doesn't help much)

### Exploration Policy

```python
from forge.eq.types import ExplorationPolicy

generate_eq_games_gpu(
    model, hands, decl_ids,
    exploration_policy=ExplorationPolicy(
        mode="boltzmann",  # or "epsilon", "blunder"
        temperature=1.0,
        epsilon=0.1,
        seed=42,
    )
)
```

When exploration_policy is provided, action selection can deviate from greedy:
- Boltzmann sampling with temperature
- Epsilon-greedy with random legal action fallback
- Deliberate "blunders" for robustness training

Tracks `q_gap` (regret) when exploration causes suboptimal action selection.

## Files

The GPU pipeline is organized as a package in `forge/eq/generate/`:

| File | Purpose |
|------|---------|
| `generate/` | **Package containing GPU pipeline** |
| `generate/pipeline.py` | Main entry point: `generate_eq_games_gpu()` |
| `generate/types.py` | `PosteriorConfig`, `AdaptiveConfig`, `DecisionRecordGPU`, `GameRecordGPU` |
| `generate/sampling.py` | `sample_worlds_batched`, `infer_voids_batched` |
| `generate/tokenization.py` | `tokenize_batched`, `build_remaining_bitmasks` |
| `generate/actions.py` | `select_actions`, `record_decisions` |
| `generate/posterior.py` | `compute_posterior_weighted_eq` |
| `generate/adaptive.py` | `sample_until_convergence` |
| `generate/enumeration.py` | `enumerate_or_sample_worlds` |
| `generate/eq_compute.py` | `compute_eq_with_counts`, `compute_eq_pdf` |
| `generate/model.py` | `query_model` |
| `generate/deals.py` | `build_hypothetical_deals` |
| `generate/cli.py` | CLI entry point for `python -m forge.eq.generate` |
| `game_tensor.py` | `GameStateTensor` - vectorized game state on GPU |
| `sampling_mrv_gpu.py` | `WorldSamplerMRV` - MRV constraint solver (guaranteed valid) |
| `tokenize_gpu.py` | `GPUTokenizer` - pure tensor tokenization |
| `collate.py` | Convert `GameRecordGPU` to Stage 2 training format |
| `types.py` | `ExplorationPolicy`, `PosteriorDiagnostics` dataclasses |

## API Usage

```python
from forge.eq.generate import generate_eq_games_gpu, PosteriorConfig
from forge.eq.collate import collate_batch
from forge.eq.oracle import Stage1Oracle
from forge.oracle.rng import deal_from_seed

# Load model
oracle = Stage1Oracle("checkpoints/model.ckpt", device='cuda')

# Generate deals
hands = [deal_from_seed(i) for i in range(32)]
decl_ids = [i % 10 for i in range(32)]

# Basic usage (uniform weighting, greedy action selection)
results = generate_eq_games_gpu(
    model=oracle.model,
    hands=hands,
    decl_ids=decl_ids,
    n_samples=50,
    device='cuda',
    greedy=True,
)

# With posterior weighting
results = generate_eq_games_gpu(
    model=oracle.model,
    hands=hands,
    decl_ids=decl_ids,
    n_samples=50,
    device='cuda',
    posterior_config=PosteriorConfig(enabled=True, window_k=4),
)

# Each result contains:
# - results[i].decisions: List of DecisionRecordGPU (28 per game)
#     - .e_q: [7] E[Q] mean values
#     - .e_q_var: [7] E[Q] variance
#     - .ess: Effective sample size (if posterior enabled)
#     - .q_gap: Regret from exploration (if exploration enabled)
#     - .n_samples: Actual samples used (if adaptive enabled)
#     - .converged: True if SEM < threshold (if adaptive enabled)
# - results[i].hands: Initial deal
# - results[i].decl_id: Declaration

# Convert to Stage 2 training format
batch = collate_batch(results, game_indices=range(32))
```

## Performance

### Empirical Results

| Hardware | n_games | n_samples | Mode | Throughput |
|----------|---------|-----------|------|------------|
| RTX 3050 Ti (4GB) | 32 | 50 | Uniform | ~5 games/s |
| RTX 3050 Ti (4GB) | 32 | 50 | Posterior | ~2 games/s |

Posterior mode is ~2-3x slower due to second model call for past K steps.

### Memory Budget (3050 Ti, 32 games × 50 samples)

| Component | Size |
|-----------|------|
| GameStateTensor | ~10KB |
| WorldSampler buffers | ~1MB |
| Tokenizer buffers | ~2MB |
| Model (~3M params) | ~12MB |
| Tokens (1,600 × 32 × 12) | ~600KB |
| Activations | ~50MB |
| **Total** | ~65MB |

## Design Details

### GPU-Only Architecture

The pipeline is GPU-only by design. If CUDA is unavailable, it raises an error rather than silently falling back to CPU. This prevents accidentally running performance-critical workloads on CPU.

### Device-Aware Design

Handles mixed device setups:
- GameStateTensor on device A
- Model on device B
- Automatic device transfers where needed

### Pre-Allocated Buffers

All GPU components use pre-allocated buffers:
- WorldSampler: Permutation and hands buffers
- GPUTokenizer: Token and mask buffers
- Amortizes allocation cost across all decisions

### Vectorized Operations

Minimal Python loops during generation:
- All N games advance together each step
- Single model forward pass for all (N × M) worlds
- Vectorized E[Q] reduction

## Testing

```bash
# All tests
pytest forge/eq/test_generate_gpu.py -v

# Skip performance tests
pytest forge/eq/test_generate_gpu.py -k "not performance" -v

# Memory test
pytest forge/eq/test_generate_gpu.py::test_memory_no_oom -v
```

## References

- Pipeline: `forge/eq/generate/` (package)
- Game tensor: `forge/eq/game_tensor.py`
- World sampling: `forge/eq/sampling_mrv_gpu.py`
- Tokenization: `forge/eq/tokenize_gpu.py`
- Collation: `forge/eq/collate.py` (see `COLLATE.md`)
