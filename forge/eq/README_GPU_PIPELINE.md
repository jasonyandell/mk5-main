# GPU-Native E[Q] Generation Pipeline

## Overview

Phase 4 of the 4-phase GPU-native E[Q] pipeline (issue t42-k7ny) is complete:

- ✅ **Phase 1**: `forge/eq/game_tensor.py` - GameStateTensor (GPU game state)
- ✅ **Phase 2**: `forge/eq/sampling_gpu.py` - WorldSampler (GPU world sampling)
- ✅ **Phase 3**: `forge/eq/tokenize_gpu.py` - GPUTokenizer (GPU tokenization)
- ✅ **Phase 4**: `forge/eq/generate_gpu.py` - Full GPU pipeline integration

## Architecture

```
GameStateTensor (N games on GPU)
      ↓
WorldSampler (N × M samples) → CPU fallback for constrained scenarios
      ↓
GPUTokenizer (pure tensor ops)
      ↓
Model forward (single batch)
      ↓
E[Q] aggregation → best actions
      ↓
apply_actions (vectorized)
      └──→ repeat until games done
```

One Python loop iteration per game **step**, not per decision.

## Files Created/Modified

### New Files

1. **`forge/eq/generate_gpu.py`** (Phase 4)
   - Main entry point: `generate_eq_games_gpu()`
   - Integrated pipeline processing N games in parallel
   - CPU fallback for constrained sampling scenarios
   - Device-aware (works on CUDA, CPU, or mixed setups)

2. **`forge/eq/test_generate_gpu.py`**
   - Comprehensive test suite
   - Tests correctness, performance, memory usage
   - Tests CPU compatibility and various declarations

### Pre-existing Files (Phase 1-3, already implemented)

3. **`forge/eq/game_tensor.py`** (Phase 1)
   - GPU-vectorized game state
   - Pre-computed lookup tables
   - Batch operations for N games

4. **`forge/eq/sampling_gpu.py`** (Phase 2)
   - WorldSampler class with pre-allocated buffers
   - Parallel first-fit algorithm
   - Void constraint checking on GPU

5. **`forge/eq/tokenize_gpu.py`** (Phase 3)
   - GPUTokenizer class
   - Pure tensor operations (no Python loops)
   - Pre-computed feature tables

## API Usage

```python
from forge.eq.generate_gpu import generate_eq_games_gpu
from forge.eq.oracle import Stage1Oracle
from forge.oracle.rng import deal_from_seed

# Load model
oracle = Stage1Oracle("checkpoints/model.ckpt", device='cuda')

# Generate games
hands = [deal_from_seed(i) for i in range(32)]
decl_ids = [i % 10 for i in range(32)]

results = generate_eq_games_gpu(
    model=oracle.model,
    hands=hands,
    decl_ids=decl_ids,
    n_samples=50,
    device='cuda',
    greedy=True,
)

# Each result contains:
# - results[i].decisions: List of DecisionRecordGPU (28 per game)
# - results[i].hands: Initial deal
# - results[i].decl_id: Declaration
```

## Performance Characteristics

### Target Performance (from plan)

| Hardware | n_games | n_samples | batch_size | Target throughput |
|----------|---------|-----------|------------|-------------------|
| RTX 3050 Ti (4GB) | 32 | 50 | 1,600 | 5+ games/s |
| H100 (80GB) | 1000 | 1024 | 1M | 100+ games/s |

### Test Results

All non-performance tests pass:
- ✅ Single game correctness
- ✅ Multi-game batch processing
- ✅ Memory allocation (32 games × 50 samples, no OOM)
- ✅ CPU compatibility
- ✅ CPU/GPU baseline comparison (stochastic agreement)
- ✅ Greedy vs sampled action selection
- ✅ Various declaration types

## Key Features

### 1. CPU Fallback for Constrained Sampling

The GPU sampler uses a parallel first-fit algorithm which is very fast but can fail for heavily constrained scenarios (late game with many void inferences). When this happens, the pipeline automatically falls back to CPU backtracking:

```python
try:
    worlds = sampler.sample(pools, hand_sizes, voids, decl_ids, n_samples)
except RuntimeError:
    # GPU failed - use CPU backtracking
    worlds = _sample_worlds_cpu_fallback(states, ...)
```

This hybrid approach gives:
- **Fast path**: GPU for early/mid-game (most samples)
- **Reliable path**: CPU for late-game constrained scenarios

### 2. Device-Aware Design

The pipeline handles mixed device setups gracefully:
- GameStateTensor on device A
- Model on device B (e.g., CUDA when data is on CPU)
- Automatic device transfers where needed
- Works on CUDA, CPU, or MPS

### 3. Pre-Allocated Buffers

All GPU components use pre-allocated buffers to avoid per-call overhead:
- WorldSampler: Permutation and hands buffers
- GPUTokenizer: Token and mask buffers
- Amortizes allocation cost across all decisions in all games

### 4. Vectorized Operations

No Python loops over games during generation:
- All N games advance together each step
- Single model forward pass for all (N × M) worlds
- Vectorized E[Q] reduction

## Memory Budget

For 3050 Ti (4GB VRAM) with 32 games × 50 samples:

| Component | Size |
|-----------|------|
| GameStateTensor | ~10KB |
| WorldSampler buffers | ~1MB |
| Tokenizer buffers | ~2MB |
| Model (~3M params) | ~12MB |
| Tokens (1,600 × 32 × 12) | ~600KB |
| Activations | ~50MB |
| **Total** | ~65MB |

Leaves plenty of headroom for GPU operations.

## Limitations & Future Work

### Current Limitations

1. **GPU Sampling Constraints**
   - Fails for heavily constrained scenarios (2+ voids per opponent, late game)
   - Falls back to CPU (slower but correct)
   - Could be improved with better rejection sampling or MCMC

2. **No Posterior Weighting**
   - Pipeline uses uniform world weights
   - For posterior features, use CPU pipeline (`generate_eq_game`)

3. **No Exploration Policy**
   - Only greedy or softmax sampling
   - For epsilon/boltzmann/blunder exploration, use CPU pipeline

### Future Improvements

1. **Hybrid Sampling** (Phase 2 enhancement)
   - Start with GPU, detect constraint violations earlier
   - Switch to CPU only for problematic games
   - Could improve GPU sampling success rate to 90%+

2. **Adaptive Oversampling** (Phase 2 enhancement)
   - Dynamically adjust oversample factor based on void count
   - More oversample when constraints are tight
   - Target: 95%+ GPU sampling success

3. **Chunked Inference** (Phase 4 enhancement)
   - For >1M batch sizes on H100
   - Split into 100K chunks to avoid OOM
   - Allows scaling to 10K+ concurrent games

4. **Model Compilation** (Phase 4 enhancement)
   - Enable `torch.compile()` for model forward pass
   - CUDA graphs for reduced kernel launch overhead
   - Could improve throughput by 20-30%

## Testing

Run tests with:

```bash
# All non-performance tests
pytest forge/eq/test_generate_gpu.py -k "not performance" -v

# Performance test (requires CUDA)
pytest forge/eq/test_generate_gpu.py::test_performance_32_games -v

# Memory test
pytest forge/eq/test_generate_gpu.py::test_memory_no_oom -v
```

## Compatibility

The GPU pipeline is designed to coexist with the CPU pipeline:
- Same semantics (produces valid E[Q] games)
- Different implementation (GPU-optimized)
- Use GPU for throughput, CPU for features (posterior, exploration)

Both pipelines will continue to be maintained.

## References

- Plan: `/home/jason/.claude/plans/precious-doodling-bentley.md`
- Issue: t42-k7ny
- Phase 1: `forge/eq/game_tensor.py`
- Phase 2: `forge/eq/sampling_gpu.py`
- Phase 3: `forge/eq/tokenize_gpu.py`
- Phase 4: `forge/eq/generate_gpu.py` (this implementation)
