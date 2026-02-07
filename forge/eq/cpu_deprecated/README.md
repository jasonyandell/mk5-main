# CPU-based E[Q] Generation (DEPRECATED)

**Status:** Deprecated in favor of GPU pipeline
**Replacement:** `forge/eq/generate_gpu.py`
**Date Deprecated:** 2026-01-22

## Overview

This directory contains the legacy CPU-based E[Q] training data generation pipeline. These modules have been superseded by the GPU-native implementation which provides ~100x speedup.

## Why Deprecated?

The CPU pipeline was the original implementation but suffered from:
- Sequential world sampling (CPU-bound)
- High latency per decision (~22ms vs ~0.2ms GPU)
- Complex pipelining attempts to hide latency
- Difficult to maintain dual pipelines

The GPU pipeline (`generate_gpu.py`) replaced all of this with:
- Vectorized GPU sampling using `sampling_gpu.py`
- Batched posterior weighting with `posterior_gpu.py`
- Direct tensor operations (no CPU/GPU transfer overhead)
- Single, simple implementation

## Migration

**Do not use these modules for new work.**

If you need E[Q] generation:
- Use `forge/eq/generate_gpu.py::generate_eq_games_gpu()`
- See `forge/cli/generate_eq_continuous.py` for CLI usage
- See `forge/eq/README_GPU_PIPELINE.md` for architecture

## Contents

### Generation Modules
- `generate.py` - Compatibility shim (imports from CPU modules)
- `generate_game.py` - Single game generation (CPU)
- `generate_batched.py` - Multi-game batched generation (CPU)
- `generate_continuous.py` - Continuous generation CLI (CPU)
- `generate_pipelined.py` - Pipelined generation with worker threads (CPU)
- `generate_dataset.py` - Dataset generation CLI (CPU)

### Core Computation
- `posterior.py` - CPU-based posterior weighting (superseded by `posterior_gpu.py`)

### Testing & Benchmarking
- `test_generate.py` - Tests for CPU generation
- `test_batching_phase3.py` - Multi-game batching tests
- `test_pipelined.py` - Pipelined generation tests
- `benchmark_async.py` - Async oracle benchmarks
- `benchmark_pipelined.py` - Pipeline vs batched benchmarks
- `profile_throughput.py` - CPU generator profiling

## Intentionally Breaking Imports

Imports are **not** being updated. Any code still importing from these modules will break intentionally so we can:
1. Identify all remaining CPU pipeline usage
2. Migrate to GPU pipeline explicitly
3. Avoid silent degradation to slow CPU path

If you encounter import errors, migrate to the GPU pipeline.
