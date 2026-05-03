# GPU Selection Guide

## Complete GPU Specifications

| GPU | $/hour | $/second | VRAM | Architecture | Max Count |
|-----|--------|----------|------|--------------|-----------|
| T4 | $0.59 | $0.000164 | 16GB | Turing | 8 |
| L4 | $0.80 | $0.000222 | 24GB | Ada Lovelace | 8 |
| A10G | $1.10 | $0.000306 | 24GB | Ampere | 4 |
| L40S | $1.95 | $0.000542 | 48GB | Ada Lovelace | 8 |
| A100-40GB | $2.10 | $0.000583 | 40GB | Ampere | 8 |
| A100-80GB | $2.50 | $0.000694 | 80GB | Ampere | 8 |
| H100 | $3.95 | $0.001097 | 80GB | Hopper | 8 |
| H200 | $4.54 | $0.001261 | 141GB | Hopper | 8 |
| B200 | $6.25 | $0.001736 | 192GB | Blackwell | 8 |

## Availability (Best to Worst)

Based on real-world experience:

1. **A10G** - Always available, never queued
2. **A100-80GB** - Usually available
3. **H200** - Often available, excellent speed
4. **L40S** - Good availability
5. **B200** - Sometimes queued (newest hardware)
6. **H100** - Often queued (high demand)

## Selection by Use Case

### Inference
- **Budget**: T4 or L4
- **Balanced**: A10G (best availability + cost)
- **Large models**: L40S (48GB) or A100-80GB

### Training
- **Development**: A10G
- **Production**: A100-80GB or H100
- **Speed priority**: H200 or B200

### Batch Processing
- **Cost priority**: A10G
- **Speed priority**: H200
- **Memory-bound**: A100-80GB or H200

## Cost-Efficiency Analysis

Faster GPUs often have similar or better cost-per-task:

| GPU | Relative Speed | Relative Cost | Cost Efficiency |
|-----|----------------|---------------|-----------------|
| A10G | 1x (baseline) | 1x | Baseline |
| A100-80GB | ~2.5x | 2.3x | +9% efficient |
| H100 | ~4x | 3.6x | +11% efficient |
| H200 | ~5x | 4.1x | +22% efficient |

**Key insight**: H200 at 5x speed for 4.1x price = better value than A10 for wall-clock time.

## What $5 Gets You

| GPU | Approximate Tasks | Wall-clock |
|-----|-------------------|------------|
| A10G | ~10,000 shards | ~4 hours |
| H200 | ~10,000 shards | ~1 hour |
| H100 | ~10,000 shards | ~1.2 hours |

## GPU String Formats

```python
# Single GPU
gpu="A10G"
gpu="H100"

# Multi-GPU (up to max count)
gpu="A100:2"      # 2x A100
gpu="H100:4"      # 4x H100
gpu="H200:8"      # 8x H200 (1,128 GB!)

# Prevent auto-upgrade (H100 → H200)
gpu="H100!"       # Force exactly H100
```

## Auto-Upgrades

Modal may upgrade at no extra cost:
- H100 → H200 (when H200 available, H100 queued)
- A100-40GB → A100-80GB

Use `!` suffix to prevent: `gpu="H100!"`

## Memory Guidelines

| VRAM | Can Handle |
|------|------------|
| 16GB (T4) | Small models, light inference |
| 24GB (A10/L4) | Most inference, ~500M state enumeration |
| 48GB (L40S) | Large inference models |
| 80GB (A100/H100) | Large training, billion-param models |
| 141GB (H200) | Very large models, extensive batch sizes |
| 192GB (B200) | Largest models |

## Concurrency vs GPU Size

With `@modal.concurrent(max_inputs=2)`:

| GPU | Safe concurrent inputs | Risk |
|-----|------------------------|------|
| A10G (24GB) | 2 | ~2% OOM on large tasks |
| A100-80GB | 3-4 | Lower OOM risk |
| H200 (141GB) | 4-6 | Very low OOM risk |

## Strategy: Fire Multiple GPU Types

When one GPU type is queued, launch same job on multiple types:

```python
# modal_app_h200.py: gpu="H200"
# modal_app_a10.py: gpu="A10G"

# Run both - duplicates skip via file existence check
modal run modal_app_h200.py::main &
modal run modal_app_a10.py::main &
```

First available GPU type wins. Lock files prevent duplicate work.
