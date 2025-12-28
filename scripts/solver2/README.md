# `solver2`: GPU training-data generator

This package solves the full play-phase game tree for a single **(deal seed, declaration)** and writes a crash-safe output file containing:

- All reachable packed states (int64)
- Minimax value per state (int8)
- Child-value per move (7 moves, int8; `-128` means illegal)

The core algorithm and state encoding are described in `docs/SOLVER_GPU_TRAINING.md`.

## Install (common denominator)

You need a CUDA-enabled PyTorch build for GPU runs.

Suggested Python deps:
- `torch` (GPU build)
- `numpy`
- `pyarrow` (for Parquet output)

If you don't want `pyarrow`/`numpy`, use `--format pt` to write `torch.save()` outputs instead.

Quick install (excluding `torch` because GPU wheels vary by CUDA version):

```bash
python3 -m pip install -r scripts/solver2/requirements.txt
```

## Run

From repo root:

```bash
python3 -m scripts.solver2.main --seed 0 --decl fives --out data/solver2 --device cuda:0
python3 -m scripts.solver2.main --seed-range 0:1000 --decl all --out data/solver2 --device cuda:0
```

If you hit CUDA OOM on a small GPU, try smaller chunks (especially `--enum-chunk` which controls enumeration):

```bash
python3 -m scripts.solver2.main --seed 0 --decl fives --device cuda:0 \
  --enum-chunk 50000 --child-index-chunk 250000 --solve-chunk 250000
```

To diagnose which phase uses the most memory, add `--log-memory`:

```bash
python3 -m scripts.solver2.main --seed 0 --decl fives --device cuda:0 --log-memory
```

Output files are per-seed for easy resume:

```
data/solver2/seed_00000000_decl_5.parquet
```

## GPU farm / job array pattern

Run disjoint seed ranges per job (and per GPU) and point all jobs at the same shared output directory; existing files are skipped by default.

Example (one GPU per process):

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m scripts.solver2.main --seed-range 0:100000 --decl all --out /mnt/datasets/solver2 --device cuda:0
```
