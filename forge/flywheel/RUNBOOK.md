# Flywheel Runbook

**For Claude Code**: Read `state.yaml` first, then follow instructions below.

## Quick Start

```bash
# Run flywheel continuously (trains forever until Ctrl+C or error)
python -m forge.cli.flywheel

# Run just one iteration
python -m forge.cli.flywheel --once

# Check status
python -m forge.cli.flywheel status

# Initialize fresh flywheel
python -m forge.cli.flywheel init --wandb-group my-experiment --start-seed 200
```

**Important**: The flywheel runs **continuously** by default. It will keep training iteration after iteration until you interrupt it (Ctrl+C) or an error occurs.

## State Machine

```
ready ──run──> running ──success──> ready (next iteration)
                  │
                  └──failure──> failed
                                  │
                         fix & ──┘
                         retry
```

## Handling Each Status

### `status: ready`
Run: `python -m forge.cli.flywheel`

Each iteration will:
0. Ensure golden val/test shards exist (auto-generates if missing)
1. Generate oracle shards for `current.seed_range` (1 decl per seed)
2. Tokenize all data to `data/flywheel-tokenized/`
3. Fine-tune from previous checkpoint (2 epochs, LR=3e-5)
4. Evaluate against golden val/test sets and log to W&B
5. Update state.yaml with results
6. **Automatically start next iteration** (unless `--once` flag)

**Monitor with haiku subagent** - spawn to watch for OOM/stuck.

### `status: running`
Another process is running. Check:
- Is it still alive? `pgrep -af flywheel`
- Check logs in `runs/domino/` or W&B

If stuck, set `status: failed` with error message.

### `status: failed`
Read `last_error` and `next_action` for recovery steps.

Common fixes:
- **OOM**: Reduce batch size, clear GPU memory, or use smaller seed range
- **Data error**: Check shards exist, re-run tokenization
- **Checkpoint missing**: Verify path in state.yaml

After fixing: set `status: ready` and re-run.

### `status: done`
Training complete. Review history for results.

## Haiku Monitor Pattern

Each iteration takes 30-60+ minutes. Spawn a haiku subagent to monitor:

```
Monitor the flywheel training process. Check every 60 seconds:

1. Is training running? Run: pgrep -af 'forge.cli.train'
2. Check latest metrics: tail -20 runs/domino/version_*/metrics.csv 2>/dev/null | tail -5
3. Check for errors: grep -i 'error\|oom\|cuda' runs/domino/version_*/events.out.tfevents.* 2>/dev/null

Watch for these problems:
- "CUDA out of memory" → OOM error
- No new CSV lines for 10+ minutes → stuck
- Process not running but state.yaml says "running" → crashed

If problem detected:
1. Edit forge/flywheel/state.yaml:
   - Change status: failed
   - Set last_error: "<describe what you saw>"
   - Set next_action: "<suggested fix>"
2. Report back to main agent

If training completes successfully, the flywheel script auto-updates state.yaml.
Just confirm completion and report final metrics.
```

### Running in Background

To run flywheel and monitor simultaneously:
```bash
# Option 1: Run flywheel, spawn haiku monitor
python -m forge.cli.flywheel  # (in foreground or background)

# Option 2: Check W&B dashboard for live metrics
# Project: crystal-forge, Group: <wandb_group from state.yaml>
```

## Manual Recovery

If state.yaml gets corrupted:
```bash
# Check what checkpoints exist
ls runs/domino/*/checkpoints/

# Check W&B for last successful run
# Then manually edit state.yaml with correct values
```

## Architecture

- **817K param model** (Large v2)
- **~50K samples per seed** (1 decl per seed, decl = seed % 10)
- **10 seeds per iteration** = ~500K train samples
- **2 epochs per iteration** at LR=3e-5 (lower for fine-tuning)

### Golden Val/Test Sets

For consistent benchmarking across iterations, the flywheel uses "golden" val/test shards that are **binary identical** to the original training data:

- **Val**: Seeds 900-902 with ALL 10 decls (30 shards, ~1.5M samples)
- **Test**: Seed 950 with ALL 10 decls (10 shards, ~500K samples)

These are auto-generated on first run if missing. Oracle generation is deterministic by seed, so regenerating produces identical data to the original baseline training. This ensures true apples-to-apples comparison.

Each iteration is one W&B run, all grouped under `wandb_group`.

## Data Layout

```
data/flywheel-shards/       # Oracle parquet files
  # Training shards (1 decl per seed, decl = seed % 10)
  seed_00000200_decl_0.parquet
  seed_00000201_decl_1.parquet
  ...

  # Golden val shards (seeds 900-902, all 10 decls = 30 shards)
  seed_00000900_decl_0.parquet
  ...
  seed_00000902_decl_9.parquet

  # Golden test shards (seed 950, all 10 decls = 10 shards)
  seed_00000950_decl_0.parquet
  ...
  seed_00000950_decl_9.parquet

data/flywheel-tokenized/    # Pre-tokenized numpy arrays
  train/                    # From flywheel training shards
  val/                      # From golden val shards (~1.5M samples)
  test/                     # From golden test shards (~500K samples)
```
