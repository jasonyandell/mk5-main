# Flywheel Runbook

**For Claude Code**: Read `state.yaml` first, then follow instructions below.

## Quick Start

```bash
# Check status
python -m forge.cli.flywheel status

# Run next iteration
python -m forge.cli.flywheel

# Initialize fresh flywheel
python -m forge.cli.flywheel init --wandb-group my-experiment --start-seed 200
```

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

This will:
1. Generate oracle shards for `current.seed_range`
2. Tokenize new data
3. Fine-tune from previous checkpoint (1-2 epochs)
4. Evaluate and log to W&B
5. Update state.yaml with results

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

Spawn a haiku subagent to monitor long-running flywheel:

```
Task: Monitor flywheel training. Tail the latest log in runs/domino/.
If you see "CUDA out of memory" or no output for 10+ minutes:
1. Update forge/flywheel/state.yaml: status: failed, last_error: <what you saw>
2. Report back
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
- **~50K samples per seed** (1 decl per seed)
- **10 seeds per iteration** = ~500K samples
- **1-2 epochs per iteration**

Each iteration is one W&B run, all grouped under `wandb_group`.
