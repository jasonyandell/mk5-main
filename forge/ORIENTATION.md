# Crystal Forge — Operational Reference

Knowledge lives in the wiki — see [wiki/entities/forge.md](../wiki/entities/forge.md)
(architecture, mental models, findings) and
[wiki/topics/expected-q-value.md](../wiki/topics/expected-q-value.md) (E[Q] semantics
and consumption foot-guns). This file is the CLI command reference and data-path list.

## Setup

```bash
pip install -r forge/requirements.txt

# Verify imports
python -c "from forge.ml import module, data, metrics; from forge.oracle import schema; print('OK')"
```

## Data paths

```
External drive (/mnt/d/):
- shards-standard/       215GB, 1124 train parquet files
- shards-marginalized/   191GB, marginalized oracle output

Local (mk5-tailwind/data/):
- tokenized-full/        5GB, 11.2M samples (ready to use)
- bidding-results/       P(make) Monte Carlo evaluations
- eq-games/{train,val,test}/seed_*.pt   E[Q] per-seed files

Forge-relative:
- runs/                  Training outputs
- forge/models/          Pre-trained checkpoints (see models/README.md)
- forge/data/tokenized/  Small test tokenized data
```

Split routing is deterministic by `seed % 1000`: 0-899 train, 900-949 val,
950-999 test (test is sacred — never touched during development).

## Essential commands

### Oracle generation (Stage 1)

```bash
python -m forge.cli.generate_continuous                  # Standard: 1 decl per seed
python -m forge.cli.generate_continuous --marginalized   # N opp seeds per P0 hand
python -m forge.cli.generate_continuous --dry-run        # Preview gaps
python -m forge.cli.generate_continuous --start-seed 1000

# Single-seed debugging with Q-value inspection
python -m forge.oracle.generate --seed 0 --decl sixes --show-qvals --out /dev/null

# Fix P0's hand for oracle Q-value investigation
python -m forge.oracle.generate --seed 0 --decl sixes \
    --p0-hand "6-6,6-5,6-4,6-2,6-1,6-0,2-2" --show-qvals --out /dev/null
```

### Tokenization

```bash
python -m forge.cli.tokenize_data --input data/shards-standard --output data/tokenized
python -m forge.cli.tokenize_data --dry-run
```

### Training and evaluation

```bash
python -m forge.cli.train --fast-dev-run --no-wandb      # Quick sanity check
python -m forge.cli.train --data ../data/tokenized-full --batch-size 4096 --epochs 20 --wandb
python -m forge.cli.train --precision bf16-mixed --n-layers 4 --n-heads 8 --embed-dim 128

python -m forge.cli.eval --checkpoint runs/domino/version_0/checkpoints/best.ckpt
```

Training auto-detects hardware; use `--precision bf16-mixed` on A100/H100.

### E[Q] generation (Stage 2 training data)

See [eq/README.md](eq/README.md) for the full CLI reference.

```bash
python -m forge.cli.generate_eq_continuous \
    --checkpoint forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt \
    --adaptive --posterior --posterior-window 4

python -m forge.eq.viewer data/eq-games/train/seed_00000042.pt   # Interactive inspector
```

### Bidding evaluation

```bash
python -m forge.bidding.evaluate --hand "6-4,5-5,4-2,3-1,2-0,1-1,0-0" --samples 100
python -m forge.cli.bidding_continuous              # Continuous, N=500
python -m forge.cli.bidding_continuous --dry-run

# Replay losing games trick-by-trick
python -m forge.bidding.investigate --hand "..." --trump sixes --below 42 --samples 500
```

### Flywheel (iterative fine-tuning)

```bash
python -m forge.cli.flywheel init --wandb-group my-experiment --start-seed 200
python -m forge.cli.flywheel status
python -m forge.cli.flywheel --once   # One iteration
python -m forge.cli.flywheel          # Run continuously
```

See [flywheel/RUNBOOK.md](flywheel/RUNBOOK.md) for operations.

## State diagnosis

```bash
cat data/tokenized/manifest.yaml 2>/dev/null || echo "No tokenized data"
ls runs/domino/ 2>/dev/null || echo "No runs yet"
pgrep -af "forge.oracle\|forge.cli" || echo "Nothing running"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}')"
```

Use `--help` on any command for full options.
