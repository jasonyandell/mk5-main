# Crystal Forge

Knowledge lives in the wiki — see [wiki/entities/forge.md](../wiki/entities/forge.md)
for what Forge is, its lineage, and its findings, and
[wiki/topics/the-oracle.md](../wiki/topics/the-oracle.md) for solver history and the
shard data format. This file is the module map.

## Module map

```
forge/
├── oracle/          Stage 1: GPU minimax solver (backward induction)
├── ml/              PyTorch Lightning training core (models, data, metrics)
├── eq/              Stage 2: E[Q] imperfect-info pipeline (GPU-native)
├── zeb/             MCTS self-play with distributed Vast.ai workers
├── bidding/         Monte Carlo P(make) estimation for bid evaluation
├── cli/             Command-line interfaces for all major operations
├── flywheel/        Automated generate→tokenize→train→evaluate loop
├── models/          Pre-trained model catalog (see models/README.md)
├── analysis/        25-module scientific analysis (98 Jupyter notebooks)
├── scripts/         Cloud training helpers (Lambda Labs, etc.)
├── modal_app.py     Modal.com cloud GPU orchestration
├── MODAL_ORIENTATION.md / MODAL_MONITOR.md   Modal ops runbooks
├── ORIENTATION.md   Operational reference: CLI commands + data paths
└── README.md        You are here
```

## Getting started

```bash
# Install dependencies
pip install -r forge/requirements.txt

# Verify imports
python -c "from forge.ml import module, data, metrics; from forge.oracle import schema; print('OK')"

# Quick training sanity check
python -m forge.cli.train --fast-dev-run --no-wandb
```

For CLI options and data locations, see [ORIENTATION.md](ORIENTATION.md).
For the E[Q] pipeline, see [eq/README.md](eq/README.md).
