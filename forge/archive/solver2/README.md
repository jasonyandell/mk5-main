# Legacy Scripts (Frozen)

Historical scripts from solver2/ preserved for reference.

**DO NOT MODIFY.** Use forge/ for all new work.

## What's Here

- Original GPU solver code
- Various training experiments (train_*.py)
- Diagnostic scripts (q_diagnostic.py, etc.)

## Why Archived

These scripts worked but accumulated slop over time:
- Duplicated model definitions
- Inline GPU/CPU hacks
- Non-deterministic sampling
- Scattered logging

The forge/ directory consolidates everything with:
- PyTorch Lightning structure
- Single source of truth for each component
- Deterministic, reproducible pipelines

Archived: 2025-12-30
