---
name: texas-42-analytics
description: Oracle data analytics - activates for analysis notebooks, utility functions, feature extraction, data loading patterns, compression metrics, visualization, exploring game structure findings, and managing analysis workflow with beads.
---

# Texas 42 Analytics Skill

Analysis pipeline for exploring 300M+ game states from the oracle. Uses Jupyter notebooks with reusable Python utilities.

## Quick Setup

```bash
source forge/venv/bin/activate  # or .venv
python -c "from forge.analysis.utils import loading, features, viz; print('OK')"
```

## Data Locations

| Location | Purpose |
|----------|---------|
| `/mnt/d/shards-standard/` | Primary oracle shards (external drive) |
| `data/flywheel-shards/` | Smaller local dataset for testing |

**Schema:** `state` (int64), `V` (int8 -42 to +42), `q0`-`q6` (int8 actions, -128 = illegal)

## Directory Structure

```
forge/analysis/
├── utils/           # Reusable Python modules
├── notebooks/       # Numbered analysis modules (00-09)
├── results/
│   ├── figures/     # PNG visualizations
│   └── tables/      # CSV statistical outputs
└── report/          # Markdown findings (00-09)
```

## Utility Modules

| Module | Purpose |
|--------|---------|
| `loading.py` | Shard discovery, parallel loading, caching |
| `features.py` | Extract depth/player/team/counts from packed states |
| `compression.py` | Entropy, mutual information, lzma ratio |
| `viz.py` | Matplotlib helpers with consistent styling |
| `navigation.py` | State traversal, principal variation tracing |
| `symmetry.py` | Canonical forms, orbit enumeration |

## Analysis Modules

| # | Topic | Focus |
|---|-------|-------|
| 00 | Quickstart | Setup verification, data loading intro |
| 01 | Baseline | V/Q distributions by depth |
| 02 | Information | Entropy, compression bounds |
| 03 | Counts | Count domino locations and capture |
| 04 | Symmetry | Exact/approximate equivalences |
| 05 | Topology | Level sets, Reeb graphs |
| 06 | Scaling | State growth, temporal correlations |
| 07 | Synthesis | Unified findings, minimal representations |
| 08 | Count Capture Deep | Lock-in, residuals, manifold structure |
| 09 | Path Analysis | Convergence, geometry, temporal dynamics |

## Key Commands

```bash
# Run notebook interactively
jupyter notebook forge/analysis/notebooks/01_baseline/01a_v_distribution.ipynb

# Execute notebook from CLI
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=600 <notebook.ipynb>
```

## Common Pattern

```python
from forge.analysis.utils import loading, features, viz
DATA_DIR = "/mnt/d/shards-standard/"

# Load data
df, seed, decl_id = loading.load_seed(seed=123, base_dir=DATA_DIR)

# Extract features
depths = features.depth(df['state'])
teams = features.team(df['state'])

# Visualize
viz.setup_notebook_style()
viz.plot_v_distribution(df['V'])
```

## Anti-Patterns

| DO NOT | DO INSTEAD |
|--------|------------|
| Load all shards at once | Keep N_SHARDS ≤ 5 |
| Skip memory cleanup | `del df; gc.collect()` after each shard |
| Use notebooks for heavy compute | Convert to .py scripts (see run_08c.py) |
| Forget mount check | `ls /mnt/d/` before assuming data exists |

## Bead Close Protocol

When closing a bead for analysis work:

1. **Update report** - Add/update findings in `forge/analysis/report/`
2. **Save outputs** - Figures to `results/figures/`, tables to `results/tables/`
3. **Git commit** - Stage and commit all changes
4. **bd sync** - Sync beads database
5. **Git push** - Push to remote

## References

- [architecture.md](architecture.md) - System design and utility details
- [workflows.md](workflows.md) - Running analyses, adding notebooks
- `forge/analysis/CLAUDE.md` - Quick setup reference
- `forge/analysis/report/00_executive_summary.md` - Key findings
