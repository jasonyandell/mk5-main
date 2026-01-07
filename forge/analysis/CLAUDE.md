# Analysis Notebooks - Claude Context

Working knowledge for analysis notebook development.

## Environment Setup

```bash
# Activate the forge venv (from project root)
source forge/venv/bin/activate

# Or use the root .venv if forge/venv doesn't exist
source .venv/bin/activate

# Verify imports work
python -c "from forge.analysis.utils import loading, features, viz; print('OK')"
```

## Data Locations

**Oracle shards (primary data source):**
```
/mnt/d/shards-standard/          # External drive mount
├── train/                        # seed % 1000 < 900
├── val/                          # seed % 1000 in [900, 950)
└── test/                         # seed % 1000 >= 950
```

**Local flywheel shards (smaller dataset):**
```
data/flywheel-shards/            # ~22 files, for quick testing
```

**NOT here:** `forge/data/` doesn't exist - data lives in `data/` at project root or on external drive.

## Notebook Configuration Pattern

Every notebook should start with:
```python
# === CONFIGURATION ===
DATA_DIR = "/mnt/d/shards-standard/"
PROJECT_ROOT = "/home/jason/v2/mk5-tailwind"

# === Setup imports ===
import sys
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
```

## Key Imports

```python
from forge.analysis.utils import loading, features, viz, navigation
from forge.oracle import schema, tables

# For visualization
viz.setup_notebook_style()
```

## Loading Data

```python
# Find all shard files
shard_files = loading.find_shard_files(DATA_DIR)

# Load a single shard
df, seed, decl_id = schema.load_file(shard_files[0])

# Load specific seed
df, seed, decl_id = loading.load_seed(seed=123, base_dir=DATA_DIR)

# Load multiple seeds (parallel)
df = loading.load_seeds([0, 1, 2], base_dir=DATA_DIR)
```

## Parquet Schema

Each shard file contains:
- `state`: int64 - Packed game state (41 bits)
- `V`: int8 - Minimax value (-42 to +42, Team 0 perspective)
- `q0`-`q6`: int8 - Q-values for each action (-128 = illegal)

## Count Dominoes

The 5 "count" dominoes worth points:
```python
from forge.analysis.utils import features
from forge.oracle import schema, tables

for d in features.COUNT_DOMINO_IDS:
    pips = schema.domino_pips(d)
    points = tables.DOMINO_COUNT_POINTS[d]
    print(f"{pips[0]}-{pips[1]}: {points} points")
```

## Output Locations

- Figures: `forge/analysis/results/figures/`
- Tables: `forge/analysis/results/tables/`
- Reports: `forge/analysis/report/`

## Gotchas

1. **Mount check**: `/mnt/d/` may not be mounted. Check with `ls /mnt/d/` first.
2. **Large files**: Each shard is 75-275MB. Don't load too many at once.
3. **State packing**: Use `schema.unpack_state()` to decode packed states.
4. **Team 0 perspective**: All V/Q values are from Team 0's viewpoint (positive = good for Team 0).

## Running Notebooks

**Execute from command line** (more reliable than IDE):
```bash
source forge/venv/bin/activate
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 <notebook.ipynb>
```

**Memory constraints**: Shard sizes vary dramatically:
- Small shards: ~5-8M rows
- Large shards: 20-75M rows (decl_id 2,3,5,7,8,9 tend to be larger)

Keep `N_SHARDS` low (3-5) for notebooks, or use Python scripts that skip large shards:
```python
# Skip large shards
if len(df) > 20_000_000:
    del df
    gc.collect()
    continue

# Always cleanup after each shard
del df, state_to_idx, V, Q, states
gc.collect()
```

**Python scripts vs notebooks**: For memory-intensive analysis, convert notebook logic to a `.py` script (see `run_08c.py`, `run_08d_manifold.py` as examples). Scripts handle memory better than jupyter kernels.

## Useful One-Liners

```bash
# Count shard files
ls /mnt/d/shards-standard/*/*.parquet | wc -l

# Check data is accessible
python -c "from forge.analysis.utils import loading; print(len(loading.find_shard_files('/mnt/d/shards-standard/')))"
```
