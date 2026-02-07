# Analytics Workflows

Practical patterns for analysis development.

## Running an Existing Analysis

**Interactive:**
```bash
source forge/venv/bin/activate
jupyter notebook forge/analysis/notebooks/01_baseline/01a_v_distribution.ipynb
```

**Batch execution:**
```bash
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=600 \
  forge/analysis/notebooks/01_baseline/01a_v_distribution.ipynb
```

## Adding a New Analysis Notebook

### 1. Create Directory

```bash
mkdir -p forge/analysis/notebooks/NN_topic/
```

Naming: `NN` = two-digit number, `topic` = descriptive slug.

### 2. Standard Setup Block

Every notebook starts with:

```python
# === CONFIGURATION ===
DATA_DIR = "/mnt/d/shards-standard/"
PROJECT_ROOT = "/home/jason/v2/mk5-tailwind"

# === Setup imports ===
import sys
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from forge.analysis.utils import loading, features, viz, navigation
from forge.oracle import schema, tables

viz.setup_notebook_style()
```

### 3. Output Organization

- Figures: `forge/analysis/results/figures/NNx_description.png`
- Tables: `forge/analysis/results/tables/NNx_description.csv`

Example: `03b_count_locations.png`, `03b_count_locations.csv`

## Memory Management

### Shard Size Awareness

| Decl ID | Typical Size |
|---------|--------------|
| 0, 1, 4, 6 | 5-8M rows (small) |
| 2, 3, 5, 7, 8, 9 | 20-75M rows (large) |

### Best Practices

```python
N_SHARDS = 5  # Keep low in notebooks

# Skip large shards
if len(df) > 20_000_000:
    del df
    gc.collect()
    continue

# Always cleanup
del df, states, V
gc.collect()
```

### Python Script Fallback

For memory-intensive work, convert to `.py`:

```bash
python forge/analysis/notebooks/08_count_capture_deep/run_08c.py
```

See `run_08c.py` and `run_08d_manifold.py` as examples.

## Report Writing

### Structure

```
forge/analysis/report/
├── 00_executive_summary.md   # Key findings, open questions
├── 01_baseline.md            # Module-specific details
├── ...
└── 09_path_analysis.md
```

### When to Update

- New significant finding → Update relevant `NN_*.md`
- Cross-cutting insight → Update `00_executive_summary.md`
- Methodology change → Update relevant module report

## Debugging Common Issues

| Issue | Solution |
|-------|----------|
| External drive not mounted | `ls /mnt/d/` to check; mount or use `data/flywheel-shards/` |
| OOM on large shards | Reduce N_SHARDS, skip large decl_ids, use .py scripts |
| Import errors | Verify venv activated, check PROJECT_ROOT in sys.path |
| Stale notebook output | Re-execute with nbconvert, or clear all outputs first |

## Bead Close Protocol

**When closing a bead for analysis work, follow this checklist:**

```
[ ] 1. Update report
    - Add/modify findings in forge/analysis/report/NN_*.md
    - Update 00_executive_summary.md if significant

[ ] 2. Save outputs
    - Figures → results/figures/NNx_description.png
    - Tables → results/tables/NNx_description.csv

[ ] 3. Git status
    git status

[ ] 4. Stage changes
    git add forge/analysis/
    git add <any other changed files>

[ ] 5. Sync beads
    bd sync

[ ] 6. Commit
    git commit -m "analysis: <description of work>"

[ ] 7. Sync beads again (captures close)
    bd sync

[ ] 8. Push
    git push
```

**Why this order:**
- Report updates ensure findings are documented before closing
- bd sync before commit captures any bead updates from work
- bd sync after commit captures the close event
- Push ensures nothing is lost

## Common Commands

```bash
# Development
source forge/venv/bin/activate
jupyter notebook

# Quality
python -m pytest forge/  # If tests exist

# Data check
python -c "from forge.analysis.utils import loading; \
  print(len(loading.find_shard_files('/mnt/d/shards-standard/')))"

# Mount check
ls /mnt/d/shards-standard/train/ | head -5
```

## Request Flow

1. User identifies analysis question
2. Create bead: `bd create --title="..." --type=task`
3. Claim work: `bd update <id> --status=in_progress`
4. Develop notebook/script
5. Generate outputs (figures, tables)
6. Write findings to report
7. Follow **Bead Close Protocol**
8. Close bead: `bd close <id>`
