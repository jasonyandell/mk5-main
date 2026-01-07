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
| `/mnt/d/shards-standard/` | Primary oracle shards (~200GB, external drive) |
| `data/shards-marginalized/` | Imperfect info data: same hand, 3 opponent configs (~92GB) |
| `data/flywheel-shards/` | Smaller local dataset for testing |

**Schema:** `state` (int64), `V` (int8 -42 to +42), `q0`-`q6` (int8 actions, -128 = illegal)

### Marginalized Data (for Imperfect Information Analysis)

**Purpose:** Analyze how V varies when P0's hand is fixed but opponent hands differ.

**Structure:**
- 201 unique base_seeds, 3 opponent configs each (opp0, opp1, opp2)
- Filename: `seed_{BASE_SEED:08d}_opp{OPP_SEED}_decl_{DECL_ID}.parquet`
- P0's hand = `deal_from_seed(base_seed)[0]` (fixed across all 3 opp configs)
- `deal_with_fixed_p0(p0_hand, opp_seed)` reconstructs full deal

**Key finding:** V varies 2-68 points across opponent configs for the same hand!

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
| `seed_db.py` | **DuckDB interface** for SQL queries on parquet (see below) |

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
| 11 | Imperfect Info | Count locks, V variance, bidding heuristics (uses marginalized data) |

## Key Commands

```bash
# Run notebook interactively
jupyter notebook forge/analysis/notebooks/01_baseline/01a_v_distribution.ipynb

# Execute notebook from CLI
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=600 <notebook.ipynb>
```

## Common Pattern (Traditional Loading)

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

## DuckDB Pattern (Preferred for Large Data)

Use `SeedDB` for efficient SQL queries on 100GB+ data without loading into memory:

```python
from forge.analysis.utils.seed_db import SeedDB

db = SeedDB("data/shards-marginalized/train")

# Get root V (depth=28) from a file - fast, no full load
result = db.get_root_v("seed_00000000_opp0_decl_0.parquet")
print(f"Root V: {result.data}, took {result.elapsed_ms:.1f}ms")

# Aggregate root V across many files
result = db.root_v_stats(limit=100)
df = result.data  # DataFrame: file, root_v, rows

# Query with filtering - only scans needed data
result = db.query_columns(
    pattern="*.parquet",
    columns=["state", "V"],
    depth_filter=28,  # Root states only
    limit=1000,
)

# Custom SQL with depth() UDF
result = db.execute("""
    SELECT depth(state) as d, AVG(V) as mean_v
    FROM read_parquet('/mnt/d/shards-standard/train/*.parquet')
    GROUP BY depth(state)
    ORDER BY d DESC
""")
```

**QueryResult fields:** `data`, `elapsed_ms`, `cpu_time_ms`, `io_wait_ms`, `rows_scanned`, `rows_returned`

**When to use DuckDB vs traditional loading:**
- **DuckDB**: Root V extraction, aggregations, filtered queries, large-scale analysis
- **Traditional**: Full shard access for navigation, PV tracing, state-by-state analysis

## Anti-Patterns

| DO NOT | DO INSTEAD |
|--------|------------|
| Load all shards at once | Keep N_SHARDS ≤ 5 (or use DuckDB) |
| Skip memory cleanup | `del df; gc.collect()` after each shard |
| Use notebooks for heavy compute | Convert to .py scripts (see run_08c.py) |
| Forget mount check | `ls /mnt/d/` before assuming data exists |
| Write `get_root_v_fast()` in scripts | Use `SeedDB.get_root_v()` |
| Read CSV intermediate files | Query Parquet directly with DuckDB |

## Bead Close Protocol

When closing a bead for analysis work:

1. **Update report** - Add/update findings in `forge/analysis/report/`
2. **Save outputs** - Figures to `results/figures/`, tables to `results/tables/`
3. **Git commit** - Stage and commit all changes
4. **bd sync** - Sync beads database
5. **Git push** - Push to remote

## Adding a New Analysis

### 1. Pick Module Number
```bash
ls forge/analysis/notebooks/   # Check existing
mkdir -p forge/analysis/notebooks/10_topic/
```

### 2. Required Boilerplate
Every notebook starts with:
```python
# === CONFIGURATION ===
DATA_DIR = "/mnt/d/shards-standard/"
PROJECT_ROOT = "/home/jason/v2/mk5-tailwind"

import sys
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from forge.analysis.utils import loading, features, viz
from forge.oracle import schema

viz.setup_notebook_style()
```

### 3. Data Access Patterns

| To Get | Use |
|--------|-----|
| Shard files list | `loading.find_shard_files(DATA_DIR)` |
| Load one shard | `df, seed, decl_id = schema.load_file(path)` |
| Hands from seed | `hands = schema.deal_from_seed(seed)` → 4 lists of 7 domino IDs |
| Depth (dominoes left) | `features.depth(df['state'].values)` |
| Current player/team | `features.player()`, `features.team()` |
| Unpack state bits | `schema.unpack_state(states)` → remaining, leader, trick_len, p0, p1, p2 |
| Domino pips | `schema.domino_pips(domino_id)` → (high, low) |
| Count points | `tables.DOMINO_COUNT_POINTS[domino_id]` |

### 4. V/Q Semantics (Critical!)

| Concept | Details |
|---------|---------|
| **V perspective** | Always Team 0. Positive = good for Team 0. |
| **Q perspective** | Same. Team 0's turn: argmax(Q). Team 1's turn: argmin(Q). |
| **Player→Team** | P0/P2 = Team 0, P1/P3 = Team 1 |
| **Adjust for player** | Use V as-is for P0/P2; **negate V** for P1/P3 |
| **Depth meaning** | Dominoes remaining (28 at start, 0 at end) |
| **Local indices** | Each player's hand is indexed 0-6 locally |

### 5. Output Naming
- Figures: `results/figures/NNx_description.png`
- Tables: `results/tables/NNx_description.csv`
- Report: `report/NN_topic.md`

### 6. Canonical Representations
```python
# Canonical hand (for grouping across seeds)
def canonical_hand(hand: list[int]) -> tuple[int, ...]:
    return tuple(sorted(hand))

# Canonical state (for symmetry analysis)
from forge.analysis.utils import symmetry
canonical = symmetry.canonical_form(state)
```

## References

- [architecture.md](architecture.md) - System design and utility details
- [workflows.md](workflows.md) - Running analyses, adding notebooks
- `forge/analysis/CLAUDE.md` - Quick setup reference
- `forge/analysis/report/00_executive_summary.md` - Key findings
