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
| Use `pyarrow.parquet as pq` | Use `SeedDB.query_columns()` |
| Use undefined `PROJECT_ROOT` | Define it at top of script |

## Converting Scripts to SeedDB

When converting `run_*.py` scripts from pyarrow/schema.load_file to SeedDB:

### Common Bugs to Fix

1. **Undefined PROJECT_ROOT** - Many scripts reference it without defining:
   ```python
   # FIX: Add at top of script after imports
   PROJECT_ROOT = "/home/jason/v2/mk5-tailwind"
   DATA_DIR = Path(PROJECT_ROOT) / "data/shards-marginalized/train"
   ```

2. **Replace import statements**:
   ```python
   # REMOVE
   import pyarrow.parquet as pq
   from forge.oracle import schema  # if only used for load_file

   # ADD
   from forge.analysis.utils.seed_db import SeedDB
   ```

3. **Update function signatures** to accept `db` parameter:
   ```python
   # OLD
   def analyze_seed(base_seed: int):
       df, _, _ = schema.load_file(path)

   # NEW
   def analyze_seed(db: SeedDB, base_seed: int):
       result = db.query_columns(files=[filename], columns=['state', 'V'])
       df = result.data
   ```

4. **Initialize and close SeedDB in main()**:
   ```python
   def main():
       db = SeedDB(DATA_DIR)
       for seed in seeds:
           analyze_seed(db, seed)
       db.close()  # Don't forget!
   ```

### Conversion Checklist

- [ ] Define `PROJECT_ROOT` at top
- [ ] Replace `import pyarrow.parquet as pq` with SeedDB import
- [ ] Update `DATA_DIR` to use `Path(PROJECT_ROOT) / "data/..."`
- [ ] Add `db: SeedDB` parameter to analysis functions
- [ ] Replace `pq.read_table()` → `db.query_columns(files=[filename])`
- [ ] Replace `schema.load_file()` → `db.query_columns(files=[filename])`
- [ ] Replace custom `get_root_v_fast()` → `db.get_root_v(filename)`
- [ ] Add `db = SeedDB(DATA_DIR)` at start of main()
- [ ] Add `db.close()` after processing loop
- [ ] Use filename (not full path) for SeedDB methods

## Bead Close Protocol

When closing a bead for analysis work:

1. **Update report** - Add/update findings in `forge/analysis/report/`
2. **Save outputs** - Figures to `results/figures/`, tables to `results/tables/`
3. **Update CLAUDE.md** - Any failed tool call and its fix go to `forge/analysis/CLAUDE.md`
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

## Key Statistical Findings (Cite These)

### The Napkin Formula (Validated by Cross-Validation)
**Only two features matter for E[V] prediction:**
- `n_doubles`: +5.7 points per double (95% CI: [+2.3, +9.2])
- `trump_count`: +3.2 points per trump (95% CI: [+1.3, +4.7])

Full model R² = 0.26, but Napkin model (2 features) has **better CV R² = 0.15** (less overfitting).

### E[V] vs σ(V) Negative Correlation
| Metric | r | 95% CI | Effect Size |
|--------|---|--------|-------------|
| r(E[V], σ[V]) | **-0.381** | [-0.494, -0.256] | Medium |

**Key insight:** Good hands have LOWER variance. Opposite of financial markets.

### Risk is Unpredictable
- σ(V) prediction: R² = 0.08 (95% CI: [0.06, 0.20])
- **CV R² is NEGATIVE** - model is worse than mean prediction
- Risk comes from opponent hands, not your own

### Strongest E[V] Predictors (Bivariate)
| Feature | r with E[V] | p-value |
|---------|-------------|---------|
| n_doubles | **+0.395** | 6.9×10⁻⁹ |
| has_trump_double | +0.242 | 5.6×10⁻⁴ |
| trump_count | +0.229 | 1.1×10⁻³ |

### SHAP Analysis Confirms
- n_doubles: Mean |SHAP| = **4.84** (rank 1)
- trump_count: Mean |SHAP| = **4.39** (rank 2)
- Effects are **68% additive** - interactions are small

### Phase Transition (Order → Chaos → Resolution)
| Phase | Depth | Best-Move Consistency |
|-------|-------|----------------------|
| Opening | 24-28 | 40% |
| Mid-game | 5-23 | 22% (chaos) |
| End-game | 0-4 | **100%** (locked) |

### Count Ownership Dominates
- Count capture explains ~92% of game value variance
- Basin model R² > 0.99 at late game (depth ≤ 12)

### Enrichment Analysis
- **5-5**: 2.8× more common in winners, reduces risk
- **6-0**: 3× more common in losers (worst domino)

### Power Analysis
All key findings have power > 80% at n=200. Main effects need only n≈50.

## Statistical Rigor Requirements

Always follow theme 13 patterns:
1. **Bootstrap CIs**: 1000 iterations, percentile method
2. **Effect sizes**: Report Cohen's d, r, or R² with interpretation
3. **Power analysis**: Verify n is sufficient for effect size
4. **Multiple comparison correction**: BH FDR for > 3 tests
5. **Cross-validation**: 10-fold, 10 repeats for generalization

## References

- [architecture.md](architecture.md) - System design and utility details
- [workflows.md](workflows.md) - Running analyses, adding notebooks
- `forge/analysis/CLAUDE.md` - Complete reference with all findings
- `forge/analysis/report/00_executive_summary.md` - Key findings
