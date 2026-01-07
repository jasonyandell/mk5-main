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
/mnt/d/shards-standard/          # External drive mount (~200GB)
├── train/                        # seed % 1000 < 900
├── val/                          # seed % 1000 in [900, 950)
└── test/                         # seed % 1000 >= 950
```

**Marginalized shards (imperfect info analysis):**
```
data/shards-marginalized/      # External drive mount (~92GB)
└── train/                        # 601 files (201 base_seeds × 3 opponent configs)
```

Marginalized data structure:
- **Filename**: `seed_{BASE_SEED:08d}_opp{OPP_SEED}_decl_{DECL_ID}.parquet`
- **P0 hand fixed**: For each base_seed, P0's hand = `deal_from_seed(base_seed)[0]`
- **3 opponent configs**: opp0, opp1, opp2 shuffle remaining 21 dominoes among P1, P2, P3
- **Use case**: Imperfect information analysis - same hand, different opponent distributions
- **Key insight**: V varies significantly across opponent configs (spread 2-68 points observed)

```python
from forge.oracle.rng import deal_from_seed, deal_with_fixed_p0

# Get P0's hand for a base_seed
p0_hand = deal_from_seed(base_seed)[0]

# Reconstruct the full deal for a specific opponent config
hands = deal_with_fixed_p0(p0_hand, opp_seed)  # hands[0] == p0_hand
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
from forge.analysis.utils.seed_db import SeedDB  # DuckDB interface
from forge.oracle import schema, tables

# For visualization
viz.setup_notebook_style()
```

## DuckDB Interface (SeedDB)

For efficient queries on 100GB+ datasets without loading into memory:

```python
from forge.analysis.utils.seed_db import SeedDB

# Create database connection
db = SeedDB("data/shards-marginalized/train")

# Get root V (depth=28 state) from a single file
result = db.get_root_v("seed_00000000_opp0_decl_0.parquet")
print(f"Root V: {result.data}, took {result.elapsed_ms:.1f}ms")

# Get root V stats across all files
result = db.root_v_stats(limit=100)
df = result.data  # DataFrame: file, root_v, rows

# Query specific columns with filtering
result = db.query_columns(
    files=["seed_00000000_opp0_decl_0.parquet"],
    columns=["state", "V", "q0"],
    where="V > 0",
    limit=1000,
)

# Filter by game depth (dominoes remaining)
result = db.query_columns(
    pattern="*.parquet",
    columns=["state", "V"],
    depth_filter=28,  # Root states only
)

# Register a view for repeated queries
db.register_view("all_shards", "*.parquet")
result = db.execute("SELECT COUNT(*) FROM all_shards")

# Custom SQL with depth UDF
result = db.execute("""
    SELECT depth(state) as d, AVG(V) as mean_v
    FROM read_parquet('/mnt/d/shards-standard/train/*.parquet')
    GROUP BY depth(state)
    ORDER BY d DESC
""")
```

**QueryResult fields:**
- `data`: Query result (DataFrame, scalar, etc.)
- `elapsed_ms`: Wall-clock time
- `cpu_time_ms`: CPU compute time
- `io_wait_ms`: Estimated I/O time (property)
- `rows_scanned`, `rows_returned`, `files_accessed`

**Depth UDF:** `depth(state)` computes dominoes remaining (0-28) using bit operations.

## Loading Data (Traditional)

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

**Loading marginalized data:**
```python
from pathlib import Path
from forge.oracle import schema
from forge.oracle.rng import deal_from_seed, deal_with_fixed_p0

MARG_DIR = Path("data/shards-marginalized/train")

def load_marginalized_group(base_seed: int) -> list[tuple]:
    """Load all 3 opponent configs for a base_seed.

    Returns: [(df, seed, decl_id, opp_seed, hands), ...]
    """
    decl_id = base_seed % 10
    p0_hand = deal_from_seed(base_seed)[0]
    results = []

    for opp_seed in range(3):
        path = MARG_DIR / f"seed_{base_seed:08d}_opp{opp_seed}_decl_{decl_id}.parquet"
        df, seed, decl_id = schema.load_file(path)
        hands = deal_with_fixed_p0(p0_hand, opp_seed)
        results.append((df, seed, decl_id, opp_seed, hands))

    return results
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

## Converting Scripts to SeedDB

When converting existing `run_*.py` scripts to use SeedDB, follow this checklist:

### Common Issues to Fix

1. **Undefined PROJECT_ROOT**: Many scripts use `Path(PROJECT_ROOT)` without defining it
   ```python
   # BAD - PROJECT_ROOT undefined
   DATA_DIR = Path(PROJECT_ROOT) / "data/shards-marginalized/train"

   # GOOD - Define PROJECT_ROOT first
   PROJECT_ROOT = "/home/jason/v2/mk5-tailwind"
   DATA_DIR = Path(PROJECT_ROOT) / "data/shards-marginalized/train"
   ```

2. **Replace pyarrow imports**:
   ```python
   # REMOVE
   import pyarrow.parquet as pq
   from forge.oracle import schema  # if only used for load_file

   # ADD
   from forge.analysis.utils.seed_db import SeedDB
   ```

3. **Replace loading patterns**:
   ```python
   # OLD: pq.read_table() or schema.load_file()
   table = pq.read_table(path, columns=['state', 'V'])
   df, seed, decl_id = schema.load_file(path)

   # NEW: SeedDB.query_columns()
   result = db.query_columns(files=[filename], columns=['state', 'V'])
   df = result.data
   ```

4. **For root V extraction**:
   ```python
   # OLD: Custom get_root_v_fast() function
   def get_root_v_fast(path):
       pf = pq.ParquetFile(path)
       for batch in pf.iter_batches(...):
           # find depth==28

   # NEW: Use SeedDB.get_root_v()
   result = db.get_root_v(filename)  # filename, not full path
   root_v = float(result.data) if result.data is not None else None
   ```

### Conversion Pattern

```python
# 1. Update imports at top
import sys
PROJECT_ROOT = "/home/jason/v2/mk5-tailwind"
sys.path.insert(0, PROJECT_ROOT)

from forge.analysis.utils.seed_db import SeedDB

DATA_DIR = Path(PROJECT_ROOT) / "data/shards-marginalized/train"

# 2. Update function signatures to accept db parameter
def analyze_something(db: SeedDB, base_seed: int) -> dict | None:
    filename = f"seed_{base_seed:08d}_opp0_decl_0.parquet"
    result = db.query_columns(files=[filename], columns=['state', 'V'])
    # ... rest of analysis

# 3. Initialize SeedDB in main()
def main():
    db = SeedDB(DATA_DIR)

    for seed in seeds:
        result = analyze_something(db, seed)

    db.close()  # Don't forget to close!
```

### Key Differences

| Old Pattern | New Pattern |
|-------------|-------------|
| `pq.read_table(path)` | `db.query_columns(files=[filename])` |
| `schema.load_file(path)` | `db.query_columns(files=[filename])` |
| Custom root V extraction | `db.get_root_v(filename)` |
| Full path to file | Filename only (relative to DATA_DIR) |
| Return DataFrame directly | Return `QueryResult`, access `.data` |

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

## Key Statistical Findings

### E[V] vs σ(V) Negative Correlation (12a)

At n=200 seeds, there's a **confirmed negative correlation** between expected value and risk:

| Metric | r | 95% CI | p-value |
|--------|---|--------|---------|
| r(E[V], σ[V]) | **-0.381** | [-0.494, -0.256] | 2.6×10⁻⁸ |
| r(E[V], V_spread) | **-0.398** | [-0.509, -0.274] | 5.4×10⁻⁹ |

**Interpretation**: Good hands (high E[V]) have **lower** variance/risk. This is the opposite of typical financial markets where higher returns require higher risk. Effect size is "medium" by Cohen's conventions (|r| ≈ 0.38-0.40).

### Unified Feature Extraction (12b)

Created `forge/analysis/utils/hand_features.py` to replace duplicated feature extraction in run_11*.py scripts.

**Master feature file**: `results/tables/12b_unified_features.csv` (200 seeds × 20 columns)

**Strongest E[V] predictors**:
| Feature | r with E[V] | p-value |
|---------|-------------|---------|
| n_doubles | **+0.395** | 6.9×10⁻⁹ |
| has_trump_double | +0.242 | 5.6×10⁻⁴ |
| trump_count | +0.229 | 1.1×10⁻³ |

**Key insight**: Doubles are the strongest predictor of E[V] (r = 0.40). Each double improves expected outcome significantly.

### Bootstrap CIs for Regression Coefficients (13a)

95% bootstrap confidence intervals (1000 iterations) for the E[V] regression model:

| Feature | Coefficient | 95% CI | Significant? |
|---------|-------------|--------|--------------|
| n_doubles | **+5.7** | [+2.3, +9.2] | **Yes** |
| trump_count | **+3.2** | [+1.3, +4.7] | **Yes** |
| has_trump_double | +2.8 | [-2.6, +8.4] | No |
| n_voids | +2.8 | [-3.5, +8.9] | No |
| n_6_high | -1.6 | [-5.0, +1.8] | No |

**Key insight**: Only **n_doubles** and **trump_count** have CIs that exclude zero - they are the only statistically significant predictors. Other features (has_trump_double, n_voids, n_6_high) have wide CIs that include zero.

**R² = 0.26** (95% CI: [0.20, 0.40]) - model explains 20-40% of E[V] variance.

### Bootstrap CIs for Risk Formula (13b)

Risk (σ(V)) is **nearly unpredictable** from hand features:

| Feature | Coefficient | 95% CI | Significant? |
|---------|-------------|--------|--------------|
| total_pips | +0.30 | [+0.01, +0.57] | Marginal |
| n_doubles | -1.40 | [-3.32, +0.77] | No |
| n_5_high | -1.09 | [-2.84, +0.63] | No |

**R² = 0.08** (95% CI: [0.06, 0.20]) - model explains only 6-20% of σ(V) variance.

**Key insight**: Risk is fundamentally unpredictable from your hand. The uncertainty in 42 comes from opponent hands, not your own.

### Effect Sizes Summary (13c)

Effect sizes distinguish "statistically significant" from "practically meaningful":

| Finding | Effect Size | Magnitude |
|---------|-------------|-----------|
| n_doubles → E[V] | r = +0.40 | **Medium** |
| E[V] ↔ σ(V) | r = -0.38 | **Medium** |
| ≥2 doubles vs <2 on E[V] | d = +0.76 | **Medium** |
| Hand features → E[V] | R² = 0.26 | **Large** |
| Hand features → σ(V) | R² = 0.08 | **Small** |

**Key insight**: The n_doubles and E[V]-σ(V) relationships have medium effect sizes - practically meaningful, not just statistically significant. Risk prediction is weak (small R²).

### Fisher z-Transform Correlation CIs (13d)

Fisher z-transform confidence intervals for all correlations (n=200):

**Significant correlations (10 of 16)**:
| Comparison | r | 95% CI |
|------------|---|--------|
| n_doubles vs E[V] | +0.40 | [+0.27, +0.51] |
| E[V] vs σ(V) | -0.38 | [-0.49, -0.26] |
| has_trump_double vs E[V] | +0.24 | [+0.11, +0.37] |
| trump_count vs E[V] | +0.23 | [+0.09, +0.36] |
| n_voids vs E[V] | +0.20 | [+0.06, +0.33] |
| count_points vs E[V] | +0.20 | [+0.06, +0.33] |

**Key insight**: Many features show bivariately significant correlations with E[V], but in multivariate regression (13a), only n_doubles and trump_count survive. This indicates that has_trump_double, n_voids, and count_points are largely explained by their association with the two key predictors.

### SHAP Analysis on E[V] Model (14a)

SHAP (SHapley Additive exPlanations) analysis using GradientBoostingRegressor + TreeExplainer:

| Feature | Mean |SHAP| | Rank |
|---------|-------------|------|
| n_doubles | **4.84** | 1 |
| trump_count | **4.39** | 2 |
| n_singletons | 2.17 | 3 |
| count_points | 2.17 | 4 |
| total_pips | 2.00 | 5 |

**Key insight**: SHAP confirms n_doubles and trump_count as top predictors. Unlike linear regression, GradientBoosting captures nonlinear effects, showing that n_singletons, count_points, and total_pips also contribute meaningful |SHAP| values (~2 points each). Waterfall plots provide per-hand explainability.

## Useful One-Liners

```bash
# Count shard files
ls /mnt/d/shards-standard/*/*.parquet | wc -l

# Check data is accessible
python -c "from forge.analysis.utils import loading; print(len(loading.find_shard_files('/mnt/d/shards-standard/')))"
```
