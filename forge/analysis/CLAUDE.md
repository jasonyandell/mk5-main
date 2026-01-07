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

### SHAP Analysis on σ(V) Model (14b)

SHAP analysis on risk/variance model:

| Metric | Value |
|--------|-------|
| CV R² | **-0.34** (worse than mean prediction) |
| Train R² | 0.67 (pure overfitting) |

**Key insight**: The σ(V) model has negative CV R², proving risk is unpredictable from hand features. n_doubles has 6.9× higher SHAP importance for E[V] than for σ(V). Focus bidding decisions on expected value (doubles, trumps); outcome variance is determined by unknown opponent hands.

### SHAP Interaction Values (14c)

SHAP interaction analysis reveals main effects dominate:

| Feature | Main Effect | Total Interactions | Main/Total |
|---------|-------------|-------------------|------------|
| n_doubles | 4.92 | 2.27 | **68%** |
| trump_count | 3.94 | 2.18 | **64%** |

**Key insight**: n_doubles and trump_count effects are largely **additive** (60-70% main effect). n_doubles × trump_count interaction (0.37) is much smaller than main effects (4.9, 3.9). This validates the simple napkin formula without interaction terms.

### Risk-Return Scatter Plot (15a)

Publication-quality visualization of the headline finding:
- **r = -0.38** (95% CI: [-0.49, -0.26])
- Good hands have LOWER risk (inverse of typical markets)
- Points colored by n_doubles show higher doubles → upper left quadrant

Available formats:
- `results/figures/15a_risk_return_scatter.png` (300 DPI, annotated)
- `results/figures/15a_risk_return_scatter.pdf` (vector for publication)
- `results/figures/15a_risk_return_clean.png` (simplified version)

### UMAP Hand Space (15b)

UMAP dimensionality reduction (10 features → 2D) reveals:
- **No distinct clusters** - hand space is a continuous manifold
- Gradual E[V] gradient across embedding
- has_trump_double and n_voids drive UMAP structure
- Best/worst hands occupy distinct regions

Available at:
- `results/figures/15b_umap_hand_space.png` - E[V] and σ(V) coloring
- `results/figures/15b_umap_annotated.png` - With extreme hands labeled
- `results/tables/15b_umap_coordinates.csv` - Coordinates for all hands

### Pareto Frontier (15c)

Pareto optimality analysis (max E[V], min σ(V)):
- **Only 3 hands (1.5%) are Pareto-optimal** - all have E[V]=42, σ(V)=0
- 197 hands (98.5%) are dominated
- Degenerate frontier due to inverse risk-return relationship
- No meaningful risk-return tradeoff exists in Texas 42

Available at:
- `results/figures/15c_pareto_frontier.png` - Visualization
- `results/tables/15c_pareto_frontier.csv` - Classification

### Power Analysis (13e)

Statistical power analysis confirms n=200 is sufficient for all key findings:

| Analysis | Effect Size | Power | n for 80% |
|----------|-------------|-------|-----------|
| r(E[V], σ[V]) | -0.38 | 1.000 | 51 |
| r(n_doubles, E[V]) | +0.40 | 1.000 | 46 |
| r(trump_count, E[V]) | +0.23 | 0.911 | 145 |
| d(≥2 doubles vs <2) | 0.76 | 1.000 | 58 total |
| R²(hand→E[V]) | 0.26 | 1.000 | 57 |
| R²(hand→σ[V]) | 0.08 | 0.810 | 197 |

**Key insight**: All key findings have power >80%. Main effects (E[V]-σ[V] correlation, n_doubles) have power ≈ 1.00 and would only need n≈50. No immediate scale-up needed. To detect smaller effects (r=0.1), would need n≈782.

Available at:
- `results/tables/13e_power_analysis.csv` - Summary table
- `results/figures/13e_power_curves.png` - Power curves

### Multiple Comparison Correction (13f)

Benjamini-Hochberg FDR correction applied to all 16 correlation tests:

| Method | Significant | Type |
|--------|-------------|------|
| Uncorrected | 10 | None |
| **BH FDR** | **9** | FDR |
| Bonferroni | 5 | FWER |

**Key insight**: 9 of 10 correlations survive BH FDR correction. Only total_pips vs σ(V) (r=0.15, p_adj=0.056) is lost - a marginal finding anyway. All core findings remain robust:
- E[V] vs σ(V) (r = -0.38, p_adj < 0.0001)
- n_doubles vs E[V] (r = +0.40, p_adj < 0.0001)
- trump_count vs E[V] (r = +0.23, p_adj = 0.0036)

Available at:
- `results/tables/13f_multiple_comparison.csv` - Adjusted p-values
- `results/figures/13f_multiple_comparison.png` - BH procedure visualization

### Cross-Validation (13g)

K-fold cross-validation (10-fold, 10 repeats) for regression models:

**E[V] Prediction:**
| Model | Train R² | CV R² | Overfit |
|-------|----------|-------|---------|
| **Napkin (2 features)** | 0.23 | **0.15** | 1.5x |
| Full (10 features) | 0.26 | 0.11 | 2.4x |

**σ(V) Prediction:** All models have **negative CV R²** (fails completely)

**Key insight**: The napkin formula (n_doubles, trump_count) generalizes best with lowest overfitting. Full model hurts generalization. σ(V) prediction is confirmed impossible.

Available at:
- `results/tables/13g_cross_validation.csv` - Summary
- `results/figures/13g_cross_validation.png` - Train vs CV bars
- `results/figures/13g_learning_curve.png` - Learning curve

### Phase Transition (15d)

Best-move consistency across game depth reveals three distinct phases:

| Phase | Depth | Consistency | States |
|-------|-------|-------------|--------|
| Early game | 24-28 | 40% | 18 |
| Mid-game | 5-23 | 22% | 147,529 |
| End-game | 0-4 | **100%** | 19,472 |

**Key insight**: Game transitions from **order → chaos → resolution**:
1. **Opening**: Few states, declarers control the game
2. **Mid-game**: Maximum uncertainty, multiple good strategies exist
3. **End-game**: Outcomes locked in, mechanical play

Available at:
- `results/figures/15d_phase_transition.png` - Progress-based view
- `results/figures/15d_phase_by_depth.png` - Depth-based view with state counts

### Word2Vec Domino Embeddings (16a)

Word2Vec trained on hand co-occurrence (40K hands, skip-gram):

| Comparison | Mean Similarity |
|------------|-----------------|
| Double-to-double | **0.079** |
| Double-to-non-double | 0.071 |
| Random baseline | 0.069 |

**Key insight**: Domino embeddings show **weak structure**. Doubles cluster slightly (11% higher sim), but suit structure is near-random. The random deal mechanism means hands don't have "themes" - strategic value comes from game context, not co-occurrence.

Available at:
- `results/tables/16a_word2vec_embeddings.csv` - 32D embeddings
- `results/tables/16a_word2vec_similarity.csv` - 28×28 similarity matrix
- `results/models/16a_word2vec.model` - Trained gensim model
- `results/figures/16a_word2vec_tsne.png` - t-SNE visualization

### UMAP Domino Embeddings (16b)

UMAP projection of Word2Vec embeddings confirms weak structure:

- **No strong clusters** emerge in 2D projection
- Doubles partially group but not tightly
- Suit membership doesn't create clusters
- Category separation ratios ~1.0 (random)

**Key insight**: Dominoes are strategically undifferentiated in co-occurrence space. Value comes from game context (trump, position), not hand composition.

Available at:
- `results/figures/16b_umap_dominoes.png` - 2×2 visualization grid
- `results/figures/16b_umap_annotated.png` - Annotated view
- `results/tables/16b_umap_coordinates.csv` - UMAP coordinates

### Domino Interaction Matrix (16c)

Synergy scores for domino pairs (observed - expected under additive model):

**Top single-domino effects**:
- 4-4: **+8.21**, 5-5: **+7.67**, 5-0: +6.12 (doubles dominate)
- 6-0: **-9.55** (worst - weak trick winner)

**Synergy range**: -11.86 to +14.61

**Key insight**: The additive model mostly works - most pair synergies near zero. Some non-additive interactions exist (e.g., 2-2 + 3-3 = -11.9, two doubles can conflict).

Available at:
- `results/tables/16c_interaction_matrix.csv` - 28×28 synergy matrix
- `results/tables/16c_pair_synergies.csv` - Pairs ranked by synergy
- `results/tables/16c_single_effects.csv` - Single-domino effects

## Useful One-Liners

```bash
# Count shard files
ls /mnt/d/shards-standard/*/*.parquet | wc -l

# Check data is accessible
python -c "from forge.analysis.utils import loading; print(len(loading.find_shard_files('/mnt/d/shards-standard/')))"
```
