# Analytics Architecture

Deep dive into the analysis system design.

## Data Pipeline

```
Oracle Shards (parquet)
    ↓
Loading (parallel, cached)
    ↓
Feature Extraction (from packed states)
    ↓
Analysis (notebooks/scripts)
    ↓
Results (figures/, tables/)
    ↓
Reports (markdown)
```

## Parquet Schema

| Column | Type | Description |
|--------|------|-------------|
| `state` | int64 | Packed game state (41 bits used) |
| `V` | int8 | Minimax value (-42 to +42, Team 0 perspective) |
| `q0`-`q6` | int8 | Q-values per action (-128 = illegal) |

**Team 0 perspective**: Positive V = good for Team 0.

## Utility Modules

### loading.py

```python
# Find all shards
shard_files = loading.find_shard_files(DATA_DIR)

# Load specific seed
df, seed, decl_id = loading.load_seed(seed=123, base_dir=DATA_DIR)

# Load multiple (parallel)
df = loading.load_seeds([0, 1, 2], base_dir=DATA_DIR, parallel=True)

# Iterate efficiently
for df, seed, decl_id in loading.iterate_shards(DATA_DIR, limit=10):
    process(df)
```

**ShardCache**: LRU cache for frequently accessed shards.

### features.py

Extract features from packed 64-bit states:

```python
depths = features.depth(df['state'])        # 0-28 (trick depth)
players = features.player(df['state'])      # 0-3 (current player)
teams = features.team(df['state'])          # 0-1 (current team)

# Count domino analysis
locations = features.count_locations(df['state'])  # Where each count is
remaining = features.counts_remaining(df['state']) # Which counts uncaptured

# Q-value statistics
q_stats = features.q_stats(df)  # gap, spread, legal count
```

**COUNT_DOMINO_IDS**: The 5 counting dominoes (0-5, 1-4, 2-3, 3-3, 4-4, 5-5, 6-4).

### compression.py

Information-theoretic metrics:

```python
bits = compression.entropy_bits(values)
cond_ent = compression.conditional_entropy(X, Y)
mi = compression.mutual_information(X, Y)
ratio = compression.lzma_ratio(data)  # Kolmogorov proxy
```

### viz.py

Matplotlib helpers:

```python
viz.setup_notebook_style()  # Consistent formatting

# Pre-built plots
viz.plot_v_distribution(V)
viz.plot_v_by_depth(V, depths)
viz.plot_entropy_curve(entropies)
viz.plot_log_log(x, y, fit=True)
viz.plot_q_structure(q_matrix)
```

### navigation.py

State space traversal:

```python
# Pack/unpack states
packed = navigation.pack_state(state_dict)
children = navigation.get_children(packed_state)

# Trace optimal play
pv = navigation.trace_principal_variation(state, state_lookup)

# Build lookup for fast access
lookup = navigation.build_state_lookup_fast(df)

# Track count captures through game tree
captures = navigation.track_count_captures(root_state, lookup)
```

### symmetry.py

Algebraic symmetries:

```python
swapped = symmetry.team_swap(state)       # Swap teams
rotated = symmetry.seat_rotate(state, k)  # Rotate seats by k
canonical = symmetry.canonical_form(state) # Smallest equivalent

# Orbit analysis
orbits = symmetry.enumerate_orbits(states)
sizes = symmetry.orbit_sizes(states)

# Verify V consistency across equivalents
consistent = symmetry.check_v_consistency(state, V, lookup)
```

## Analysis Phases

### Phase 1: Baseline (01-03)

Establish foundational distributions:
- **01_baseline**: V distribution by depth, Q-value structure
- **02_information**: Entropy decomposition, compression bounds
- **03_counts**: Count domino locations, capture basins

### Phase 2: Structure (04-06)

Discover structural properties:
- **04_symmetry**: Exact symmetries (mostly trivial), approximate clustering
- **05_topology**: Level sets (same V), fragmentation analysis
- **06_scaling**: State growth, 4-depth periodicity, DFA

### Phase 3: Synthesis (07-09)

Unify findings:
- **07_synthesis**: Key insights, 5-dimensional manifold
- **08_count_capture_deep**: Lock-in depth, residual decomposition
- **09_path_analysis**: Convergence patterns, temporal dynamics

## Key Findings

1. **Count capture explains 92% of variance** - R² > 0.99 in endgame
2. **No compression from symmetries** - Only 1.005x despite math existence
3. **Strong temporal persistence** - Hurst H = 0.925 (not random walk)
4. **Highly fragmented level sets** - 20-90% disconnected components
5. **4-depth periodicity** - Tied to trick structure (4 plays per trick)
6. **5-dimensional game** - PCA needs exactly 5 components for 95% variance

## Output Organization

```
results/
├── figures/
│   ├── 01a_v_distribution.png
│   ├── 01b_v_by_depth.png
│   └── ...
└── tables/
    ├── 01a_v_stats.csv
    ├── 02a_entropy.csv
    └── ...
```

Naming: `{notebook_id}_{description}.{ext}`

## Integration Points

- **Oracle training** (`forge/`): Validates game tree structure
- **Game engine** (`src/core/`): Reference values for testing
- **Bidding system** (`forge/bidding/`): Bid strength evaluation
