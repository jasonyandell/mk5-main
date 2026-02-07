# Structural Analysis Survey: Texas 42 Oracle Data

## Overview

You have 200GB of solved game: every reachable state with its minimax value V ∈ [-42, +42] and Q-values for all legal moves. This document surveys analytical approaches to discover hidden structure—compressible patterns, symmetries, topological features, or a derivable closed form.

The goal isn't prediction. It's understanding. If structure exists, we find it. If not, we know.

---

## 1. Baseline Characterization

Before hunting for exotic structure, establish the basics.

### 1.1 Distribution Profiles

```python
# Per seed, per depth
for seed in seeds:
    for depth in range(28, 0, -1):  # remaining dominoes
        states = get_states_at_depth(seed, depth)
        record(
            seed=seed,
            depth=depth,
            n_states=len(states),
            v_mean=np.mean(states.V),
            v_std=np.std(states.V),
            v_min=np.min(states.V),
            v_max=np.max(states.V),
            v_unique=len(np.unique(states.V)),
            v_entropy=entropy(np.bincount(states.V + 42, minlength=85)),
        )
```

**What to look for:**
- Does `n_states` follow a power law with depth? (fractal branching)
- Does `v_unique` saturate early? (low intrinsic dimensionality)
- Does `v_entropy` scale logarithmically or slower? (structure)
- Is `v_std` predictable from depth? (exploitable for compression)

### 1.2 Q-Value Structure

```python
# For each state, characterize the decision landscape
for state in states:
    legal_q = [q for q in state.q0_q6 if q != -128]
    record(
        n_legal=len(legal_q),
        q_spread=max(legal_q) - min(legal_q),
        q_gap=sorted(legal_q)[-1] - sorted(legal_q)[-2] if len(legal_q) > 1 else 0,
        n_optimal=sum(1 for q in legal_q if q == max(legal_q)),  # ties
    )
```

**What to look for:**
- How often is there a unique optimal move vs ties?
- Distribution of q_spread: fat tails = volatile decisions, tight = forced
- Correlation between q_gap and depth (do decisions get easier or harder?)

---

## 2. Symmetry and Equivalence

The true state space may be much smaller than the apparent one.

### 2.1 Exact Symmetries

Texas 42 has structural symmetries:

| Symmetry | Group | Action |
|----------|-------|--------|
| Team reflection | Z₂ | Swap teams 0↔1, negate V |
| Seat rotation | Z₂ | Rotate by 2 seats (within teams) |
| Suit permutation | S₆ or smaller | Relabel non-trump suits |

```python
def canonical_form(state, seed):
    """Reduce state to canonical representative under symmetry group."""
    # Generate all symmetric variants
    # Return lexicographically smallest (state, sign)
    # sign indicates if V should be negated
    ...

# Count distinct orbits
orbits = defaultdict(list)
for state in all_states:
    canon, sign = canonical_form(state)
    orbits[canon].append((state, sign))

print(f"Raw states: {len(all_states)}")
print(f"Orbits: {len(orbits)}")
print(f"Compression ratio: {len(all_states) / len(orbits):.2f}x")
```

**What to look for:**
- Compression ratio from symmetry alone
- Orbit size distribution (uniform = clean symmetry, skewed = partial symmetry)

### 2.2 Approximate Equivalence

States may be "morally equivalent" even if not exactly symmetric.

```python
# Cluster states by V and local structure
from sklearn.cluster import DBSCAN

features = extract_features(states)  # remaining counts, trick state, etc.
clusters = DBSCAN(eps=0.5).fit(np.column_stack([features, V]))

# Within each cluster, is V constant?
for cluster_id in np.unique(clusters.labels_):
    cluster_V = V[clusters.labels_ == cluster_id]
    print(f"Cluster {cluster_id}: V variance = {np.var(cluster_V):.4f}")
```

**What to look for:**
- Clusters where V variance is near zero = equivalence classes
- Features that perfectly partition these clusters = sufficient statistics

---

## 3. Information-Theoretic Analysis

How much information does V actually contain?

### 3.1 Entropy Decomposition

```python
# Total entropy of V
H_V = entropy(V)

# Conditional entropy given features
def conditional_entropy(V, features):
    """H(V|features) - remaining uncertainty after knowing features."""
    H = 0
    for feature_val in np.unique(features):
        mask = features == feature_val
        p = mask.sum() / len(V)
        H += p * entropy(V[mask])
    return H

# Information gain from each feature
features_to_test = [
    ('depth', remaining_count(states)),
    ('leader', leader),
    ('counts_remaining', count_dominoes_remaining(states)),
    ('count_locations', who_holds_counts(states, seed)),
    # ... more
]

for name, feature in features_to_test:
    H_cond = conditional_entropy(V, feature)
    I = H_V - H_cond  # mutual information
    print(f"{name}: I(V;feature) = {I:.3f} bits, H(V|feature) = {H_cond:.3f} bits")
```

**What to look for:**
- Which features eliminate the most uncertainty?
- Does any feature set drive H(V|features) → 0? (V is deterministic function)
- Diminishing returns curve: how many features to capture 90%, 99%, 99.9%?

### 3.2 Kolmogorov Complexity Estimation

```python
import lzma

def compressibility(data: bytes) -> float:
    """Ratio of compressed to uncompressed size."""
    compressed = lzma.compress(data, preset=9)
    return len(compressed) / len(data)

# Serialize V values in different orderings
v_by_depth = serialize_v_depth_order(states)
v_by_state_order = serialize_v_state_order(states)
v_random = serialize_v_random_order(states)

print(f"Depth order: {compressibility(v_by_depth):.3f}")
print(f"State order: {compressibility(v_by_state_order):.3f}")
print(f"Random order: {compressibility(v_random):.3f}")
```

**What to look for:**
- If depth ordering compresses much better, V has depth-coherent structure
- If all orderings compress similarly, structure is global not local
- Absolute ratio: 0.1 = very structured, 0.9 = near random

### 3.3 Minimum Description Length

```python
# Fit progressively complex models, track description length
# MDL = model_bits + data_given_model_bits

models = [
    ("constant", lambda s: 0),
    ("depth_mean", lambda s: depth_means[depth(s)]),
    ("linear_features", lambda s: w @ features(s)),
    ("decision_tree_d5", fit_tree(max_depth=5)),
    ("decision_tree_d10", fit_tree(max_depth=10)),
    # ...
]

for name, model in models:
    predictions = model(states)
    residuals = V - predictions
    model_bits = model_complexity(model)
    residual_bits = entropy(residuals) * len(residuals)
    mdl = model_bits + residual_bits
    print(f"{name}: MDL = {mdl:.0f} bits (model: {model_bits}, residual: {residual_bits})")
```

**What to look for:**
- Where does MDL hit minimum? That's the "right" model complexity
- Huge drop from constant → simple model = strong structure
- Plateau = you've found the irreducible complexity

---

## 4. Fractal and Scaling Analysis

Does the structure repeat across scales?

### 4.1 State Count Scaling

```python
# How does reachable state count scale with depth?
state_counts = [len(states_at_depth(d)) for d in range(28, 0, -1)]

# Log-log regression
log_depth = np.log(np.arange(1, 29))
log_count = np.log(state_counts)
slope, intercept = np.polyfit(log_depth, log_count, 1)

print(f"N(d) ~ d^{slope:.2f}")
# slope ≈ integer suggests polynomial growth (combinatorial)
# non-integer suggests fractal dimension
```

### 4.2 Value Landscape Roughness

```python
# Compute "roughness" at each scale
def landscape_roughness(V, states, scale):
    """Average |V_i - V_j| for states i,j that are `scale` moves apart."""
    differences = []
    for s in states:
        neighbors = get_states_n_moves_away(s, scale)
        for n in neighbors:
            differences.append(abs(V[s] - V[n]))
    return np.mean(differences)

scales = [1, 2, 3, 4, 5, 7, 10, 14]
roughness = [landscape_roughness(V, states, s) for s in scales]

# Log-log plot
# If linear: roughness ~ scale^H where H is Hurst-like exponent
```

**What to look for:**
- Linear log-log = self-affine value landscape
- H < 0.5: value changes are anti-persistent (oscillating)
- H > 0.5: value changes are persistent (trending)
- H = 0.5: random walk (no exploitable structure at this level)

### 4.3 Principal Variation Time Series

```python
def extract_pv(seed, decl_id, df):
    """Extract V along the principal variation (optimal play sequence)."""
    state = initial_state(seed, decl_id)
    pv_values = [get_V(state, df)]
    
    while not terminal(state):
        q_values = get_Q(state, df)
        team = current_team(state)
        if team == 0:
            best_move = np.argmax(q_values)
        else:
            best_move = np.argmin([q if q != -128 else 999 for q in q_values])
        state = apply_move(state, best_move)
        pv_values.append(get_V(state, df))
    
    return np.array(pv_values)

# Compute Hurst exponent for each seed
from hurst import compute_Hc

H_values = []
for seed in seeds:
    pv = extract_pv(seed, decl_id, df)
    H, _, _ = compute_Hc(pv, kind='change', simplified=True)
    H_values.append(H)

print(f"Mean Hurst: {np.mean(H_values):.3f} ± {np.std(H_values):.3f}")
```

### 4.4 Detrended Fluctuation Analysis

More robust than Hurst for non-stationary series:

```python
def dfa(series, scales=None):
    """Detrended Fluctuation Analysis."""
    N = len(series)
    if scales is None:
        scales = np.logspace(0.5, np.log10(N//4), 20).astype(int)
        scales = np.unique(scales)
    
    # Cumulative sum (profile)
    profile = np.cumsum(series - np.mean(series))
    
    fluctuations = []
    for scale in scales:
        # Divide into windows
        n_windows = N // scale
        F2 = 0
        for i in range(n_windows):
            segment = profile[i*scale:(i+1)*scale]
            # Fit and remove linear trend
            x = np.arange(scale)
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)
            F2 += np.mean((segment - trend)**2)
        fluctuations.append(np.sqrt(F2 / n_windows))
    
    # Fit log-log slope
    coeffs = np.polyfit(np.log(scales), np.log(fluctuations), 1)
    alpha = coeffs[0]  # DFA exponent
    
    return alpha, scales, fluctuations

# alpha ≈ 0.5: white noise
# alpha ≈ 1.0: 1/f noise (pink)
# alpha ≈ 1.5: Brownian motion
# alpha > 1: non-stationary, long-range correlations
```

---

## 5. Topological Data Analysis

Shape of the value landscape.

### 5.1 Level Set Analysis

```python
# For each possible V value, characterize the level set
for v in range(-42, 43):
    level_set = states[V == v]
    if len(level_set) == 0:
        continue
    
    # Build adjacency graph within level set
    G = build_move_graph(level_set)
    n_components = nx.number_connected_components(G)
    
    record(v=v, n_states=len(level_set), n_components=n_components)
```

**What to look for:**
- Few components = V defines coherent regions
- Many components = V is fragmented (less structure)
- Pattern in component count vs V (e.g., more fragmentation at V ≈ 0?)

### 5.2 Persistent Homology

```python
import gudhi

def persistent_homology(states, V, max_dim=2):
    """
    Compute persistence diagram using V as filtration function.
    """
    # Build filtered simplicial complex
    # Vertices: states
    # Edges: legal transitions
    # Filter value: max(V) of vertices in simplex
    
    st = gudhi.SimplexTree()
    
    for i, s in enumerate(states):
        st.insert([i], filtration=V[i])
    
    for i, s in enumerate(states):
        for j, neighbor in enumerate(get_neighbors(s)):
            if j > i:  # avoid duplicates
                filt = max(V[i], V[j])
                st.insert([i, j], filtration=filt)
    
    st.compute_persistence()
    return st.persistence()

# Analyze persistence diagram
persistence = persistent_homology(states, V)
lifetimes = [(death - birth) for dim, (birth, death) in persistence if death != float('inf')]

# Long-lived features are "real" structure
# Short-lived are noise
plt.hist(lifetimes, bins=50)
plt.xlabel("Lifetime")
plt.title("Persistence Barcode")
```

**What to look for:**
- Long bars = robust topological features
- Gaps in the barcode = natural scales/thresholds in V
- Betti numbers: β₀ = components, β₁ = loops, β₂ = voids

### 5.3 Mapper Algorithm

Topological summary that's more interpretable:

```python
import kmapper as km

mapper = km.KeplerMapper()

# Project states to lower dimension
projected = mapper.fit_transform(features, projection="l2norm")

# Build simplicial complex
graph = mapper.map(
    projected,
    features,
    cover=km.Cover(n_cubes=15, perc_overlap=0.5),
    clusterer=sklearn.cluster.DBSCAN(eps=0.5)
)

# Visualize
html = mapper.visualize(graph, color_values=V, title="Texas 42 State Space")
```

**What to look for:**
- Clusters = regions of state space
- Connections = how regions relate
- Color by V: smooth gradients = structure, chaotic = less structure

### 5.4 Reeb Graph

```python
def reeb_graph(states, V, neighbors_fn):
    """
    Contract level sets of V to points, preserve adjacency.
    
    Returns: graph where nodes are connected components of level sets
    """
    # Group states by V
    level_sets = defaultdict(list)
    for s in states:
        level_sets[V[s]].append(s)
    
    # Find connected components within each level set
    reeb_nodes = []
    state_to_node = {}
    
    for v, level_states in level_sets.items():
        G = build_subgraph(level_states, neighbors_fn)
        for component in nx.connected_components(G):
            node_id = len(reeb_nodes)
            reeb_nodes.append({'v': v, 'size': len(component)})
            for s in component:
                state_to_node[s] = node_id
    
    # Add edges between adjacent level set components
    reeb_edges = set()
    for s in states:
        for n in neighbors_fn(s):
            if state_to_node[s] != state_to_node[n]:
                edge = tuple(sorted([state_to_node[s], state_to_node[n]]))
                reeb_edges.add(edge)
    
    return reeb_nodes, reeb_edges
```

**What to look for:**
- Simple Reeb graph (few branches) = V is "nice" function
- Many branches at certain V values = critical points, decision boundaries
- Linear chain = monotonic resolution

---

## 6. Count Domino Analysis

The boulders in the river.

### 6.1 Count Domino Identification

```python
# The count dominoes (pips sum to 5 or 10)
COUNT_DOMINOES = {
    (1, 4): 5,   # 5-count
    (2, 3): 5,   # 5-count
    (0, 5): 5,   # 5-count
    (5, 5): 10,  # 10-count (double 5)
    (4, 6): 10,  # 10-count
    (3, 6): 10,  # WAIT - this is 9, not count
    # Actually: 5-count = sum to 5, 10-count = sum to 10
    # (0,5), (1,4), (2,3) = 5-count
    # (4,6), (5,5) = 10-count
    # Plus (6,4) same as (4,6)
}

# Given a seed, identify count domino locations
def count_analysis(seed):
    hands = deal_from_seed(seed)
    counts_by_player = defaultdict(list)
    
    for player, hand in enumerate(hands):
        for domino_id in hand:
            pips = domino_pips(domino_id)
            total = sum(pips)
            if total == 5 or total == 10:
                counts_by_player[player].append((domino_id, total))
    
    return counts_by_player
```

### 6.2 Basin Analysis

```python
def terminal_count_outcome(seed, terminal_state):
    """
    At terminal state, determine which team captured each count domino.
    Returns: dict mapping domino_id -> team (0 or 1)
    """
    # This requires tracking who won which tricks
    # and which counts were in those tricks
    ...

def classify_by_outcome(seed, states, V):
    """
    Partition states by eventual count capture outcome under optimal play.
    """
    # For terminal states, outcome is known
    # For non-terminal, trace PV to terminal
    
    outcomes = {}
    for s in states:
        if is_terminal(s):
            outcomes[s] = terminal_count_outcome(seed, s)
        else:
            terminal = trace_to_terminal(s)  # follow PV
            outcomes[s] = terminal_count_outcome(seed, terminal)
    
    # Group by outcome
    basins = defaultdict(list)
    for s, outcome in outcomes.items():
        key = tuple(sorted(outcome.items()))
        basins[key].append((s, V[s]))
    
    return basins

# Analyze within-basin variance
for outcome, state_values in basins.items():
    vs = [v for s, v in state_values]
    print(f"Outcome {outcome}: n={len(vs)}, V_mean={np.mean(vs):.1f}, V_std={np.std(vs):.2f}")
```

**What to look for:**
- If within-basin variance is tiny, counts determine V
- If within-basin variance is large, suit/trump structure matters
- The residual variance IS the tactical structure, isolated

### 6.3 Count Capture Probability Model

```python
# Simple model: V ≈ Σ count_value × capture_probability

def estimate_count_capture_prob(states, V, seed):
    """
    Fit: V ≈ Σ_i (value_i × P_i) + c
    where P_i = probability team 0 captures count i
    """
    # Build feature matrix
    # Each row: for each count domino, 1 if team 0 captures it (from basin analysis)
    
    X = []  # (n_states, n_counts)
    y = V
    
    for s in states:
        outcome = get_outcome(s)  # who captures each count
        features = [1 if outcome[c] == 0 else -1 for c in count_dominoes]
        X.append(features)
    
    X = np.array(X)
    
    # Fit linear model
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X, y)
    
    print(f"Coefficients: {model.coef_}")
    print(f"Intercept: {model.intercept_}")
    print(f"R²: {model.score(X, y):.4f}")
    
    # If R² ≈ 1, counts explain everything
    # Residuals are the suit/trump structure
    residuals = y - model.predict(X)
    return model, residuals
```

---

## 7. Critical Point Analysis

Where do decisions actually matter?

### 7.1 Decision Criticality

```python
def decision_criticality(state, Q):
    """
    How much does the decision at this state matter?
    """
    legal_q = [q for q in Q if q != -128]
    if len(legal_q) <= 1:
        return 0  # forced move
    
    best = max(legal_q)
    second_best = sorted(legal_q)[-2]
    
    return best - second_best  # gap

# Map criticality across state space
criticality = [decision_criticality(s, get_Q(s)) for s in states]

# Where are the critical decisions?
critical_states = [s for s, c in zip(states, criticality) if c > 10]

# What do they have in common?
analyze_features(critical_states)
```

### 7.2 Value Function Gradient

```python
def v_gradient(state, V, neighbors_fn):
    """
    Discrete gradient: how much does V change per move?
    """
    current_v = V[state]
    neighbor_vs = [V[n] for n in neighbors_fn(state)]
    
    if not neighbor_vs:
        return 0
    
    return max(abs(current_v - nv) for nv in neighbor_vs)

# High gradient = unstable region
# Low gradient = stable region
gradients = [v_gradient(s, V, get_children) for s in states]

plt.scatter(remaining_count(states), gradients, alpha=0.01)
plt.xlabel("Remaining Dominoes")
plt.ylabel("V Gradient")
```

**What to look for:**
- Gradient distribution: fat tails = volatile state space
- Gradient vs depth: does volatility increase or decrease?
- Spatial pattern: are high-gradient states clustered?

---

## 8. Spectral Analysis

Eigenvectors of the state transition structure.

### 8.1 Graph Laplacian Spectrum

```python
import scipy.sparse.linalg as spla

def build_transition_matrix(states):
    """Build (sparse) adjacency matrix of state transitions."""
    n = len(states)
    state_to_idx = {s: i for i, s in enumerate(states)}
    
    rows, cols, vals = [], [], []
    for s in states:
        i = state_to_idx[s]
        children = get_children(s)
        for c in children:
            if c in state_to_idx:
                j = state_to_idx[c]
                rows.append(i)
                cols.append(j)
                vals.append(1)
    
    A = scipy.sparse.coo_matrix((vals, (rows, cols)), shape=(n, n))
    return A

# Compute Laplacian
A = build_transition_matrix(states)
D = scipy.sparse.diags(np.array(A.sum(axis=1)).flatten())
L = D - A

# Get smallest eigenvalues (excluding 0)
eigenvalues, eigenvectors = spla.eigsh(L, k=20, which='SM')
```

**What to look for:**
- Spectral gap (λ₂): large gap = well-clustered state space
- Eigenvalue distribution: decay rate indicates structure
- Eigenvectors: natural coordinates for the state space

### 8.2 Diffusion Maps

```python
def diffusion_embedding(A, n_components=3, t=1):
    """
    Embed states using diffusion process on transition graph.
    """
    # Normalize to transition probabilities
    D_inv = scipy.sparse.diags(1.0 / np.array(A.sum(axis=1)).flatten())
    P = D_inv @ A
    
    # Power iteration for diffusion
    P_t = P ** t
    
    # Eigendecomposition
    eigenvalues, eigenvectors = spla.eigsh(P_t, k=n_components+1, which='LM')
    
    # Diffusion coordinates (skip first trivial eigenvector)
    embedding = eigenvectors[:, 1:] * eigenvalues[1:]
    
    return embedding

embedding = diffusion_embedding(A, n_components=3)

# Visualize with V as color
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=V, cmap='RdBu', alpha=0.1)
```

**What to look for:**
- Clean separation by V in diffusion coordinates = V is "natural" function
- Clusters = equivalence classes
- Smooth gradient = continuous structure

---

## 9. Algebraic Structure

Is there a group or ring hiding in here?

### 9.1 Suit Algebra

```python
# In Texas 42, suits matter only relatively (trump vs non-trump)
# This suggests a quotient structure

def suit_equivalence_class(state, trump):
    """
    Two states are suit-equivalent if they differ only by
    permutation of non-trump suits.
    """
    # Extract suit-independent features
    # e.g., "player 0 has 3 trump, 2 of suit A, 2 of suit B"
    # becomes "player 0 has 3 trump, and 2+2 off-suit"
    ...

# Count equivalence classes
classes = defaultdict(list)
for s in states:
    canonical = suit_equivalence_class(s, trump)
    classes[canonical].append(s)

print(f"States: {len(states)}, Classes: {len(classes)}")
print(f"Compression: {len(states)/len(classes):.2f}x")

# Check V consistency within classes
for canonical, members in classes.items():
    vs = [V[m] for m in members]
    if len(set(vs)) > 1:
        print(f"Class {canonical}: V varies! {set(vs)}")
```

### 9.2 Invariant Polynomials

```python
# What polynomial functions of the state are invariant under symmetry?
# And can V be expressed as a function of these invariants?

def compute_invariants(state, seed):
    """
    Compute values that don't change under symmetry operations.
    """
    hands = deal_from_seed(seed)
    
    invariants = {
        'total_remaining': sum(popcount(remaining[p]) for p in range(4)),
        'team_balance': (popcount(remaining[0]) + popcount(remaining[2])) - 
                        (popcount(remaining[1]) + popcount(remaining[3])),
        'count_control': count_control_metric(state, hands),
        # ... more invariants
    }
    return invariants

# Test if V is function of invariants
invariant_vectors = [tuple(compute_invariants(s, seed).values()) for s in states]
unique_invariants = set(invariant_vectors)

for inv in unique_invariants:
    matching_states = [s for s, iv in zip(states, invariant_vectors) if iv == inv]
    vs = [V[s] for s in matching_states]
    if len(set(vs)) == 1:
        print(f"Invariants {inv} -> V = {vs[0]} (deterministic!)")
```

---

## 10. Neural Network Probing

Use NNs as function approximators to discover structure.

### 10.1 Minimum Viable Network

```python
# What's the simplest network that achieves ~0 error?

architectures = [
    ("linear", [1]),
    ("shallow", [32, 1]),
    ("medium", [64, 64, 1]),
    ("deep", [128, 128, 128, 1]),
    ("wide_shallow", [1024, 1]),
]

for name, layers in architectures:
    model = build_mlp(input_dim=feature_dim, layers=layers)
    model.fit(X, V, epochs=100)
    mse = mean_squared_error(V, model.predict(X))
    mae = mean_absolute_error(V, model.predict(X))
    print(f"{name}: MSE={mse:.4f}, MAE={mae:.4f}")

# If linear achieves low error, V is (nearly) linear in features
# If shallow suffices, V is simple nonlinear function
# If deep required, V is complex or features are wrong
```

### 10.2 Learned Features

```python
# Train autoencoder, examine bottleneck

class Autoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, bottleneck_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
        self.value_head = nn.Linear(bottleneck_dim, 1)
    
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        v_pred = self.value_head(z)
        return x_recon, v_pred, z

# Train with reconstruction + value prediction loss
# Examine z: what dimensions correlate with V?
# What do the learned features represent?
```

### 10.3 Feature Importance via Ablation

```python
# Which input features can be removed without hurting V prediction?

full_model = train_model(X, V)
baseline_error = evaluate(full_model, X, V)

for i in range(X.shape[1]):
    X_ablated = X.copy()
    X_ablated[:, i] = 0  # or mean, or random
    ablated_error = evaluate(full_model, X_ablated, V)
    importance = ablated_error - baseline_error
    print(f"Feature {i}: importance = {importance:.4f}")
```

---

## 11. Cross-Seed Analysis

Structure that holds across different deals.

### 11.1 Universal Patterns

```python
# Do certain state configurations always have similar V across seeds?

def state_signature(state, seed):
    """
    Seed-independent description of state.
    """
    return (
        remaining_count(state),
        trick_length(state),
        count_dominoes_remaining(state, seed),
        # ... other seed-independent features
    )

# Collect V by signature across all seeds
signature_values = defaultdict(list)
for seed in seeds:
    for state in get_states(seed):
        sig = state_signature(state, seed)
        signature_values[sig].append(V[state])

# Analyze consistency
for sig, vs in signature_values.items():
    if len(vs) >= 10:
        print(f"Signature {sig}: V = {np.mean(vs):.1f} ± {np.std(vs):.2f}")
```

**What to look for:**
- Low variance signatures = universal structure
- High variance signatures = deal-specific tactics

### 11.2 Transfer Learning

```python
# Train on seeds 0-499, test on 500-999
train_seeds = range(500)
test_seeds = range(500, 1000)

# Model using only seed-independent features
X_train = extract_universal_features(train_seeds)
V_train = get_values(train_seeds)

model.fit(X_train, V_train)

X_test = extract_universal_features(test_seeds)
V_test = get_values(test_seeds)

print(f"Train R²: {model.score(X_train, V_train):.4f}")
print(f"Test R²: {model.score(X_test, V_test):.4f}")

# High test R² = universal structure exists
# Low test R² = V depends on deal-specific details
```

---

## 12. Experimental Execution Plan

### Phase 1: Baseline (Day 1)
1. Load 10 seeds, compute distribution profiles
2. Count unique V values per depth
3. Run LZMA compression test
4. Plot basic histograms

### Phase 2: Information Theory (Day 2)
1. Compute entropy of V
2. Test conditional entropy for obvious features
3. Find minimal sufficient statistics (greedy feature selection)

### Phase 3: Count Domino Analysis (Day 3)
1. Implement basin analysis
2. Partition by count capture outcome
3. Measure within-basin variance
4. If variance is low → counts explain everything
5. If variance is high → analyze residuals

### Phase 4: Symmetry (Day 4)
1. Implement canonical form under team/seat symmetry
2. Count orbits
3. Test V consistency within orbits

### Phase 5: Topology (Day 5)
1. Compute level set connectivity
2. Build Reeb graph for one seed
3. Run persistent homology if level sets are interesting

### Phase 6: Scaling/Fractal (Day 6)
1. Log-log plots of state count vs depth
2. DFA on principal variation time series
3. Value roughness at multiple scales

### Phase 7: Synthesis (Day 7)
1. What did we find?
2. If structure exists: what's the minimal representation?
3. If no structure: why not, and what does that tell us?

---

## Appendix: Feature Extraction Reference

```python
def extract_all_features(state, seed):
    """
    Comprehensive feature extraction for analysis.
    """
    remaining, leader, trick_len, p0, p1, p2 = unpack_state(state)
    hands = deal_from_seed(seed)
    
    return {
        # Basic
        'depth': sum(popcount(remaining[p]) for p in range(4)),
        'leader': leader,
        'trick_len': trick_len,
        'current_player': (leader + trick_len) % 4,
        'current_team': ((leader + trick_len) % 4) % 2,
        
        # Hand balance
        'hand_sizes': [popcount(remaining[p]) for p in range(4)],
        'team0_dominoes': popcount(remaining[0]) + popcount(remaining[2]),
        'team1_dominoes': popcount(remaining[1]) + popcount(remaining[3]),
        
        # Count dominoes
        'counts_remaining': count_dominoes_still_in_play(remaining, hands),
        'count_locations': who_holds_which_counts(remaining, hands),
        
        # Trick state
        'trick_plays': [p0, p1, p2][:trick_len],
        
        # Suit structure (requires declaration context)
        # 'trump_remaining': ...,
        # 'suit_voids': ...,
    }
```

---

## Summary

| Analysis Type | What It Reveals | Complexity |
|---------------|-----------------|------------|
| Entropy decomposition | Sufficient statistics | Low |
| Basin analysis | Count domino primacy | Medium |
| Symmetry orbits | True state space size | Medium |
| Level sets / Reeb graph | Value function topology | Medium |
| Persistent homology | Robust topological features | High |
| Spectral analysis | Natural coordinates | High |
| Fractal scaling | Self-similarity across depths | Medium |
| DFA / Hurst | Time series structure in PV | Low |

Start with information theory and basin analysis. They'll tell you whether the structure is obvious (counts explain everything) or subtle (residuals have hidden patterns). Everything else builds on that foundation.