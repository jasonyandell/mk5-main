# E[Q] Training Pipeline

Training data generation for Stage 2: an imperfect-information policy that predicts E[Q] (expected quality) from observable game state only.

## The Problem

Texas 42 AIs using hand-coded heuristics are **point estimates applied to a distribution**. Rules like "don't pull partner's trump" assume one world state when reality is a probability distribution across many possible opponent hands. Heuristics don't compose, conflict with each other, and fail at edge cases.

## The Solution: Two-Stage Architecture

```
┌─────────────────────────────────────────────────────────┐
│  STAGE 1: Perfect-Info Oracle (EXISTS)                 │
│                                                         │
│  Input:  All 4 hands + game state                      │
│  Output: Q-values for 7 actions                        │
│  Model:  domino-large-817k-valuehead-*.ckpt            │
└─────────────────────────────────────────────────────────┘
                          │
                          │ Query N=100 sampled worlds
                          ▼
┌─────────────────────────────────────────────────────────┐
│  E[Q] MARGINALIZATION (THIS PIPELINE)                  │
│                                                         │
│  For each decision:                                     │
│    1. Infer voids from play history                    │
│    2. Sample N consistent opponent hands               │
│    3. Query Stage 1 for Q-values on each world         │
│    4. E[Q] = mean(Q across worlds)                     │
│    5. Record (transcript, E[Q]) as training example    │
│                                                         │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 2: Imperfect-Info Policy (TO BUILD)             │
│                                                         │
│  Input:  Transcript only (public information)          │
│  Output: E[Q] for 7 actions                            │
│  Role:   Deployed player (single forward pass)         │
└─────────────────────────────────────────────────────────┘
```

## Documentation

- Research spec (semantics, invariants, ML tradeoffs): `docs/EQ_STAGE2_TRAINING.md`
- Ops manual (how to run/debug generation): `forge/eq/README.md`

## Quick Start

```bash
# Generate training data (1000 games, ~7 minutes on RTX 3050 Ti)
python -m forge.eq.generate_dataset --n-games 1000 --output forge/data/eq_dataset.pt

# Continuous generation with monitoring (resumes from existing dataset)
python -m forge.eq.generate_continuous --target-games 100000

# View training examples interactively
python -m forge.eq.viewer forge/data/eq_dataset.pt

# Jump to specific game
python -m forge.eq.viewer forge/data/eq_dataset.pt --game 42
```

## File Structure

```
forge/eq/
├── __init__.py              # Public API: Stage1Oracle, generate_eq_game, tokenize_transcript
├── voids.py                 # infer_voids() - detect void suits from play history
├── sampling.py              # sample_consistent_worlds() - backtracking hand sampler
├── oracle.py                # Stage1Oracle - wraps trained model for batch queries
├── game.py                  # GameState - immutable game state tracker
├── transcript_tokenize.py   # Tokenizer for Stage 2 (public info only)
├── generate.py              # generate_eq_game() - play one game, record decisions
├── generate_dataset.py      # CLI for batch generation
├── generate_continuous.py   # Continuous generation with monitoring (Ctrl+C safe)
├── viewer.py                # Interactive training data inspector
└── test_*.py                # Unit tests for each module
```

## Pipeline Components

### 1. Void Inference (`voids.py`)

Detects which suits each player is void in by observing failed follows:

```python
from forge.eq import infer_voids

# plays: list of (player, domino_played, lead_domino)
voids = infer_voids(plays, decl_id)
# Returns: {0: set(), 1: {7}, 2: set(), 3: {5, 6}}
# Player 1 is void in called suit, Player 3 void in fives and sixes
```

### 2. World Sampling (`sampling.py`)

Generates opponent hand configurations consistent with observed voids:

```python
from forge.eq.sampling import sample_consistent_worlds

worlds = sample_consistent_worlds(
    my_player=0,
    my_hand=[0, 1, 2, 3, 4, 5, 6],
    played=set(),
    hand_sizes=[7, 7, 7, 7],
    voids={0: set(), 1: set(), 2: set(), 3: set()},
    decl_id=5,
    n_samples=100,
)
# Returns: list of 100 worlds, each is [hand0, hand1, hand2, hand3]
```

Uses **backtracking with MRV heuristic** - guaranteed to find a valid solution if one exists (and one always does - the real game state is proof).

### 3. Stage 1 Oracle (`oracle.py`)

Wraps the trained perfect-info model for efficient batch queries:

```python
from forge.eq import Stage1Oracle

oracle = Stage1Oracle("forge/models/domino-qval-large-*.ckpt", device="cuda")

# Query 100 sampled worlds in a single forward pass
q_values = oracle.query_batch(worlds, game_state_info, current_player)
# Shape: (100, 7) - Q-values (in POINTS) for each action in each world
# IMPORTANT: These are point estimates, NOT logits - do NOT apply softmax!
```

### 4. Game State (`game.py`)

Immutable game state with functional transitions:

```python
from forge.eq.game import GameState

state = GameState.from_hands(hands, decl_id, leader=0)
player = state.current_player()
legal = state.legal_actions()
state = state.apply_action(domino_id)  # Returns new state
```

### 5. Transcript Tokenizer (`transcript_tokenize.py`)

Encodes public information for Stage 2:

```python
from forge.eq import tokenize_transcript

tokens = tokenize_transcript(
    my_hand=[5, 7, 12],           # Current player's remaining dominoes
    plays=[(0, 20), (1, 15)],     # (absolute_player, domino_id) pairs
    decl_id=5,                    # Trump declaration
    current_player=1              # Perspective for relative player IDs
)
# Shape: (6, 8) = 1 decl + 3 hand + 2 plays
```

**Key difference from Stage 1**: Stage 2 sees only public information (my hand + transcript), not opponent hands.

### 6. Game Generation (`generate.py`)

Plays complete games using E[Q] policy, recording all decisions:

```python
from forge.eq import Stage1Oracle, generate_eq_game

oracle = Stage1Oracle(checkpoint_path)
record = generate_eq_game(oracle, hands, decl_id, n_samples=100)
# Returns GameRecord with 28 DecisionRecords (4 players x 7 tricks)
```

Each `DecisionRecord` contains:
- `transcript_tokens`: Stage 2 input
- `e_q_mean`: Target E[Q] values in POINTS (7,) - **NOT logits, do NOT softmax**
- `legal_mask`: Which actions were legal (7,)
- `action_taken`: Index of action played

For `DecisionRecordV2` (with posterior weighting + exploration):
- All above fields plus:
- `e_q_var`: Variance of Q across worlds in points² (7,)
- `u_mean`: State-level uncertainty = mean(σ) over legal actions
- `u_max`: State-level uncertainty = max(σ) over legal actions
- `diagnostics`: Posterior health metrics (ESS, max weight, etc.)

### 7. Continuous Generation (`generate_continuous.py`)

Runs indefinitely generating training data with monitoring:

```bash
# Run until 100K games (resumes automatically if dataset exists)
python -m forge.eq.generate_continuous --target-games 100000

# Custom batch size (saves after each batch)
python -m forge.eq.generate_continuous --batch-size 50 --target-games 100000

# Run forever (Ctrl+C to stop)
python -m forge.eq.generate_continuous --target-games 0
```

Features:
- **Resumable**: Detects existing dataset and continues from where it left off
- **Atomic saves**: Uses temp file + rename to prevent corruption
- **Graceful shutdown**: Ctrl+C saves current state before exiting
- **Progress monitoring**: Shows games/sec, file size, ETA to target
- **Deterministic seeds**: Same dataset reproduced if starting fresh

Example output:
```
Batch   42 | Games:    5100/100000 |   2.1 g/s |  326.4 MB | ETA: 12h 32m
```

### 8. Debug Viewer (`viewer.py`)

Interactive console viewer for inspecting training data:

```
Example 14/28000  |  Game 42  |  Decision 14/28
────────────────────────────────────────────────────────────
Declaration: Sixes (id=6)

My Hand: [6:4] >[6:2]< [5:3] (4:1) (3:0)  (5 remaining)

Trick History:
  T1: ME [6:6] → L [6:5] → P [6:3] → R [5:5]

Current Trick: ME [5:2] → L [5:1] → P [5:0] → ME: ?

E[Q] (points):
    [6:4]  +3.2 ████████████████████
    [6:2]  +1.1 ███████████████ ← SELECTED
    [5:3]  -0.4 ███████
    [4:1]  -2.0 ████
    [3:0]  -3.5 ██

[←] Prev  [→] Next  [j] Jump  [q] Quit
```

## Dataset Format

Output from `generate_dataset.py`:

```python
{
    "transcript_tokens": Tensor,  # (N, 36, 8) - padded transcripts
    "transcript_lengths": Tensor, # (N,) - actual lengths
    "e_q_mean": Tensor,           # (N, 7) - E[Q] in POINTS (NOT logits!)
    "e_q_var": Tensor,            # (N, 7) - Var[Q] in points²
    "legal_mask": Tensor,         # (N, 7) - boolean mask
    "action_taken": Tensor,       # (N,) - action indices
    "game_idx": Tensor,           # (N,) - which game (0-999)
    "decision_idx": Tensor,       # (N,) - position in game (0-27)
    "train_mask": Tensor,         # (N,) - True for train, False for val
    "u_mean": Tensor,             # (N,) - state-level uncertainty (mean σ)
    "u_max": Tensor,              # (N,) - state-level uncertainty (max σ)
    "ess": Tensor,                # (N,) - effective sample size
    "metadata": dict,             # Generation info with schema semantics
}
```

**IMPORTANT**: `e_q_mean` contains Q-values in **POINTS** (roughly [-42, +42]), NOT logits.
Do NOT apply softmax to these values - they are already interpretable point estimates.

---

## What Worked and What Didn't

### World Sampling: Rejection Sampling vs Backtracking

**What Didn't Work: Rejection Sampling**

The initial design used rejection sampling:

```python
# NAIVE APPROACH - DON'T DO THIS
while len(worlds) < n_samples:
    random.shuffle(remaining_dominoes)
    hands = distribute_to_players(remaining_dominoes)
    if all_void_constraints_satisfied(hands, voids):
        worlds.append(hands)
```

Problems:
1. **Exponential rejection rate**: As voids accumulate, valid distributions become rare
2. **No termination guarantee**: With tight constraints, could loop forever
3. **Invisible failures**: Hard to debug when sampling stalls

The original TypeScript implementation (`src/game/ai/hand-sampler.ts`) had the same problem and solved it with backtracking.

**What Worked: Backtracking with MRV Heuristic**

```python
def _backtrack(opponents, remaining, available, candidates_per_opponent, hands, rng):
    # Find player with MINIMUM slack (fewest excess candidates)
    most_constrained = min(
        [p for p in opponents if remaining[p] > 0],
        key=lambda p: len(candidates_per_opponent[p] & available) - remaining[p]
    )

    # Try each candidate, backtrack on failure
    for candidate in shuffled(candidates_per_opponent[most_constrained] & available):
        assign(candidate, most_constrained)
        if backtrack(...):  # Recurse
            return True
        unassign(candidate, most_constrained)  # Backtrack

    return False
```

Benefits:
1. **Guaranteed termination**: Always finds a solution if one exists
2. **Efficient pruning**: MRV heuristic fails fast on impossible branches
3. **Debuggable**: If it fails, there's a bug in void inference (not bad luck)

The real game state is always a valid solution, so backtracking should never fail. If it does, it indicates a bug in constraint tracking.

### Oracle Tokenization: Duplicating vs Reusing Stage 1

**What Didn't Work: Duplicating Tokenization**

Initial attempts tried to write a new tokenizer from scratch:

```python
# WRONG - diverged from Stage 1 format, caused silent bugs
def tokenize_for_oracle(world, game_state):
    tokens = my_custom_format(world)  # Different from training format!
    return tokens
```

Problems:
1. **Silent distribution shift**: Model trained on format A, queried with format B
2. **Maintenance burden**: Two tokenizers to keep in sync
3. **Subtle bugs**: Token ordering, feature scaling differences

**What Worked: Reusing Stage 1 Tokenization**

```python
# oracle.py imports directly from Stage 1 tokenizer
from forge.ml.tokenize import (
    DOMINO_HIGH, DOMINO_LOW, DOMINO_IS_DOUBLE,
    TOKEN_TYPE_PLAYER0, TOKEN_TYPE_TRICK_P0,
    # ... same constants used during training
)
```

The `Stage1Oracle._tokenize_worlds()` method replicates the exact token format from `forge/ml/tokenize.py`, ensuring the model sees the same distribution it was trained on.

### Vectorization: Python Loops vs NumPy Broadcasting

**What Didn't Work: Triple-Nested Python Loops**

```python
# SLOW - 100 worlds × 4 players × 7 dominoes = 2800 iterations
for world_idx, world in enumerate(worlds):
    for player_idx, hand in enumerate(world):
        for local_idx, domino_id in enumerate(hand):
            tokens[world_idx, ...] = compute_features(domino_id)
```

At N=100 samples × 28 decisions × 1000 games, this adds up.

**What Worked: Vectorized NumPy Operations**

```python
# FAST - single array operations
worlds_array = np.array(worlds)  # (N, 4, 7)
flat_ids = worlds_array.reshape(N, 28)
all_features = domino_features[flat_ids]  # (N, 28, 5) - one lookup!
tokens[:, 1:29, 0:5] = all_features
```

The optimized `_tokenize_worlds()` in `oracle.py` uses:
- Precomputed domino feature lookup table
- NumPy broadcasting for player normalization
- Vectorized remaining bitmask computation
- Only falls back to loops for variable-length trick tokens

### Loading Checkpoints: Lightning vs Direct State Dict

**What Didn't Work: Standard Lightning Loading**

```python
# CRASHED - RNG state compatibility issues between PyTorch versions
model = DominoLightningModule.load_from_checkpoint(path)
```

PyTorch Lightning's `load_from_checkpoint` tries to restore optimizer state, RNG seeds, and other training state. This caused cryptic errors when the checkpoint was saved with a different PyTorch version.

**What Worked: Loading Weights Only**

```python
# WORKS - skip training state, load only what we need for inference
checkpoint = torch.load(path, map_location=device, weights_only=False)
hparams = checkpoint['hyper_parameters']
model = DominoLightningModule(**hparams)
model.load_state_dict(checkpoint['state_dict'])  # Weights only!
model.eval()
```

For inference, we only need the model weights. Skipping optimizer state, RNG restoration, and other training artifacts avoids compatibility issues.

**Note**: The oracle also handles `torch.compile()` checkpoints by stripping the `_orig_mod.` prefix from state dict keys.

### Q-Value Models vs Logit-Based Models for E[Q]

**What Didn't Work: Using Logit Outputs as Point Estimates**

Logit-based models (trained with soft cross-entropy) output arbitrary-scale values:

```python
# Policy model output for a decision
policy_logits = [-0.72, -1.09, -1.87, -1.96, -2.12, -2.22, -1.21]
# Range: [-2.22, -0.72] - arbitrary scale, NOT points!
```

These logits require softmax to become probabilities, which doesn't give point estimates. While `argmax(logits) = argmax(Q)` (ordering is preserved), the magnitudes aren't interpretable.

**What Works: Q-Value Models for Interpretable E[Q]**

Q-value models (trained with `loss_mode='qvalue'`) output expected points directly:

```python
# Q-value model output for the same decision
e_q_values = [-17.89, -21.05, -21.45, -21.29, -21.02, -20.79, -18.18]
# Range: [-21.45, -17.89] - directly interpretable as expected points!
```

Benefits:
- **Interpretable**: E[Q] = -17.89 means "expect to be 17.89 points behind"
- **Meaningful deltas**: The 3.5-point spread between best and worst reflects actual strategic value
- **Debuggable**: Can verify values are in valid game range (roughly ±42 points)

**Recommendation**: Use Q-value model checkpoints (e.g., `domino-qval-*.ckpt`) for E[Q] marginalization when interpretability matters.

---

## Performance

On RTX 3050 Ti (4GB VRAM):

| Metric | Value |
|--------|-------|
| Games/second | ~2.5 |
| Time for 1000 games | ~7 minutes |
| Examples generated | 28,000 (28 per game) |
| GPU memory | ~2GB |

Bottleneck is oracle queries (100 forward passes per decision). Could parallelize across GPUs with Modal for larger datasets.

## Testing

```bash
# All unit tests (< 30 seconds)
python -m pytest forge/eq/ -v --timeout=60

# Individual modules
python -m pytest forge/eq/test_voids.py -v
python -m pytest forge/eq/test_sampling.py -v
python -m pytest forge/eq/test_oracle.py -v
python -m pytest forge/eq/test_game.py -v
python -m pytest forge/eq/test_generate.py -v
```

## Validation & Debugging

### Validation Scripts

| Script | Purpose |
|--------|---------|
| `scripts/validate_eq_computation.py` | Verify E[Q] matches fresh oracle queries |
| `scripts/analyze_eq_predictions.py` | Compare predictions to actual game outcomes |

Run validation on any dataset:

```bash
# Verify E[Q] computation is correct
python scripts/validate_eq_computation.py forge/data/eq_v2_*.pt

# Analyze prediction quality vs actual outcomes
python scripts/analyze_eq_predictions.py forge/data/eq_v2_*.pt
```

### Key Findings

**E[Q] computation is correct:**
- Fresh oracle queries match dataset E[Q] within sampling variance (~1-3 pts)
- Average E[Q] across a game correlates strongly with outcomes (r ≈ +0.76)

**Opening lead uncertainty is genuine:**
- Opening E[Q] has weak correlation with outcomes (r ≈ +0.16)
- High variance (σ ≈ 20 pts) reflects true uncertainty about opponents' hands
- RMSE ≈ 24 pts is expected given the variance

**Oracle behavior at opening:**
- Stage 1 oracle gives flat Q-values (~0.5 pt spread) for ~65% of random opponent configurations
- Only ~15% of configurations produce strong discrimination (>10 pt spread)
- Oracle rates some high trump leads poorly at opening (may be conserving trump strategy)

### Viewer Debug Mode

Press `d` in the interactive viewer to see debug details including:
- Per-world Q-values from oracle
- Weight distribution (if posterior weighting enabled)
- ESS and max weight diagnostics

```bash
python -m forge.eq.viewer forge/data/eq_v2_*.pt
# Then press 'd' for debug mode
```

---

## Next Steps

1. **Train Stage 2**: Use generated data to train imperfect-info policy
2. **Slam Dunk Test**: Validate 6-6 vs 2-2 strategy fusion scenario (see t42-xtu1)
3. **Scale up**: Generate 100K games on Modal GPUs
