# BEAD: Value MLP for Texas 42

**Status:** ready for implementation  
**Created:** 2025-12-28  
**Depends on:** DP solver (complete, running)  
**Enables:** Fast PIMC → Transformer training  

---

## The Pipeline Vision

```
┌─────────────────────────────────────────────────────────────────────┐
│                        COMPLETE PIPELINE                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │  DP Solver   │ →  │  Value MLP   │ →  │  Fast PIMC           │  │
│  │              │    │              │    │                      │  │
│  │  Perfect     │    │  Compressed  │    │  Uncertainty-aware   │  │
│  │  answers for │    │  evaluation  │    │  play using fast     │  │
│  │  fixed deals │    │  function    │    │  value estimates     │  │
│  │              │    │              │    │                      │  │
│  │  ✓ DONE      │    │  ← YOU ARE   │    │                      │  │
│  │  (running)   │    │     HERE     │    │                      │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│                                                   │                 │
│                                                   ↓                 │
│                                          ┌──────────────────────┐  │
│                                          │  Transformer         │  │
│                                          │                      │  │
│                                          │  Bidding, meta-game, │  │
│                                          │  opponent modeling,  │  │
│                                          │  style/personality   │  │
│                                          │                      │  │
│                                          │  (future)            │  │
│                                          └──────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Why this order:**

| Step | What It Gives You | Time Scale |
|------|-------------------|------------|
| DP Solver | Ground truth for perfect-info positions | Done (2-3 days) |
| Value MLP | Fast position evaluation (ms not minutes) | Days |
| Fast PIMC | Real-time play under uncertainty | Days |
| Transformer | Bidding, strategy, style, personality | Weeks |

The MLP compresses your 65GB of solved positions into ~2MB of learned function. That's the bridge from "I have the answers" to "I can use them fast enough."

---

## §1. What Is a Value MLP?

A Multi-Layer Perceptron (MLP) is the simplest neural network:

```
Input → Linear → Activation → Linear → Activation → ... → Output
```

No attention. No recurrence. No convolution. Just:
- Matrix multiply (Linear)
- Nonlinearity (ReLU, tanh)
- Repeat

**For our task:**

```
f(state) → value

Input:  Encoded game state (~150-200 numbers)
Output: Expected score differential (-42 to +42, normalized to -1 to +1)
```

The MLP learns to approximate the DP solver's perfect answers.

---

## §2. State Encoding

The MLP sees a flat vector. Every game state must encode to the same shape.

### 2.1 Minimal Encoding (Start Here)

```python
def encode_state(state, context):
    """
    Encode a game state as a flat vector.
    
    Inputs:
        state: packed game state from DP solver
        context: seed context (declaration, initial hands)
    
    Output:
        numpy array of shape [input_dim]
    """
    features = []
    
    # Remaining dominoes per player: 4 × 28 bits = 112 features
    for player in range(4):
        remaining = extract_remaining(state, player)
        for domino in range(28):
            features.append(1.0 if (remaining >> domino) & 1 else 0.0)
    
    # Score: 1 feature, normalized
    score = extract_score(state)
    features.append(score / 42.0)  # 0 to 1
    
    # Leader: 4 features (one-hot)
    leader = extract_leader(state)
    for p in range(4):
        features.append(1.0 if p == leader else 0.0)
    
    # Declaration: 10 features (one-hot)
    # 0-6: pip trump, 7: doubles trump, 8: doubles suit (nello), 9: no-trump
    decl = context.declaration
    for d in range(10):
        features.append(1.0 if d == decl else 0.0)
    
    # Trick plays (current incomplete trick): 4 × 28 bits = 112 features
    # Position i has domino if played, else zeros
    for position in range(4):
        domino = extract_trick_play(state, position)
        for d in range(28):
            if domino is not None and d == domino:
                features.append(1.0)
            else:
                features.append(0.0)
    
    # Tricks completed: 1 feature
    tricks_done = extract_tricks_done(state)
    features.append(tricks_done / 7.0)
    
    return np.array(features, dtype=np.float32)

# Total: 112 + 1 + 4 + 10 + 112 + 1 = 240 features
```

### 2.2 Derived Features (Add If Needed)

If learning stalls, add features the MLP would otherwise have to learn:

```python
def encode_derived_features(state, context):
    """Additional features derived from raw state."""
    features = []
    
    # Count dominoes by location: 5 count dominoes × 5 locations = 25 features
    # Locations: player 0, 1, 2, 3, or played
    for count_domino in COUNT_DOMINOES:  # [5-0, 4-1, 3-2, 5-5, 6-4]
        location = find_domino_location(state, count_domino)
        for loc in range(5):
            features.append(1.0 if loc == location else 0.0)
    
    # Voids: 4 players × 7 suits = 28 features
    for player in range(4):
        for suit in range(7):
            is_void = player_void_in_suit(state, player, suit)
            features.append(1.0 if is_void else 0.0)
    
    # Trump count per player: 4 features
    for player in range(4):
        trump_count = count_trumps(state, player, context.declaration)
        features.append(trump_count / 7.0)
    
    # Team control proxy: who has more high cards? 2 features
    team0_strength = hand_strength(state, 0) + hand_strength(state, 2)
    team1_strength = hand_strength(state, 1) + hand_strength(state, 3)
    features.append(team0_strength / 100.0)  # normalize appropriately
    features.append(team1_strength / 100.0)
    
    return np.array(features, dtype=np.float32)
```

**Philosophy:** Start minimal. The MLP might surprise you by learning the derived features implicitly. Only add them if you hit a wall.

---

## §3. Architecture

### 3.1 Starting Architecture

```python
import torch
import torch.nn as nn

class ValueMLP(nn.Module):
    def __init__(self, input_dim=240, hidden_dims=[256, 128, 64]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))  # helps training stability
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Tanh())  # output in [-1, 1]
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Parameter count:
# 240×256 + 256×128 + 128×64 + 64×1 = 61,440 + 32,768 + 8,192 + 64 = ~102K parameters
# Model size: ~400KB
```

### 3.2 Output Options

**Option A: Single value (regression)**
```python
output = nn.Tanh()  # continuous in [-1, 1]
loss = nn.MSELoss()
```

**Option B: Classification (85 classes for -42 to +42)**
```python
output = nn.Linear(64, 85)  # logits for each possible score
loss = nn.CrossEntropyLoss()
# At inference: take argmax or expected value
```

Option A is simpler. Option B can capture uncertainty. Start with A.

---

## §4. Training Recipe

### 4.1 Data Loading

```python
def load_training_data(solved_dir, max_seeds=100):
    """
    Load solved seeds and extract (state, value) pairs.
    
    Each seed file contains:
        - context: declaration, initial hands
        - states: dict mapping state_hash → (value, best_move)
    """
    X, y = [], []
    
    for seed_file in glob(f"{solved_dir}/*.bin")[:max_seeds]:
        context, solutions = load_seed(seed_file)
        
        for state_hash, (value, best_move) in solutions.items():
            state = unpack_state(state_hash)
            features = encode_state(state, context)
            
            X.append(features)
            y.append(value / 42.0)  # normalize to [-1, 1]
    
    return np.array(X), np.array(y)
```

### 4.2 Training Loop

```python
def train_value_mlp(X, y, epochs=50, batch_size=1024, val_split=0.1):
    # Split data
    split_idx = int(len(X) * (1 - val_split))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
        batch_size=batch_size
    )
    
    # Model, optimizer, loss
    model = ValueMLP(input_dim=X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    loss_fn = nn.MSELoss()
    
    # Training loop
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch).squeeze()
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                pred = model(X_batch).squeeze()
                val_loss += loss_fn(pred, y_batch).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}: train={train_loss:.4f}, val={val_loss:.4f}")
    
    return model
```

### 4.3 Hyperparameters (Starting Point)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Hidden layers | [256, 128, 64] | Decrease if overfitting |
| Learning rate | 1e-3 | Adam default, reduce on plateau |
| Batch size | 1024 | Larger is fine with big data |
| Epochs | 50-100 | Watch val loss, stop early if plateau |
| Dropout | 0.0 | Add 0.1-0.2 if overfitting |
| Weight decay | 0.0 | Add 1e-5 if overfitting |

---

## §5. Confidence Ladder

**This is how you gain confidence systematically.**

### Step 1: One Seed Sanity Check

**Goal:** Verify encoding works, MLP can learn.

```python
# Load ONE seed
X, y = load_training_data("solved/", max_seeds=1)

# Train on 80%, test on 20%
model = train_value_mlp(X, y, val_split=0.2)

# Expected result:
# - Train loss: < 0.01
# - Val loss: < 0.02 (slightly higher is fine)
```

**If val loss is HIGH (> 0.1):**
- Encoding bug (check feature dimensions)
- Label bug (check value normalization)
- Architecture too small (unlikely for one seed)

**If val loss is LOW:** Proceed to Step 2.

### Step 2: Cross-Seed Generalization

**Goal:** MLP generalizes to unseen deals.

```python
# Load 100 seeds
X, y = load_training_data("solved/", max_seeds=100)

# Shuffle by SEED, not by state
# Ensure test set contains entire seeds never seen in training
X_train, X_test, y_train, y_test = split_by_seed(X, y, test_seeds=10)

model = train_value_mlp(X_train, y_train)
test_loss = evaluate(model, X_test, y_test)

# Expected:
# - Val loss (same seeds): < 0.01
# - Test loss (new seeds): < 0.02
```

**If test loss is MUCH HIGHER than val loss:**
- Overfitting to specific deals
- Need more seeds
- Add regularization (dropout, weight decay)

**If test loss is LOW:** Proceed to Step 3.

### Step 3: Spot-Check Predictions

**Goal:** Eyeball actual predictions.

```python
def spot_check(model, test_states, dp_values, n=20):
    """Print side-by-side comparisons."""
    indices = random.sample(range(len(test_states)), n)
    
    print("DP Value | MLP Value | Error | State Summary")
    print("-" * 60)
    
    total_error = 0
    for i in indices:
        dp_val = dp_values[i] * 42  # denormalize
        mlp_val = model(test_states[i:i+1]).item() * 42
        error = abs(dp_val - mlp_val)
        total_error += error
        
        print(f"{dp_val:+6.1f}   | {mlp_val:+6.1f}    | {error:4.1f}  | {summarize(i)}")
    
    print(f"\nMean absolute error: {total_error/n:.2f} points")
```

**What to look for:**
- Errors < 2 points: Excellent
- Errors 2-5 points: Good enough for PIMC
- Errors > 5 points: Investigate systematic patterns
- All errors same sign: Bias in encoding or labels

### Step 4: Integration Test with PIMC

**Goal:** MLP is accurate enough to guide search.

```python
def pimc_comparison_test(model, test_positions, n_samples=50):
    """
    Compare PIMC results using MLP vs DP lookup.
    """
    agreement = 0
    
    for position in test_positions:
        # PIMC with DP (ground truth)
        best_move_dp = pimc_with_dp(position, n_samples)
        
        # PIMC with MLP
        best_move_mlp = pimc_with_mlp(position, model, n_samples)
        
        if best_move_dp == best_move_mlp:
            agreement += 1
    
    print(f"Move agreement: {agreement}/{len(test_positions)} ({100*agreement/len(test_positions):.1f}%)")
```

**Target:** 90%+ agreement. The MLP can be wrong on exact values as long as it preserves move ordering.

### Step 5: Scale Up

**Goal:** Train on full dataset, maximize generalization.

```python
# All 300 seeds
X, y = load_training_data("solved/", max_seeds=300)

# Train with proper cross-validation
model = train_with_cv(X, y, n_folds=5)

# Final test on held-out seeds
final_test_loss = evaluate(model, held_out_seeds)
```

---

## §6. Success Criteria

| Metric | Target | Meaning |
|--------|--------|---------|
| Val MSE (same seeds) | < 0.005 | Can fit the data |
| Test MSE (new seeds) | < 0.01 | Generalizes to new deals |
| Mean abs error | < 2 points | Practical accuracy |
| PIMC move agreement | > 90% | Useful for search |
| Inference time | < 1ms | Fast enough for real-time |
| Model size | < 2MB | Fits in PWA |

---

## §7. What Can Go Wrong

### 7.1 Overfitting

**Symptom:** Train loss low, val/test loss high.

**Fix:**
- More training seeds (you have 300, use them)
- Dropout (add 0.1-0.2 between layers)
- Weight decay (1e-5 to 1e-4)
- Smaller model (fewer hidden units)

### 7.2 Underfitting

**Symptom:** Both train and val loss stuck high.

**Fix:**
- Larger model (more hidden units, more layers)
- Train longer
- Check encoding (are features normalized?)
- Add derived features

### 7.3 Systematic Errors

**Symptom:** Errors correlate with game phase, trump type, etc.

**Fix:**
- Examine which states have high error
- Add features specific to that pattern
- Consider separate models per declaration (probably overkill)

### 7.4 Poor PIMC Integration

**Symptom:** MLP has low MSE but PIMC picks wrong moves.

**Fix:**
- MSE might not capture move ordering
- Switch to ranking loss or classification
- Increase PIMC sample count to average out MLP noise

### 7.5 Cross-Seed Generalization Gap (Step 2 Finding)

**Symptom:** Val loss (same seeds) ≈ 0.022, but test loss (held-out seeds) ≈ 0.040.
The model learns seed-specific patterns, not generalizable game understanding.

**Root Cause:** Raw domino IDs are the wrong encoding basis.

| Encoding | Seed 0 | Seed 1 | Problem |
|----------|--------|--------|---------|
| "Player 0 has domino 14" | 4-3 in hand | NOT in hand | Meaning changes per seed |
| "Player 0 has 3rd-highest trump" | Same meaning | Same meaning | Seed-invariant! |

The model learned "when player 0 has domino 14 and player 1 has domino 22, value is +8" — but that's **seed-specific memorization**, not game understanding.

**The Fix: τ-Based Encoding**

Use the `trick_rank()` function (τ) to encode dominoes by power rank, not ID:

```python
# The τ function already exists in tables.py
trick_rank(domino_id, led_suit, decl_id) → 6-bit key (tier << 4 | rank)

# For remaining hands: encode POTENTIAL power
# Trumps: rank within trump tier (0=boss, 6=lowest)
# Off-suit: rank within own suit (how strong if my suit is led)

# For current trick: encode ACTUAL power
# Use actual led suit to determine who's winning
```

**Stratified Encoding Design:**

```python
def encode_hand_tau(hand_mask, decl):
    """
    Encode hand by τ-rank, not domino ID.

    Returns per player:
    - 7 bits: which trump rank slots held (0=boss through 6=lowest)
    - Per off-suit: which rank slots held (or just presence/void)
    """
    features = []

    # Trump holdings by rank
    trump_ranks = [0.0] * 7
    for domino in range(28):
        if (hand_mask >> domino) & 1 and is_trump(domino, decl):
            rank = trick_rank(domino, trump_suit, decl) & 0xF
            trump_ranks[rank] = 1.0
    features.extend(trump_ranks)

    # Off-suit holdings (similar pattern)
    # ...

    return features
```

**Key Insight:** "Has boss trump" transfers across seeds. "Has domino 14" doesn't.

**Validation Approach:**

Before implementing full τ-encoding, run a diagnostic:
1. Take worst predictions (error > 15 points)
2. Find nearest neighbors by τ-encoding vs raw encoding
3. If τ-nearest has similar true value → hypothesis confirmed
4. If τ-nearest also wrong → something else is missing

See beads t42-wzsq (diagnostic) and t42-74vy (implementation).

---

## §8. Export for Inference

```python
# Save PyTorch model
torch.save(model.state_dict(), "value_mlp.pt")

# Export to ONNX for browser/mobile
dummy_input = torch.randn(1, input_dim)
torch.onnx.export(
    model, dummy_input, "value_mlp.onnx",
    input_names=['state'],
    output_names=['value'],
    dynamic_axes={'state': {0: 'batch'}}
)

# Verify ONNX
import onnxruntime as ort
session = ort.InferenceSession("value_mlp.onnx")
result = session.run(None, {'state': dummy_input.numpy()})
```

**Browser inference:** Use ONNX Runtime Web or TensorFlow.js

---

## §9. Next Step After This

Once MLP passes the confidence ladder:

1. **Integrate with PIMC** - Replace DP lookup with MLP inference
2. **Benchmark speed** - Measure games/second with MLP-powered PIMC
3. **Generate transformer training data** - Run millions of PIMC games
4. **Begin transformer work** - Tokenize game histories, train on PIMC decisions

---

## Summary

```
You have:      65GB of perfect answers (DP solver output)
You're building: 2MB function that approximates them (Value MLP)
You gain:      Fast evaluation for PIMC (~1000× speedup)
You verify:    Confidence ladder (one seed → cross-seed → spot-check → PIMC test)
```

The MLP is the compression step. It doesn't need to be perfect. It needs to be good enough that PIMC still makes correct decisions when using MLP estimates instead of DP lookups.