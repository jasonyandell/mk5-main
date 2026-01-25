# 20: Proposed Fix - Shuffle Hand Ordering During Training

## Summary

**Root Cause Confirmed**: The `deal_from_seed()` function in `forge/oracle/rng.py` sorts hands by domino ID, creating a systematic training distribution bias where slot 0 only sees low-ID dominoes (0-0, 1-0, 1-1, etc.) and never sees high-value dominoes (6-4, 6-5, 6-6).

**Proposed Fix**: Shuffle the within-hand ordering during tokenization with a deterministic per-sample RNG.

**Impact**: Requires re-tokenization and retraining. Existing oracle shards remain valid (state/Q-values unchanged).

## The Problem

From `forge/oracle/rng.py:13`:
```python
hands = [sorted(dominos[i * 7 : (i + 1) * 7]) for i in range(4)]
```

This creates:
- **Slot 0**: Always minimum-ID domino (1.74-bit KL divergence from uniform)
- **Slot 6**: Always maximum-ID domino
- **Training asymmetry**: Model learns different Q-value patterns per slot position

The model sees 500K+ examples of 0-0 in slot 0, but zero examples of 6-6 in slot 0.

## Where to Apply the Fix

### Option 1: During Tokenization (RECOMMENDED)

Shuffle within-hand domino ordering in `forge/ml/tokenize.py::process_shard()` after calling `deal_from_seed()`.

**Location**: Lines 190-192 of `forge/ml/tokenize.py`:
```python
# Current code:
hands = deal_from_seed(seed)
hands_flat = np.array([hands[p][i] for p in range(4) for i in range(7)])
```

**Proposed change**:
```python
# Deal and shuffle within-hand ordering
hands = deal_from_seed(seed)

# Deterministic shuffle using per-shard RNG (already exists)
for p in range(4):
    # Shuffle each player's hand independently
    shard_rng.shuffle(hands[p])

hands_flat = np.array([hands[p][i] for p in range(4) for i in range(7)])
```

**Why tokenization is the best place**:
1. **Deterministic**: Uses existing `shard_rng` keyed by `(global_seed, seed, decl_id)`
2. **One-time cost**: Shuffle once during tokenization, not per-epoch
3. **No runtime overhead**: Training reads pre-shuffled data
4. **Preserves oracle data**: Parquet files with Q-values remain unchanged

### Option 2: During Training (Data Augmentation)

Shuffle in `DominoDataset.__getitem__()` or via a custom collate function.

**Advantages**:
- Works with existing tokenized data
- Can generate different permutations each epoch

**Disadvantages**:
- Runtime overhead on every batch
- Must also permute targets, legal masks, and Q-values consistently
- More complex implementation

### Option 3: At Oracle Data Generation

Modify `deal_from_seed()` to return unsorted hands.

**Disadvantages**:
- Breaks backward compatibility with existing shards
- Complicates state comparison/deduplication
- Too invasive for the benefit

### Option 4: Architecture Change (Set Transformer)

Use permutation-invariant architecture (DeepSets, Set Transformer).

**Disadvantages**:
- Major architecture change
- More complex, less interpretable
- Overkill for this specific problem

## Recommended Implementation

### Step 1: Modify `process_shard()` in `forge/ml/tokenize.py`

```python
def process_shard(
    path: Path,
    global_seed: int,
    max_samples: int | None = None,
    shuffle_hand_order: bool = True,  # NEW PARAMETER
) -> ShardResult | None:
    """Process one parquet file with per-shard deterministic RNG.

    Args:
        ...
        shuffle_hand_order: If True, randomly permute within-hand domino order
                           to prevent position-based learning artifacts.
    """
    # ... existing code ...

    # Deal hands
    hands = deal_from_seed(seed)

    # NEW: Shuffle within-hand ordering
    if shuffle_hand_order:
        # Each hand gets an independent permutation
        # Uses shard_rng for determinism
        for p in range(4):
            shard_rng.shuffle(hands[p])

    hands_flat = np.array([hands[p][i] for p in range(4) for i in range(7)])
    # ... rest unchanged ...
```

### Step 2: Update Target/Q-value Mapping

**Critical**: The targets (oracle best action) and Q-values are stored as local indices (0-6). When we shuffle the hand ordering, we must map these indices through the same permutation.

```python
# Store the permutation for target/Q remapping
if shuffle_hand_order:
    # For current player's hand, need to know old->new index mapping
    # This requires tracking the permutation for each player
    permutations = []
    for p in range(4):
        original = list(range(7))
        shuffled = original.copy()
        shard_rng.shuffle(shuffled)
        # shuffled[new_idx] = old_idx
        # So inverse gives: old_idx -> new_idx
        inv_perm = [0] * 7
        for new_idx, old_idx in enumerate(shuffled):
            inv_perm[old_idx] = new_idx
        permutations.append((shuffled, inv_perm))
        hands[p] = [hands[p][i] for i in shuffled]
```

**Target remapping**: When the oracle says "play action 3" (the 4th domino in sorted order), we need to translate that to the new shuffled index.

### Step 3: Add Manifest Flag

Track whether tokenized data used shuffled ordering:

```python
@dataclass
class TokenizationManifest:
    # ... existing fields ...
    shuffle_hand_order: bool = True  # NEW
```

### Step 4: Re-tokenize and Retrain

```bash
# Regenerate tokenized data with shuffling
python -m forge.cli.tokenize \
    --input data/solver2 \
    --output data/tokenized-shuffled \
    --force

# Retrain from scratch
python -m forge.cli.train \
    --data data/tokenized-shuffled \
    --output checkpoints/model-shuffled
```

## Impact Assessment

### What Changes

| Component | Change Required | Effort |
|-----------|-----------------|--------|
| `forge/ml/tokenize.py` | Add shuffle logic | Small |
| Tokenized data | Re-generate | ~30 min |
| Model training | Retrain from scratch | ~2 hours |
| Inference code | None (model handles any order) | None |

### What Stays the Same

| Component | Status |
|-----------|--------|
| Oracle parquet shards | Unchanged |
| State encoding | Unchanged |
| Q-value semantics | Unchanged |
| Model architecture | Unchanged |

### Inference Considerations

**After training with shuffled data**, the model learns that:
- Any domino can appear in any slot position
- Position carries no semantic meaning

**At inference time**:
- If hands are provided sorted (e.g., from `deal_from_seed()`), model works fine
- If hands are provided shuffled, model works fine
- Position-agnostic learning is the goal

## Expected Outcomes

### Before (Sorted Training)
| Slot | r(model, oracle) | Notes |
|------|------------------|-------|
| 0 | 0.81 | Only sees IDs 0-20 |
| 1-5 | 0.99+ | Sees full range |
| 6 | ~0.98 | Only sees IDs 7-27 |

### After (Shuffled Training)
| Slot | r(model, oracle) | Notes |
|------|------------------|-------|
| 0-6 | ~0.99 (expected) | All see uniform distribution |

## Alternative: Quick Fix for Validation

If full re-tokenization is not immediately feasible, a simpler validation fix:

```python
# In evaluation code, shuffle test hands before inference
def evaluate_with_shuffled_input(model, hands, ...):
    # Randomly permute each hand's ordering
    shuffled_hands = [random.sample(h, len(h)) for h in hands]
    # Run model on shuffled input
    # Compare to oracle on original ordering (remap outputs)
```

This validates whether shuffling at inference helps, without retraining.

## Conclusion

**Simplest fix**: Shuffle during tokenization using existing per-shard RNG.

**Cost**: Re-tokenize (~30 min) and retrain (~2 hours).

**Benefit**: Eliminates 1.74-bit KL divergence in slot 0 training distribution, expected to raise slot 0 correlation from 0.81 to ~0.99.

**Risk**: Low. The fix is localized to tokenization and uses deterministic RNG for reproducibility.
