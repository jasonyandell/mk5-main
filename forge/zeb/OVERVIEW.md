# Zeb MCTS Training Pipeline

Oracle-guided MCTS distillation for training ZebModel to play Texas 42.

---

## Architecture Overview

This pipeline trains a small transformer (ZebModel) by distilling knowledge from a pre-trained oracle through MCTS-guided self-play. The key insight is that MCTS with oracle leaf evaluation produces high-quality policy targets (visit distributions) that are far less noisy than REINFORCE.

```
                            +-----------------+
                            |   Oracle        |
                            |  (Stage1Oracle) |
                            |  3.3M params    |
                            +-----------------+
                                    |
                                    | leaf evaluation
                                    v
     +------------+          +-----------+          +-------------+
     | Deal hands | -------> |   MCTS    | -------> | Visit counts|
     +------------+          | (batched) |          | (policy     |
                             +-----------+          |  targets)   |
                                    |               +-------------+
                                    |                      |
                                    v                      v
                             +-----------+          +-------------+
                             | Game      |          | ZebModel    |
                             | outcome   |          | learns to   |
                             | (value    |          | predict     |
                             |  target)  |          | both        |
                             +-----------+          +-------------+
```

**Why distillation instead of AlphaZero?**

The classic AlphaZero feedback loop (network guides MCTS, MCTS trains network) requires the network itself to provide useful value estimates from day one. For imperfect information games like Texas 42, bootstrapping from random is extremely slow.

Instead, we use a pre-trained oracle (Stage1Oracle, trained on perfect information game trees) as a fixed "teacher" for leaf evaluation. The student (ZebModel) learns to predict the MCTS visit distributions and game outcomes. This provides stable, high-quality training signal from the start.

---

## Key Files

### `mcts.py` - Core MCTS Engine

Implements determinized UCT (Upper Confidence bounds for Trees) with:

- **UCB selection**: Balances exploration vs exploitation with configurable `c_puct`
- **Virtual loss**: Allows parallel leaf collection without visiting same nodes
- **Batched search**: Collects wave of leaves, batch evaluates, then backpropagates (6x speedup)
- **Random rollout fallback**: Works without oracle for testing

Key classes:
- `MCTSNode`: Tree node with visit count, value sum, children
- `MCTS`: Single-game MCTS with `search()` returning visit counts
- `DeterminizedMCTS`: Root sampling for imperfect information (samples opponent hands)

```python
# Basic usage
mcts = MCTS(n_simulations=100, value_fn=oracle_fn)
visits = mcts.search(state, player=0)  # Returns {domino_id: visit_count}
```

### `mcts_self_play.py` - Game Generation

Plays full games using MCTS, converts to training examples:

- `play_game_with_mcts()`: Single game with MCTS decisions
- `MCTSTrainingExample`: State + action_probs + outcome
- `mcts_examples_to_zeb_tensors()`: Converts to model-ready format

**Critical transformation**: Domino ID to hand slot mapping:
```python
# Oracle works with domino IDs (0-27)
# ZebModel works with hand slots (0-6)
# This function maps MCTS visit counts (by domino ID) to slot probabilities

domino_to_slot = {d: slot for slot, d in enumerate(original_hand)}
slot_probs[slot] = action_probs[domino_id]
```

### `oracle_value.py` - Oracle Wrapper

Wraps Stage1Oracle for MCTS leaf evaluation:

- Converts `GameState` to oracle query format
- Returns max(Q-values) normalized to [-1, 1]
- Supports single evaluation and batch evaluation
- Handles team perspective (flips sign for Team 1)
- Vectorized bitmask computation using numpy broadcasting

```python
value_fn = create_oracle_value_fn(device='cuda', compile=True)
value = value_fn(state, player)  # Returns float in [-1, 1]

# For batched MCTS (much faster):
values = value_fn.batch_evaluate(states, players)  # Returns list of floats

# Optimized version when caller has original hands (e.g., from ActiveGame):
values = value_fn.batch_evaluate_with_originals(states, players, original_hands_list)
```

The `batch_evaluate_with_originals()` method skips expensive hand reconstruction when the caller already has access to original hands, providing ~6x speedup.

### `batched_mcts.py` - Cross-Game Batching

Maintains multiple concurrent MCTS searches to maximize GPU utilization:

- **Before**: 32 leaves/batch from 1 game -> 40% GPU utilization
- **After**: 500+ leaves/batch from 16 games -> 90%+ GPU utilization

Features:
- Cross-game leaf batching for larger GPU batches
- Async double-buffering using CUDA streams (overlaps CPU/GPU work)
- Passes pre-computed `original_hands` to oracle for fast preprocessing

```python
coordinator = BatchedMCTSCoordinator(
    n_parallel_games=16,
    target_batch_size=512,
    value_fn=oracle_fn,
)
games = coordinator.play_games(n_games=100)
```

The double-buffering pattern overlaps CPU preprocessing with GPU evaluation:
while GPU evaluates batch N, CPU collects leaves and prepares batch N+1.

### `run_mcts_training.py` - Training Loop

Main training script with:

- Epoch loop: generate games -> convert to tensors -> train batch
- W&B logging for policy/value loss, games/sec, oracle queries
- Periodic evaluation against random opponent
- Model size configs (small/medium/large)

### `observation.py` - Token Encoding

Encodes game state as 36 tokens x 8 features:

```
Token 0:      Declaration (decl_id, token_type=0)
Tokens 1-7:   Player's hand (7 fixed slots, masked if played)
Tokens 8-35:  Play history (up to 28 plays)
```

8 features per token:
| Feature | Range | Description |
|---------|-------|-------------|
| FEAT_HIGH | 0-6 | High pip value |
| FEAT_LOW | 0-6 | Low pip value |
| FEAT_IS_DOUBLE | 0-1 | Is double domino |
| FEAT_COUNT | 0-2 | Point value (0/5/10) |
| FEAT_PLAYER | 0-3 | Relative player (0=me) |
| FEAT_IS_IN_HAND | 0-1 | Still in hand vs played |
| FEAT_DECL | 0-9 | Declaration ID |
| FEAT_TOKEN_TYPE | 0-2 | Decl/Hand/Play |

### `model.py` - ZebModel Architecture

Transformer with policy + value heads:

```python
class ZebModel(nn.Module):
    embeddings: ZebEmbeddings      # 8-feature -> embed_dim
    pos_embed: nn.Embedding        # Position embeddings
    encoder: TransformerEncoder    # Pre-LN transformer
    policy_proj: nn.Linear         # [embed] -> [1] per hand slot
    value_head: nn.Sequential      # Mean pool -> tanh scalar
```

Model configs:
| Size | embed_dim | n_layers | n_heads | ff_dim | Params |
|------|-----------|----------|---------|--------|--------|
| small | 64 | 2 | 2 | 128 | ~25K |
| medium | 128 | 4 | 4 | 256 | ~150K |
| large | 256 | 6 | 8 | 512 | ~600K |

---

## Training Commands

### Basic training (CPU, no oracle)

```bash
python -m forge.zeb.run_mcts_training \
    --epochs 20 \
    --games-per-epoch 100 \
    --n-simulations 50 \
    --model-size small \
    --device cpu
```

### Oracle-guided training (GPU recommended)

```bash
python -m forge.zeb.run_mcts_training \
    --epochs 50 \
    --games-per-epoch 200 \
    --n-simulations 100 \
    --model-size medium \
    --use-oracle \
    --oracle-device cuda \
    --device cuda \
    --n-parallel-games 16 \
    --cross-game-batch-size 512
```

### Key arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 20 | Training epochs |
| `--games-per-epoch` | 100 | Games generated per epoch |
| `--n-simulations` | 50 | MCTS simulations per move |
| `--model-size` | small | small/medium/large |
| `--use-oracle` | False | Use oracle for leaf evaluation |
| `--n-parallel-games` | 16 | Concurrent games for batching |
| `--cross-game-batch-size` | 512 | Target oracle batch size |
| `--wandb/--no-wandb` | True | Enable W&B logging |

---

## Key Design Decisions

### 1. Distillation over feedback loop

AlphaZero trains by having the network guide MCTS, then training on MCTS outputs. This creates a feedback loop where early noise can compound.

We instead use a fixed, pre-trained oracle as the teacher. The student network (ZebModel) never influences MCTS leaf evaluation during training. This provides stable, high-quality gradients from epoch 1.

### 2. Observation encoding: 36x8 tokens preserving play order

The encoding preserves the temporal structure of play:
- Hand slots are fixed (indices 1-7) so the model learns slot-based policies
- Play history appears in order (indices 8-35) for pattern recognition
- Relative player IDs (0=me, 1=left, 2=partner, 3=right) for position-invariant learning

### 3. Domino ID to slot mapping

MCTS operates on domino IDs (0-27) because `GameState` uses domino IDs for actions. But ZebModel outputs slot logits (0-6) because the observation encoding uses fixed hand slots.

The `mcts_examples_to_zeb_tensors()` function performs this mapping:
```python
original_hand = (5, 12, 18, 23, 24, 26, 27)  # Fixed at deal time
domino_to_slot = {5: 0, 12: 1, 18: 2, 23: 3, 24: 4, 26: 5, 27: 6}

# MCTS says: play domino 18 with 40% probability
# Training target: slot 2 gets 40%
```

### 4. Batched leaf evaluation

Single-leaf oracle queries underutilize the GPU. The batched MCTS implementation:

1. Collects `wave_size` leaves per game using virtual loss
2. Runs multiple games in parallel (`n_parallel_games`)
3. Batches all non-terminal leaves across all games
4. Single GPU forward pass for 500+ states

This provides ~6x speedup over sequential evaluation.

---

## Performance

### Game generation speed

| Configuration | Games/sec | Notes |
|---------------|-----------|-------|
| Random rollout (CPU) | ~50 | No oracle, 50 sims |
| Oracle sequential (GPU) | ~0.5 | One leaf at a time |
| Oracle batched (GPU) | ~1-2 | 16 games, 512 batch |
| Oracle batched + optimized | ~3-5 | With preprocessing optimizations |

### Oracle throughput

| Method | States/sec | Notes |
|--------|-----------|-------|
| `batch_evaluate()` (original) | ~1,200 | Hand reconstruction bottleneck |
| `batch_evaluate_with_originals()` | ~7,000+ | Skips reconstruction |
| GPU-only (`query_batch_multi_state`) | ~6,250 | Theoretical max |

Key optimizations:
1. **Cached original hands**: `ActiveGame` stores original hands; passed to `batch_evaluate_with_originals()` to skip expensive reconstruction (~6x speedup)
2. **Vectorized bitmask**: Numpy broadcasting replaces nested Python loops for remaining-domino computation
3. **Async double-buffering**: CUDA streams overlap CPU preprocessing with GPU evaluation

### Training example generation

Each game produces 28 training examples (one per move). With 200 games/epoch:
- 5,600 examples per epoch
- ~60 examples/second with batched oracle

### Model inference speed

ZebModel (medium) on CPU: ~10,000 states/second

---

## See Also

- `ML_OVERVIEW.md` - Background on RL concepts, REINFORCE, hyperparameters
- `forge/eq/oracle.py` - Stage1Oracle implementation
- `forge/ORIENTATION.md` - Overall ML pipeline architecture
