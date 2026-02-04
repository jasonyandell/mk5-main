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

# Alternative when caller has original hands (e.g., from ActiveGame):
values = value_fn.batch_evaluate_with_originals(states, players, original_hands_list)
```

Note: `batch_evaluate_with_originals()` accepts pre-computed original hands. Both methods use vectorized post-processing for efficient GPU utilization.

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

### `run_selfplay_training.py` - Training Loop (Primary)

True AlphaZero self-play training:

- Epoch loop: generate games on GPU via MCTS -> train on MCTS targets
- Model provides both **policy priors** and **value** for MCTS leaf evaluation
- Checkpoints saved as `selfplay-epochXXXX.pt`
- Automatic W&B run resume from checkpoint's `wandb_run_id`
- `--keep-checkpoints N` to retain only last N checkpoints

This project is **CUDA-only** for training; there is no CPU training path in the default workflow.

### `run_mcts_training.py` - Oracle Training (Alternative)

Oracle-guided distillation (Stage1Oracle evaluates leaves instead of model).

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

### Default: Self-play training (AlphaZero-style)

```bash
python -u -m forge.zeb.run_selfplay_training \
    --checkpoint forge/zeb/checkpoints/selfplay-epoch0375.pt \
    --epochs 500 \
    --games-per-epoch 128 \
    --n-simulations 50 \
    --n-parallel-games 128 \
    --max-mcts-nodes 128 \
    --batch-size 64 \
    --lr 1e-4 \
    --eval-every 1 \
    --save-every 1 \
    --keep-checkpoints 3 \
    --wandb
```

### Key arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | (required) | Checkpoint to continue from |
| `--epochs` | 50 | Training epochs |
| `--games-per-epoch` | 512 | Games generated per epoch |
| `--n-simulations` | 100 | MCTS simulations per move |
| `--n-parallel-games` | 512 | Concurrent games per batch |
| `--max-mcts-nodes` | 256 | Max nodes per MCTS tree |
| `--batch-size` | 64 | SGD mini-batch size |
| `--lr` | 1e-4 | Learning rate |
| `--temperature` | 1.0 | MCTS temperature for exploration |
| `--eval-every` | 5 | Evaluate vs random every N epochs |
| `--save-every` | 5 | Save checkpoint every N epochs |
| `--keep-checkpoints` | 0 | Keep only last N checkpoints (0 = keep all) |
| `--wandb/--no-wandb` | True | Enable W&B logging |
| `--run-name` | auto | W&B run name (auto-generates timestamp) |

### Current Training Run

**Run ID**: `selfplay-500ep-from-e99` (W&B: `4jhso0ll`)
**Started from**: `selfplay-epoch0099.pt` (oracle-trained baseline)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `--n-simulations` | 100 | Doubled from 50 after plateau |
| `--n-parallel-games` | 256 | |
| `--max-mcts-nodes` | 256 | |
| `--games-per-epoch` | 128 | |
| `--batch-size` | 64 | |
| `--lr` | 1e-4 | |
| `--eval-every` | 1 | |
| `--save-every` | 1 | |
| `--keep-checkpoints` | 3 | |

**Performance**: ~4.5 games/s, ~31s/epoch total
- Game generation: ~28s
- Training: ~1s
- Eval (100 games): <2s (batched, 12x faster than sequential)

**Progress** (as of epoch 392):
- Win rate vs random (pair): ~55-70% (varies)
- Win rate vs random (solo): ~64%
- Total games: ~340k+
- Target: 100,000 epochs (runs indefinitely until stopped)

**Resume command**:
```bash
nohup python -u -m forge.zeb.run_selfplay_training \
  --checkpoint forge/zeb/checkpoints/selfplay-epoch0391.pt \
  --epochs 100000 \
  --games-per-epoch 128 \
  --n-simulations 100 \
  --n-parallel-games 256 \
  --max-mcts-nodes 256 \
  --lr 1e-4 \
  --batch-size 64 \
  --eval-every 1 \
  --save-every 1 \
  --keep-checkpoints 3 \
  --wandb \
  >> scratch/selfplay-500ep.log 2>&1 &
```

### Continuing training

Self-play saves checkpoints as `selfplay-epoch{N:04d}.pt`. To continue training, pass the most recent checkpoint:

```bash
python -u -m forge.zeb.run_selfplay_training \
    --checkpoint forge/zeb/checkpoints/selfplay-epoch0375.pt \
    --epochs 500 \
    --keep-checkpoints 3 \
    [...]
```

**W&B auto-resume**: If the checkpoint contains a `wandb_run_id`, the script automatically resumes that W&B run. No manual intervention needed - just pass the checkpoint and training continues with the same graphs and metrics.

Checkpoints include:
- Model and optimizer state
- Total games played (cumulative across all training runs)
- W&B run ID for seamless logging continuation
- Training config (mode, source checkpoint, hyperparameters)

---

## Key Design Decisions

### 1. Self-play bootstrapping (AlphaZero-style)

Training uses true self-play: the model guides MCTS with policy priors and evaluates leaves with its value head, then learns from MCTS visit targets and final outcomes. This keeps the entire loop on GPU and continuously improves checkpoints over time.

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

| Method | Throughput | GPU Util | Notes |
|--------|-----------|----------|-------|
| `batch_evaluate_with_originals()` | ~6,200/s | 92% | GameState objects |
| `batch_evaluate_gpu()` | ~7,300/s | **97%** | GPU tensors, fully GPU-native |
| Pure forward pass | ~6,600/s | 98% | Theoretical max |

**GPU-native path achieves 96% utilization**. The `batch_evaluate_gpu()` method takes pre-built GPU tensors and does all computation on GPU:
- GPU tokenization (GPUTokenizer)
- GPU bitmask computation
- GPU legal mask computation
- GPU post-processing

Key optimizations:
1. **GPU tokenization**: Tokenization moved to GPU (GPUTokenizer in `forge/eq/gpu_tokenizer.py`)
2. **GPU bitmask**: `compute_remaining_bitmask_gpu()` in `forge/zeb/gpu_preprocess.py`
3. **GPU legal mask**: `compute_legal_mask_gpu()` for action masking
4. **Vectorized post-processing**: Batch max() and team sign flip on GPU

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
