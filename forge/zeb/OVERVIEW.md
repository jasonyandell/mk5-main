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
| `--replay-buffer-size` | 50000 | Replay buffer size (0 = no buffer) |
| `--wandb/--no-wandb` | True | Enable W&B logging |
| `--run-name` | auto | W&B run name (auto-generates timestamp) |

### Current Training Run

**Run ID**: `selfplay-500ep-from-e99` (W&B: `4jhso0ll`)
**Started from**: `selfplay-epoch0099.pt` (oracle-trained baseline)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `--n-simulations` | 200 | Increased for better MCTS quality |
| `--n-parallel-games` | 256 | |
| `--max-mcts-nodes` | 512 | Increased for deeper search trees |
| `--games-per-epoch` | 256 | Matched to parallel games for full GPU utilization |
| `--batch-size` | 64 | |
| `--lr` | 1e-4 | |
| `--eval-every` | 1 | |
| `--save-every` | 1 | |
| `--keep-checkpoints` | 3 | |
| `--replay-buffer-size` | 50000 | ~7 epochs of history for stable training |

**Performance** (after kernel reduction optimizations):
- Game generation: ~69s (200 sims, ~3.7 games/s)
- Training: ~2.5s
- Eval (100 games): ~2s (batched)
- Total epoch: ~75s without eval

**Progress** (as of epoch 2499):
- Win rate vs random (pair): ~65-70%
- Win rate vs random (solo): ~64%
- Total games: ~640k+
- Replay buffer: 50k examples (full)
- Target: 100,000 epochs (runs indefinitely until stopped)

**Resume command**:
```bash
nohup python -u -m forge.zeb.run_selfplay_training \
  --checkpoint forge/zeb/checkpoints/selfplay-epoch2499.pt \
  --epochs 100000 \
  --games-per-epoch 256 \
  --n-simulations 200 \
  --n-parallel-games 256 \
  --max-mcts-nodes 512 \
  --lr 1e-4 \
  --batch-size 64 \
  --eval-every 10 \
  --save-every 10 \
  --keep-checkpoints 3 \
  --replay-buffer-size 50000 \
  --training-steps 1000 \
  --eval-games 2000 \
  --wandb \
  >> scratch/selfplay.log 2>&1 &
```

### Continuing training

Self-play saves checkpoints as `selfplay-epoch{N:04d}.pt`. To continue training, pass the most recent checkpoint:

```bash
python -u -m forge.zeb.run_selfplay_training \
    --checkpoint forge/zeb/checkpoints/selfplay-epoch2499.pt \
    --epochs 100000 \
    --keep-checkpoints 3 \
    --replay-buffer-size 50000 \
    [...]
```

**W&B auto-resume**: If the checkpoint contains a `wandb_run_id`, the script automatically resumes that W&B run. No manual intervention needed - just pass the checkpoint and training continues with the same graphs and metrics.

**Replay buffer auto-resume**: The replay buffer is saved in checkpoints and restored on resume. Training continues with the full buffer of historical examples.

Checkpoints include:
- Model and optimizer state
- Total games played (cumulative across all training runs)
- W&B run ID for seamless logging continuation
- Replay buffer (list of recent training examples)
- Training config (mode, source checkpoint, hyperparameters)

**Note**: Checkpoint size is ~47 MB with a 50k replay buffer (vs ~6.5 MB without).

---

## Key Design Decisions

### 1. Self-play bootstrapping (AlphaZero-style)

Training uses true self-play: the model guides MCTS with policy priors and evaluates leaves with its value head, then learns from MCTS visit targets and final outcomes. This keeps the entire loop on GPU and continuously improves checkpoints over time.

### 2. Replay buffer for stable training

Instead of training only on current epoch's games, we maintain a rolling buffer of ~50k recent examples (~7 epochs). Benefits:
- **Decorrelates examples**: Random sampling breaks temporal correlation within games
- **Smoother gradients**: Training distribution changes gradually, not abruptly each epoch
- **Data efficiency**: Each example trains the model multiple times before eviction
- **Stability**: Prevents policy oscillation from training on single-policy batches

The buffer is persisted in checkpoints for seamless resume.

### 3. Observation encoding: 36x8 tokens preserving play order

The encoding preserves the temporal structure of play:
- Hand slots are fixed (indices 1-7) so the model learns slot-based policies
- Play history appears in order (indices 8-35) for pattern recognition
- Relative player IDs (0=me, 1=left, 2=partner, 3=right) for position-invariant learning

### 4. Domino ID to slot mapping

MCTS operates on domino IDs (0-27) because `GameState` uses domino IDs for actions. But ZebModel outputs slot logits (0-6) because the observation encoding uses fixed hand slots.

The `mcts_examples_to_zeb_tensors()` function performs this mapping:
```python
original_hand = (5, 12, 18, 23, 24, 26, 27)  # Fixed at deal time
domino_to_slot = {5: 0, 12: 1, 18: 2, 23: 3, 24: 4, 26: 5, 27: 6}

# MCTS says: play domino 18 with 40% probability
# Training target: slot 2 gets 40%
```

### 5. Batched leaf evaluation

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

## Modal B200 Training

Self-play training can run on Modal's B200 GPUs via `forge/modal_app.py::zeb_train`.

### Basic usage

```bash
# Standard training run
modal run forge/modal_app.py::zeb_train \
    --checkpoint forge/zeb/selfplay-epoch1845.pt \
    --epochs 100

# Profiling mode: adds deliberate gaps between phases
modal run forge/modal_app.py::zeb_train \
    --checkpoint forge/zeb/selfplay-epoch1845.pt \
    --epochs 3 \
    --profile-mode
```

### B200 vs 3050 Ti Performance (Feb 2026)

| Metric | B200 | 3050 Ti | Speedup |
|--------|------|---------|---------|
| **Avg epoch** | 43.7s | 98s | **2.2x** |
| Gen time | 35.2s | 82s | 2.3x |
| Train time | 8.3s | 15-17s | 1.9x |
| Games/sec | 7.3 | 3.1 | 2.3x |

**Observation**: Only 2.2x speedup (not the expected 10-20x from B200's raw compute advantage).

**Root cause**: Kernel launch overhead. Each CUDA graph replay contained ~980 tiny kernels
from unrolled Python loops. With thousands of replays per game batch, both GPUs spent similar
time on dispatch — the B200's wider SM array was starved.

### Kernel Reduction Optimizations (Feb 2026)

Four optimizations to reduce kernel count and Python dispatch overhead:

1. **Vectorize tokenizer play history loop**: Replaced `for play_idx in range(28)` (~252 kernels)
   with batched gather/where (~12 kernels) in `gpu_training_pipeline.py`
2. **Vectorize backprop depth loop**: Replaced `for depth in range(max_depth)` (~224 kernels)
   with flat gather + `scatter_add_` (~10 kernels) in `gpu_mcts.py`
3. **Depth-variant CUDA graphs**: Capture at depths [28, 8, 1], select smallest sufficient per
   move. Average ~40% reduction in select+backprop kernels across 28 moves.
4. **Multi-step CUDA graph capture**: Capture K=10 simulation steps per graph, replay N/K times.
   Reduces 50 Python `graph.replay()` calls to 5.

**Measured impact** (3050 Ti, n_sims=50, N=128):
- Gen time: 40.4s → 16.3s (**2.4x speedup**)
- At production settings (n_sims=200, N=256, max_nodes=512): ~10-12% wall-clock improvement,
  dramatically higher GPU utilization (kernels now large enough that real compute dominates)

### Profile Mode

The `--profile-mode` flag adds deliberate 30s gaps between phases:
- Creates clear visual separation on GPU utilization graphs
- Logs phase start/end timestamps to W&B (`profile/phase`, `profile/elapsed_s`)
- Uses fresh W&B run named `b200-profiling-{timestamp}`
- Useful for diagnosing performance issues

---

## See Also

- `ML_OVERVIEW.md` - Background on RL concepts, REINFORCE, hyperparameters
- `forge/eq/oracle.py` - Stage1Oracle implementation
- `forge/ORIENTATION.md` - Overall ML pipeline architecture
