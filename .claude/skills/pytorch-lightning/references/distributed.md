# PyTorch Lightning Distributed Training Reference

Scale training from single GPU to multi-node clusters with minimal code changes.

## Hardware Configuration

### Accelerator Options

```python
# Auto-detect best available
trainer = L.Trainer(accelerator="auto")

# Specific hardware
trainer = L.Trainer(accelerator="cpu")
trainer = L.Trainer(accelerator="gpu")
trainer = L.Trainer(accelerator="tpu")
trainer = L.Trainer(accelerator="mps")  # Apple Silicon
```

### Device Selection

```python
# Auto-select all available
trainer = L.Trainer(accelerator="gpu", devices="auto")

# Specific count
trainer = L.Trainer(accelerator="gpu", devices=4)

# Specific GPUs by index
trainer = L.Trainer(accelerator="gpu", devices=[0, 2])

# All available
trainer = L.Trainer(accelerator="gpu", devices=-1)
```

### Precision Options

```python
# Full precision (default)
trainer = L.Trainer(precision="32-true")

# Mixed precision FP16 (V100, older GPUs)
trainer = L.Trainer(precision="16-mixed")

# Mixed precision BF16 (A100/H100, recommended)
trainer = L.Trainer(precision="bf16-mixed")

# FP8 (H100 only)
trainer = L.Trainer(precision="transformer-engine")

# Double precision (for debugging)
trainer = L.Trainer(precision="64-true")
```

## Distributed Strategies

### DDP (Distributed Data Parallel)

Default multi-GPU strategy. Each GPU gets full model copy, gradients synchronized.

```python
# Auto-selected for multi-GPU
trainer = L.Trainer(accelerator="gpu", devices=4, strategy="ddp")

# Spawn mode (slower but more compatible)
trainer = L.Trainer(strategy="ddp_spawn")

# Find unused parameters (needed for some architectures)
trainer = L.Trainer(strategy=DDPStrategy(find_unused_parameters=True))
```

**When to use DDP:**
- Model fits in single GPU memory
- Standard training scenarios
- Best performance for most use cases

### FSDP (Fully Sharded Data Parallel)

Shards model parameters, gradients, and optimizer states across GPUs.

```python
from lightning.pytorch.strategies import FSDPStrategy

# Basic FSDP
trainer = L.Trainer(
    accelerator="gpu",
    devices=4,
    strategy="fsdp",
)

# Advanced configuration
strategy = FSDPStrategy(
    sharding_strategy="FULL_SHARD",  # FULL_SHARD, SHARD_GRAD_OP, NO_SHARD
    cpu_offload=True,  # Offload to CPU when not computing
    mixed_precision=None,  # Use Trainer precision instead
    activation_checkpointing_policy={
        torch.nn.TransformerEncoderLayer,
        torch.nn.TransformerDecoderLayer,
    },
)
trainer = L.Trainer(strategy=strategy)
```

**Model wrapping for FSDP:**

```python
class LitModel(L.LightningModule):
    def configure_model(self):
        # Wrap layers for FSDP sharding
        if self.trainer.strategy.name == "fsdp":
            self.model = wrap(self.model)
```

**When to use FSDP:**
- Large models (>10B parameters)
- GPU memory constrained
- Native PyTorch solution preferred

### DeepSpeed

Microsoft's optimization library with ZeRO stages.

```python
# Stage 2 (shard optimizer states + gradients)
trainer = L.Trainer(strategy="deepspeed_stage_2")

# Stage 2 with CPU offload
trainer = L.Trainer(strategy="deepspeed_stage_2_offload")

# Stage 3 (shard everything including parameters)
trainer = L.Trainer(strategy="deepspeed_stage_3")

# Stage 3 with CPU offload
trainer = L.Trainer(strategy="deepspeed_stage_3_offload")
```

**Custom DeepSpeed config:**

```python
from lightning.pytorch.strategies import DeepSpeedStrategy

strategy = DeepSpeedStrategy(
    stage=3,
    offload_optimizer=True,
    offload_parameters=True,
    cpu_checkpointing=True,
    pin_memory=True,
    contiguous_gradients=True,
    overlap_comm=True,
)

# Or use config file
strategy = DeepSpeedStrategy(config="deepspeed_config.json")
```

**ZeRO Stages:**
| Stage | Shards | Memory Reduction | Speed |
|-------|--------|------------------|-------|
| 1 | Optimizer states | 4x | Fastest |
| 2 | + Gradients | 8x | Fast |
| 3 | + Parameters | Linear with GPUs | Slowest |

**When to use DeepSpeed:**
- Very large models (100B+ parameters)
- Need CPU offloading
- Training with limited GPU memory

## Multi-Node Training

```python
# Single machine, multiple GPUs
trainer = L.Trainer(accelerator="gpu", devices=8, num_nodes=1)

# Multiple machines
trainer = L.Trainer(accelerator="gpu", devices=8, num_nodes=4)
```

### SLURM Setup

```bash
#!/bin/bash
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8

srun python train.py
```

Lightning auto-detects SLURM environment variables.

### Manual Multi-Node

Set environment variables on each node:

```bash
# Node 0 (master)
export MASTER_ADDR=10.0.0.1
export MASTER_PORT=12355
export WORLD_SIZE=32
export NODE_RANK=0

# Node 1
export NODE_RANK=1
# ... etc
```

## Sync and Communication

### Logging in Distributed Training

```python
# Sync across all GPUs (required for accurate metrics)
self.log("val_loss", loss, sync_dist=True)

# Only log on rank 0 (avoid duplicate logs)
self.log("train_loss", loss, rank_zero_only=True)
```

### All-Reduce Operations

```python
# Sync tensor across GPUs
synced_tensor = self.all_gather(tensor)

# Reduce with operation
reduced = self.all_reduce(tensor, reduce_op="mean")
```

### Barrier Synchronization

```python
# Wait for all processes
self.trainer.strategy.barrier()
```

## Best Practices

### Effective Batch Size

```
effective_batch = batch_size × devices × num_nodes × accumulate_grad_batches
```

```python
# 32 × 8 × 2 × 4 = 2048 effective batch size
trainer = L.Trainer(
    devices=8,
    num_nodes=2,
    accumulate_grad_batches=4,
)
```

### Learning Rate Scaling

Scale LR with effective batch size:

```python
base_lr = 1e-4
effective_batch = 32 * trainer.world_size
scaled_lr = base_lr * (effective_batch / 32)
```

### DataLoader Workers

```python
# Scale workers with GPUs
num_workers = 4 * torch.cuda.device_count()

# Avoid too many workers
num_workers = min(num_workers, os.cpu_count())
```

### Checkpointing Large Models

```python
from lightning.pytorch.callbacks import ModelCheckpoint

# Save sharded checkpoints for FSDP/DeepSpeed
checkpoint = ModelCheckpoint(
    save_weights_only=True,  # Smaller checkpoints
)
```

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| NCCL timeout | Increase `NCCL_TIMEOUT` env var |
| OOM on validation | Use `self.trainer.strategy.barrier()` before val |
| Hanging at start | Check firewall, MASTER_ADDR/PORT |
| Slow training | Check `num_workers`, pin_memory |
| NaN gradients | Reduce LR, check gradient clipping |

### Debug Distributed

```python
# Run with only 1 GPU to debug
trainer = L.Trainer(accelerator="gpu", devices=1)

# Verbose NCCL
os.environ["NCCL_DEBUG"] = "INFO"

# Check rank
print(f"Rank {trainer.global_rank} of {trainer.world_size}")
```
