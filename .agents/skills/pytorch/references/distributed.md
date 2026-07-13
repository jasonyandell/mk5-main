# PyTorch Distributed Training

Multi-GPU and multi-node training with DistributedDataParallel (DDP).

## DDP vs DataParallel

| Feature | DataParallel | DistributedDataParallel |
|---------|--------------|-------------------------|
| Parallelism | Single-process, multi-thread | Multi-process |
| GIL | Blocked by GIL | No GIL issues |
| Speed | Slower | Faster |
| Multi-machine | No | Yes |
| Recommended | Never | Always |

**Always use DDP**, even for single-machine multi-GPU.

## Basic DDP Setup

### Single Machine, Multiple GPUs

```python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)

    # Create model and move to GPU
    model = MyModel().to(rank)
    model = DDP(model, device_ids=[rank])

    # Distributed sampler
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)  # Important for shuffling!

        model.train()
        for data, target in train_loader:
            data, target = data.to(rank), target.to(rank)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # Save only from rank 0
        if rank == 0:
            torch.save(model.module.state_dict(), "checkpoint.pt")

    cleanup()

def main():
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
```

### Using torchrun (Recommended)

Simpler launching without manual process management:

```python
# train_ddp.py
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

def main():
    # torchrun sets these environment variables
    dist.init_process_group("nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)

    model = MyModel().to(local_rank)
    model = DDP(model, device_ids=[local_rank])

    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler)

    # Training loop...

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

Launch with:
```bash
# Single node, 4 GPUs
torchrun --nproc_per_node=4 train_ddp.py

# Multi-node (run on each node)
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 \
    --master_addr=10.0.0.1 --master_port=12355 train_ddp.py
```

## Key Concepts

### Ranks and World Size

```python
rank = dist.get_rank()           # Global process ID (0 to world_size-1)
local_rank = int(os.environ["LOCAL_RANK"])  # GPU ID on this node (0 to nproc-1)
world_size = dist.get_world_size()  # Total number of processes
```

### DistributedSampler

```python
# Creates non-overlapping data splits across processes
sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
    drop_last=True,  # Ensure all ranks have same batch count
)

# CRITICAL: Set epoch for proper shuffling
for epoch in range(epochs):
    sampler.set_epoch(epoch)  # Must call this!
    for batch in loader:
        ...
```

### Accessing Original Model

```python
# DDP wraps model in .module attribute
ddp_model = DDP(model, device_ids=[rank])

# Access original model
original = ddp_model.module

# Save weights (from rank 0 only)
if rank == 0:
    torch.save(ddp_model.module.state_dict(), "weights.pt")
```

### SyncBatchNorm

```python
# Convert BatchNorm to SyncBatchNorm BEFORE wrapping with DDP
model = MyModel()
model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
model = model.to(rank)
model = DDP(model, device_ids=[rank])
```

**When to use:** Always for vision models, especially with small per-GPU batch sizes.

## Gradient Accumulation with DDP

```python
accumulation_steps = 4

for i, (data, target) in enumerate(train_loader):
    # Use no_sync to skip gradient sync on non-final steps
    with model.no_sync() if (i + 1) % accumulation_steps != 0 else nullcontext():
        output = model(data)
        loss = criterion(output, target) / accumulation_steps
        loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## Communication Primitives

### All-Reduce

```python
# Sum tensor across all processes
tensor = torch.tensor([rank], device=f"cuda:{rank}")
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
# tensor now contains sum across all ranks

# Average
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
tensor = tensor / world_size
```

### Broadcast

```python
# Send tensor from rank 0 to all others
tensor = torch.randn(10, device=f"cuda:{rank}")
dist.broadcast(tensor, src=0)
```

### All-Gather

```python
# Gather tensors from all ranks
local_tensor = torch.tensor([rank], device=f"cuda:{rank}")
gathered = [torch.zeros_like(local_tensor) for _ in range(world_size)]
dist.all_gather(gathered, local_tensor)
# gathered = [tensor_from_rank_0, tensor_from_rank_1, ...]
```

### Barrier

```python
# Wait for all processes to reach this point
dist.barrier()
```

## Checkpointing

```python
def save_checkpoint(model, optimizer, epoch, path):
    if dist.get_rank() == 0:  # Only rank 0 saves
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, path)
    dist.barrier()  # Wait for save to complete

def load_checkpoint(model, optimizer, path):
    # All ranks load (with map_location to avoid GPU 0 overload)
    checkpoint = torch.load(path, map_location=f"cuda:{dist.get_rank()}")
    model.module.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"]
```

## Logging and Metrics

```python
def log_metrics(metrics, rank):
    # Only log from rank 0
    if rank == 0:
        print(f"Loss: {metrics['loss']:.4f}, Acc: {metrics['acc']:.4f}")

def gather_metrics(loss, correct, total, world_size):
    # Gather metrics across all ranks
    metrics = torch.tensor([loss, correct, total], device="cuda")
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)

    avg_loss = metrics[0].item() / world_size
    accuracy = metrics[1].item() / metrics[2].item()
    return {"loss": avg_loss, "acc": accuracy}
```

## Multi-Node Training

### SLURM Example

```bash
#!/bin/bash
#SBATCH --job-name=ddp_training
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16

# Get master address
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12355

srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train_ddp.py
```

### Manual Multi-Node

On each node:
```bash
# Node 0 (master)
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 \
    --master_addr=10.0.0.1 --master_port=12355 train.py

# Node 1
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=1 \
    --master_addr=10.0.0.1 --master_port=12355 train.py
```

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| NCCL timeout | Increase `NCCL_TIMEOUT`, check network |
| Hanging at init | Check `MASTER_ADDR`, `MASTER_PORT`, firewall |
| OOM on GPU 0 | Use `map_location` when loading checkpoints |
| Different losses | Ensure same random seed, `set_epoch()` called |
| Slow training | Check `num_workers`, network bandwidth |

### Debug Mode

```python
# Enable NCCL debug output
os.environ["NCCL_DEBUG"] = "INFO"

# Check rank
print(f"Rank {dist.get_rank()} on {socket.gethostname()}")

# Verify all ranks reach same point
print(f"Rank {dist.get_rank()} at checkpoint 1")
dist.barrier()
print(f"Rank {dist.get_rank()} passed checkpoint 1")
```

## Scaling Guidelines

### Batch Size Scaling

```
effective_batch_size = batch_size_per_gpu × num_gpus × accumulation_steps
```

When scaling GPUs, keep effective batch size constant or scale learning rate:
```python
# Linear scaling rule
base_lr = 1e-3
base_batch = 32
scaled_lr = base_lr * (effective_batch_size / base_batch)
```

### Performance Tips

1. Use NCCL backend for GPU training
2. Set `find_unused_parameters=False` if all parameters used every forward
3. Use gradient accumulation with `no_sync()` context
4. Pin memory and use enough DataLoader workers
5. Use SyncBatchNorm for vision models
