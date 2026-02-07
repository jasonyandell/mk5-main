---
name: pytorch
description: PyTorch development - activates for neural network implementation, nn.Module design, training loops, Dataset/DataLoader patterns, GPU optimization, distributed training (DDP), mixed precision (AMP), torch.compile, debugging, and converting models to production. Use when building deep learning models from scratch.
---

# PyTorch Development Guide

The foundational deep learning framework for research and production. Full control over every aspect of model training.

## Quick Reference

### Installation
```bash
# CUDA 12.4 (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Core Pattern
```
Tensor → nn.Module → Forward Pass → Loss → Backward → Optimizer Step
```

## nn.Module: The Building Block

All neural networks inherit from `nn.Module`. It manages parameters, submodules, and computation.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()  # Always call parent __init__

        # Learnable parameters (registered automatically)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)

        # Non-learnable buffer (moves with model to device)
        self.register_buffer("running_mean", torch.zeros(hidden_dim))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# Usage
model = MyModel(784, 256, 10)
model.to("cuda")  # Move to GPU
output = model(input_tensor)
```

### Essential nn.Module Methods

| Method | Purpose |
|--------|---------|
| `forward(x)` | Define computation (called via `model(x)`) |
| `parameters()` | Iterator over learnable parameters |
| `named_parameters()` | Iterator with parameter names |
| `state_dict()` | Get all parameters and buffers as dict |
| `load_state_dict(d)` | Load parameters from dict |
| `to(device)` | Move model to device (cuda/cpu/mps) |
| `train()` | Set training mode (enables dropout, etc.) |
| `eval()` | Set evaluation mode |
| `zero_grad()` | Zero all parameter gradients |

### Building with Sequential

```python
# Simple sequential model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 10),
)

# Named sequential
model = nn.Sequential(OrderedDict([
    ("fc1", nn.Linear(784, 256)),
    ("relu", nn.ReLU()),
    ("fc2", nn.Linear(256, 10)),
]))
```

### ModuleList and ModuleDict

```python
class DynamicModel(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        # ModuleList for indexed access
        self.layers = nn.ModuleList([
            nn.Linear(256, 256) for _ in range(num_layers)
        ])
        # ModuleDict for named access
        self.heads = nn.ModuleDict({
            "classifier": nn.Linear(256, 10),
            "regressor": nn.Linear(256, 1),
        })

    def forward(self, x, head_name="classifier"):
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.heads[head_name](x)
```

## Training Loop

The canonical PyTorch training pattern:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Setup
model = MyModel().to("cuda")
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# Training loop
for epoch in range(num_epochs):
    model.train()  # Enable training mode

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to("cuda"), target.to("cuda")

        optimizer.zero_grad()          # Clear gradients
        output = model(data)           # Forward pass
        loss = criterion(output, target)  # Compute loss
        loss.backward()                # Backward pass
        optimizer.step()               # Update weights

    scheduler.step()  # Update learning rate

    # Validation
    model.eval()  # Disable dropout, etc.
    with torch.no_grad():  # Disable gradient computation
        for data, target in val_loader:
            data, target = data.to("cuda"), target.to("cuda")
            output = model(data)
            # ... compute metrics
```

### Training Loop Essentials

| Step | Code | Purpose |
|------|------|---------|
| Zero gradients | `optimizer.zero_grad()` | Clear accumulated gradients |
| Forward | `output = model(data)` | Compute predictions |
| Loss | `loss = criterion(output, target)` | Compute error |
| Backward | `loss.backward()` | Compute gradients |
| Update | `optimizer.step()` | Apply gradients to weights |

### Gradient Accumulation

Simulate larger batch sizes when GPU memory is limited:

```python
accumulation_steps = 4
optimizer.zero_grad()

for i, (data, target) in enumerate(train_loader):
    output = model(data.to("cuda"))
    loss = criterion(output, target.to("cuda"))
    loss = loss / accumulation_steps  # Normalize loss
    loss.backward()  # Accumulate gradients

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Gradient Clipping

Prevent exploding gradients:

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

## Dataset and DataLoader

### Custom Dataset

```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = self._load_samples()  # Load file paths, not data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Load data on-demand (not in __init__)
        sample, label = self._load_sample(self.samples[idx])

        if self.transform:
            sample = self.transform(sample)

        return sample, label
```

### DataLoader Configuration

```python
from torch.utils.data import DataLoader
import os

train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,               # Shuffle for training
    num_workers=4,              # Parallel data loading
    pin_memory=True,            # Faster GPU transfer
    prefetch_factor=2,          # Batches to prefetch per worker
    persistent_workers=True,    # Keep workers alive between epochs
    drop_last=True,             # Drop incomplete final batch
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,              # Can use larger batch for inference
    shuffle=False,              # Keep order for validation
    num_workers=4,
    pin_memory=True,
)
```

### num_workers Guidelines

| Workers | Use Case |
|---------|----------|
| 0 | Debugging (single process) |
| 4 | Good starting point |
| os.cpu_count() | Maximum parallelism |
| 4 * num_gpus | Rule of thumb for multi-GPU |

**Tuning**: Start at 4, increase until GPU utilization is high. Watch for RAM overflow.

### Custom Collate Function

For variable-length sequences or complex batching:

```python
def collate_fn(batch):
    # batch is list of (sample, label) tuples
    samples, labels = zip(*batch)

    # Pad sequences to same length
    samples = nn.utils.rnn.pad_sequence(samples, batch_first=True)
    labels = torch.stack(labels)

    return samples, labels

loader = DataLoader(dataset, collate_fn=collate_fn)
```

## Optimizers

### Common Optimizers

```python
# SGD with momentum
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

# Adam (good default)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# AdamW (Adam with proper weight decay)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
```

### Per-Parameter Learning Rates

```python
optimizer = optim.AdamW([
    {"params": model.backbone.parameters(), "lr": 1e-5},  # Lower LR for pretrained
    {"params": model.head.parameters(), "lr": 1e-3},     # Higher LR for new layers
], weight_decay=0.01)
```

### Learning Rate Schedulers

```python
# Step decay
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Cosine annealing
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# Warmup + cosine
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-3, total_steps=len(train_loader) * epochs
)

# Usage: call after optimizer.step()
for epoch in range(epochs):
    for batch in train_loader:
        # ... training step
        optimizer.step()
        scheduler.step()  # For OneCycleLR (per-step)

    # scheduler.step()  # For StepLR, CosineAnnealing (per-epoch)
```

## Saving and Loading

### Save/Load Checkpoints

```python
# Save checkpoint
checkpoint = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "loss": loss,
}
torch.save(checkpoint, "checkpoint.pt")

# Load checkpoint
checkpoint = torch.load("checkpoint.pt", weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
epoch = checkpoint["epoch"]
```

### Save for Inference Only

```python
# Save just the model weights
torch.save(model.state_dict(), "model_weights.pt")

# Load weights
model = MyModel()
model.load_state_dict(torch.load("model_weights.pt", weights_only=True))
model.eval()
```

## Device Management

```python
# Check availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model
model = model.to(device)

# Move data in training loop
data, target = data.to(device), target.to(device)

# Create tensors on device (preferred over .to())
x = torch.randn(32, 784, device=device)

# Multi-GPU: get current device
current_device = torch.cuda.current_device()
```

## Inference

```python
model.eval()  # Disable dropout, use running stats for BatchNorm

with torch.no_grad():  # Disable gradient computation (saves memory)
    output = model(input_tensor)

# Or use inference_mode (slightly faster)
with torch.inference_mode():
    output = model(input_tensor)
```

## Common Loss Functions

```python
# Classification
criterion = nn.CrossEntropyLoss()         # Multi-class (includes softmax)
criterion = nn.BCEWithLogitsLoss()        # Binary (includes sigmoid)
criterion = nn.NLLLoss()                  # Use with log_softmax output

# Regression
criterion = nn.MSELoss()                  # Mean squared error
criterion = nn.L1Loss()                   # Mean absolute error
criterion = nn.SmoothL1Loss()             # Huber loss

# Contrastive/Embedding
criterion = nn.TripletMarginLoss()
criterion = nn.CosineEmbeddingLoss()
```

## Anti-Patterns

| DON'T | DO |
|-------|-----|
| `x = x.cuda()` in forward | Move model once with `model.to(device)` |
| `loss.item()` every batch | Log every N steps (causes sync) |
| `torch.cuda.empty_cache()` | Let PyTorch manage memory |
| Create tensors then `.to(device)` | Create on device: `torch.randn(..., device=device)` |
| Forget `model.train()`/`model.eval()` | Always set mode explicitly |
| Skip `optimizer.zero_grad()` | Gradients accumulate by default! |
| Use `DataParallel` | Use `DistributedDataParallel` instead |
| Python loops over tensor elements | Use vectorized operations |

## Debugging Tips

```python
# Check tensor shapes
print(f"Input: {x.shape}, Output: {output.shape}")

# Check for NaN/Inf
assert not torch.isnan(loss), "NaN loss detected"
assert torch.isfinite(output).all(), "Inf in output"

# Enable anomaly detection (slow but catches issues)
torch.autograd.set_detect_anomaly(True)

# Profile memory
print(torch.cuda.memory_summary())

# Check gradient flow
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad mean={param.grad.mean():.6f}")
```

## Reference Files

For detailed documentation on specific topics:
- `references/best-practices.md` - Performance, memory, debugging, production patterns
- `references/distributed.md` - DDP, multi-GPU, multi-node training
- `references/amp-compile.md` - Mixed precision (AMP) and torch.compile optimization
