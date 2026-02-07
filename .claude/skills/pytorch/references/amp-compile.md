# PyTorch Mixed Precision (AMP) and torch.compile

Optimize training speed and memory with automatic mixed precision and JIT compilation.

## Automatic Mixed Precision (AMP)

Use lower precision (FP16/BF16) for faster computation and reduced memory.

### Basic Usage

```python
import torch
from torch.amp import autocast, GradScaler

model = MyModel().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scaler = GradScaler("cuda")

for data, target in train_loader:
    data, target = data.cuda(), target.cuda()

    optimizer.zero_grad()

    # Forward pass with autocast
    with autocast("cuda"):
        output = model(data)
        loss = criterion(output, target)

    # Backward pass with gradient scaling
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Key Components

**autocast**: Automatically chooses precision for each operation
- Math-heavy ops (matmul, conv) → FP16/BF16
- Numerically sensitive ops (softmax, layernorm) → FP32

**GradScaler**: Prevents gradient underflow in FP16
- Scales loss before backward (larger gradients)
- Unscales before optimizer step
- Skips update if NaN/Inf detected

### FP16 vs BF16

```python
# FP16 - wider range of GPUs (V100, A100, etc.)
with autocast("cuda", dtype=torch.float16):
    output = model(data)

# BF16 - better range, less overflow (A100, H100)
with autocast("cuda", dtype=torch.bfloat16):
    output = model(data)
```

| Type | GPU Support | GradScaler | Notes |
|------|-------------|------------|-------|
| FP16 | V100+, RTX 2000+ | Required | May overflow |
| BF16 | A100, H100, RTX 3000+ | Optional | Better stability |

### BF16 Training (No Scaler Needed)

```python
# BF16 is more stable, often doesn't need scaling
for data, target in train_loader:
    optimizer.zero_grad()

    with autocast("cuda", dtype=torch.bfloat16):
        output = model(data)
        loss = criterion(output, target)

    loss.backward()  # No scaler needed
    optimizer.step()
```

### Inference with AMP

```python
model.eval()

# No GradScaler needed for inference
with torch.inference_mode():
    with autocast("cuda"):
        output = model(input)
```

### Selective Autocast

```python
# Disable autocast for sensitive operations
with autocast("cuda"):
    output = model(data)

    # Force FP32 for this computation
    with autocast("cuda", enabled=False):
        sensitive_result = torch.linalg.solve(matrix, vector)
```

### Gradient Clipping with AMP

```python
scaler.scale(loss).backward()

# Unscale before clipping
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

scaler.step(optimizer)
scaler.update()
```

### Multi-GPU AMP (DDP)

```python
# AMP works seamlessly with DDP
model = DDP(model, device_ids=[rank])
scaler = GradScaler("cuda")

for data, target in train_loader:
    with autocast("cuda"):
        output = model(data)
        loss = criterion(output, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Troubleshooting AMP

```python
# Disable to check if AMP is causing issues
with autocast("cuda", enabled=False):
    output = model(data)

# Check for NaN in gradients
scaler.scale(loss).backward()
scaler.unscale_(optimizer)

for name, param in model.named_parameters():
    if param.grad is not None and torch.isnan(param.grad).any():
        print(f"NaN gradient in {name}")
```

---

## torch.compile

JIT compile models for significant speedups (PyTorch 2.0+).

### Basic Usage

```python
model = MyModel().cuda()

# Compile the model
compiled_model = torch.compile(model)

# Use normally
output = compiled_model(input)
```

### Compilation Modes

```python
# Default - good balance of compile time and speedup
model = torch.compile(model)

# Reduce overhead - faster for small models/batches
model = torch.compile(model, mode="reduce-overhead")

# Max autotune - best performance, longer compile time
model = torch.compile(model, mode="max-autotune")

# Max autotune without cudagraphs (for dynamic shapes)
model = torch.compile(model, mode="max-autotune-no-cudagraphs")
```

| Mode | Compile Time | Runtime Speed | Use Case |
|------|--------------|---------------|----------|
| default | Medium | Good | General use |
| reduce-overhead | Fast | Good | Dynamic workloads |
| max-autotune | Slow | Best | Serving, long training |

### Dynamic Shapes

```python
# Enable for variable input sizes
model = torch.compile(model, dynamic=True)

# Without dynamic=True, shape changes trigger recompilation
```

### Fullgraph Mode

```python
# Force entire model to compile (fails if graph breaks)
model = torch.compile(model, fullgraph=True)

# Useful for ensuring maximum optimization
```

### Graph Breaks

Graph breaks reduce optimization opportunities. Common causes:
- Python control flow based on tensor values
- Unsupported operations
- Data-dependent shapes

```python
# BAD: Causes graph break
def forward(self, x):
    if x.sum() > 0:  # Data-dependent condition
        return self.path_a(x)
    return self.path_b(x)

# BETTER: Use torch.where
def forward(self, x):
    return torch.where(x.sum() > 0, self.path_a(x), self.path_b(x))
```

### Regional Compilation

For large models, compile submodules:

```python
class LargeModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Compile repeated blocks individually
        self.block = torch.compile(TransformerBlock())
        self.layers = nn.ModuleList([self.block for _ in range(12)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
```

### Warmup Before Serving

```python
model = torch.compile(model)

# Warmup with representative inputs
for _ in range(3):
    with torch.no_grad():
        _ = model(torch.randn(1, 3, 224, 224, device="cuda"))

# Now ready for serving
```

### Debugging Compilation

```python
# See what's being compiled
import torch._dynamo as dynamo
dynamo.config.verbose = True

# Show graph breaks
torch._dynamo.config.suppress_errors = False

# Disable compilation for debugging
model = torch.compile(model, disable=True)
```

### Cache Compiled Models

```python
# Set cache directory
import torch._inductor.config as config
config.cache_dir = "/path/to/cache"

# Compiled kernels are cached across runs
```

---

## Combining AMP and torch.compile

```python
model = MyModel().cuda()
model = torch.compile(model)
scaler = GradScaler("cuda")

for data, target in train_loader:
    optimizer.zero_grad()

    with autocast("cuda"):
        output = model(data)
        loss = criterion(output, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Best Performance Settings

```python
# Maximum performance setup
torch.set_float32_matmul_precision("high")  # Use Tensor Cores
torch.backends.cudnn.benchmark = True        # Optimize conv algorithms

model = MyModel().cuda()
model = torch.compile(model, mode="max-autotune")
scaler = GradScaler("cuda")

# Warmup
for _ in range(3):
    with autocast("cuda"):
        _ = model(torch.randn(32, 3, 224, 224, device="cuda"))
```

---

## Performance Expectations

### Mixed Precision Speedup

| GPU | FP16 Speedup | BF16 Speedup |
|-----|--------------|--------------|
| V100 | 2-3x | N/A |
| A100 | 2-3x | 2-3x |
| H100 | 2-3x | 2-3x |
| RTX 3090 | 2x | 1.5-2x |

### torch.compile Speedup

| Model Type | Typical Speedup |
|------------|-----------------|
| Transformer | 1.5-2.5x |
| CNN | 1.3-2x |
| RNN/LSTM | 1.2-1.5x |

Actual speedups depend on model architecture, batch size, and GPU.

---

## Common Patterns

### Training Script Template

```python
import torch
from torch.amp import autocast, GradScaler

def train():
    # Setup
    model = MyModel().cuda()
    model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = GradScaler("cuda")
    criterion = torch.nn.CrossEntropyLoss()

    # Enable optimizations
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()

            with autocast("cuda"):
                output = model(data)
                loss = criterion(output, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        # Validation
        model.eval()
        with torch.inference_mode():
            with autocast("cuda"):
                for data, target in val_loader:
                    output = model(data.cuda())
                    # ... compute metrics
```

### Inference Script Template

```python
import torch
from torch.amp import autocast

def serve():
    model = MyModel().cuda()
    model.load_state_dict(torch.load("model.pt", weights_only=True))
    model.eval()
    model = torch.compile(model, mode="max-autotune")

    # Warmup
    dummy = torch.randn(1, 3, 224, 224, device="cuda")
    for _ in range(5):
        with torch.inference_mode():
            with autocast("cuda"):
                _ = model(dummy)

    # Serve
    @torch.inference_mode()
    def predict(inputs):
        with autocast("cuda"):
            return model(inputs)

    return predict
```
