# PyTorch Best Practices

Comprehensive guide for performance optimization, memory management, debugging, and production patterns.

## Performance Optimization

### Speed Checklist

```python
# 1. Use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 2. Enable cuDNN autotuner (for fixed input sizes)
torch.backends.cudnn.benchmark = True

# 3. Use mixed precision (see amp-compile.md)
# 4. Use torch.compile (see amp-compile.md)

# 5. Set matmul precision for Tensor Cores
torch.set_float32_matmul_precision("high")  # or "medium"
```

### DataLoader Optimization

```python
train_loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,              # Start here, increase if GPU waiting
    pin_memory=True,            # ALWAYS for GPU training
    prefetch_factor=2,          # Batches per worker to prefetch
    persistent_workers=True,    # Avoid worker restart overhead
    drop_last=True,             # Consistent batch sizes
)
```

**num_workers Tuning:**
1. Start with `num_workers=4`
2. Monitor GPU utilization (`nvidia-smi`)
3. If GPU < 90% utilized, increase workers
4. Stop when GPU is saturated or RAM issues appear
5. Rule of thumb: `4 * num_gpus`

### Avoid Synchronization Bottlenecks

```python
# BAD: Forces GPU sync every iteration
for batch in loader:
    loss = train_step(batch)
    print(f"Loss: {loss.item()}")  # .item() syncs!

# GOOD: Log periodically
for i, batch in enumerate(loader):
    loss = train_step(batch)
    if i % 100 == 0:
        print(f"Loss: {loss.item()}")
```

**Operations that cause sync:**
- `.item()`, `.cpu()`, `.numpy()`
- `print(tensor)`
- `torch.cuda.synchronize()`
- `torch.cuda.empty_cache()` (also bad for memory)

### Tensor Creation

```python
# BAD: Create on CPU then move
x = torch.randn(1000, 1000).cuda()

# GOOD: Create directly on device
x = torch.randn(1000, 1000, device="cuda")

# GOOD: Use same device as another tensor
y = torch.zeros_like(x)
z = torch.randn(100, device=x.device, dtype=x.dtype)
```

### Efficient Operations

```python
# BAD: Python loop over elements
result = torch.zeros(n)
for i in range(n):
    result[i] = tensor[i] * 2

# GOOD: Vectorized operation
result = tensor * 2

# BAD: Concatenate in loop
tensors = []
for i in range(n):
    tensors.append(compute(i))
result = torch.cat(tensors)

# GOOD: Pre-allocate if size known
result = torch.empty(n, dim)
for i in range(n):
    result[i] = compute(i)
```

### In-Place Operations

```python
# In-place can save memory but may break autograd
x.add_(1)      # In-place add
x.mul_(2)      # In-place multiply
x.zero_()      # In-place zero

# WARNING: Avoid in-place on tensors that require grad
# in the computation graph - can cause gradient errors
```

---

## Memory Management

### Reduce Memory Usage

```python
# 1. Use gradient checkpointing
from torch.utils.checkpoint import checkpoint

class MemoryEfficientModel(nn.Module):
    def forward(self, x):
        # Recompute activations during backward
        x = checkpoint(self.layer1, x, use_reentrant=False)
        x = checkpoint(self.layer2, x, use_reentrant=False)
        return self.output(x)

# 2. Use mixed precision (halves activation memory)
with torch.autocast("cuda"):
    output = model(input)

# 3. Delete intermediate tensors
del large_tensor
# Note: torch.cuda.empty_cache() is usually harmful

# 4. Use smaller batch size with gradient accumulation
```

### Monitor Memory

```python
# Current memory usage
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# Detailed summary
print(torch.cuda.memory_summary())

# Track peak memory
torch.cuda.reset_peak_memory_stats()
# ... run training
print(f"Peak: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
```

### Find Memory Leaks

```python
# Check if tensors are being held unnecessarily
import gc

def check_tensors():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                print(type(obj), obj.size(), obj.device)
        except:
            pass
```

### Inference Memory Optimization

```python
model.eval()

# Disable gradient tracking entirely
with torch.inference_mode():  # Preferred over no_grad
    for batch in test_loader:
        output = model(batch.to("cuda"))

# Or for older code
with torch.no_grad():
    output = model(input)
```

---

## Debugging

### Shape Debugging

```python
class DebugModel(nn.Module):
    def forward(self, x):
        print(f"Input: {x.shape}")

        x = self.layer1(x)
        print(f"After layer1: {x.shape}")

        x = self.layer2(x)
        print(f"After layer2: {x.shape}")

        return x

# Or use hooks
def shape_hook(module, input, output):
    print(f"{module.__class__.__name__}: {input[0].shape} -> {output.shape}")

for name, layer in model.named_modules():
    layer.register_forward_hook(shape_hook)
```

### Gradient Debugging

```python
# Check gradients exist and are reasonable
def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"WARNING: {name} has no gradient!")
        elif torch.isnan(param.grad).any():
            print(f"WARNING: {name} has NaN gradient!")
        elif (param.grad == 0).all():
            print(f"WARNING: {name} has zero gradient!")
        else:
            print(f"{name}: grad norm = {param.grad.norm():.6f}")

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)
# Now backward() will show which forward op caused NaN
```

### NaN/Inf Detection

```python
def training_step(batch):
    output = model(batch)
    loss = criterion(output, target)

    # Check for issues
    if torch.isnan(loss):
        print("NaN loss detected!")
        print(f"Output stats: min={output.min()}, max={output.max()}")
        print(f"Input stats: min={batch.min()}, max={batch.max()}")
        raise ValueError("NaN loss")

    return loss
```

### Reproducibility

```python
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For full determinism (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

### Profiling

```python
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=tensorboard_trace_handler("./profiler_logs"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step, batch in enumerate(train_loader):
        train_step(batch)
        prof.step()
        if step >= 5:
            break

# View: tensorboard --logdir=./profiler_logs
```

---

## Production Patterns

### Model Export: TorchScript

```python
model.eval()

# Tracing (works for most models)
example_input = torch.randn(1, 3, 224, 224)
traced = torch.jit.trace(model, example_input)
traced.save("model_traced.pt")

# Scripting (for control flow)
scripted = torch.jit.script(model)
scripted.save("model_scripted.pt")

# Load and run
loaded = torch.jit.load("model_traced.pt")
output = loaded(input_tensor)
```

### Model Export: ONNX

```python
model.eval()
example_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    example_input,
    "model.onnx",
    export_params=True,
    opset_version=17,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    },
)
```

### Inference Optimization

```python
model.eval()
model = model.to("cuda")

# 1. Use inference_mode
@torch.inference_mode()
def predict(inputs):
    return model(inputs)

# 2. Use torch.compile for speed
model = torch.compile(model, mode="reduce-overhead")

# 3. Warmup (especially with torch.compile)
for _ in range(3):
    _ = model(torch.randn(1, 3, 224, 224, device="cuda"))

# 4. Batch predictions when possible
```

### Freezing Layers

```python
# Freeze entire model
for param in model.parameters():
    param.requires_grad = False

# Freeze specific layers
for param in model.backbone.parameters():
    param.requires_grad = False

# Unfreeze classifier
for param in model.classifier.parameters():
    param.requires_grad = True

# Check what's trainable
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
```

### Weight Initialization

```python
def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

model.apply(init_weights)
```

---

## Common Patterns

### Early Stopping

```python
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False  # Continue training
        else:
            self.counter += 1
            return self.counter >= self.patience  # Stop if patience exceeded

early_stopping = EarlyStopping(patience=5)
for epoch in range(max_epochs):
    train_loss = train_epoch()
    val_loss = validate()

    if early_stopping(val_loss):
        print(f"Early stopping at epoch {epoch}")
        break
```

### Learning Rate Warmup

```python
def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return max(
            0.0,
            float(total_steps - step) / float(max(1, total_steps - warmup_steps)),
        )
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

### Model EMA (Exponential Moving Average)

```python
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1 - self.decay) * param.data
                )

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.shadow[name]
```

### Multi-Output Models

```python
class MultiTaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(...)
        self.classifier = nn.Linear(512, 10)
        self.regressor = nn.Linear(512, 1)

    def forward(self, x):
        features = self.backbone(x)
        return {
            "class": self.classifier(features),
            "value": self.regressor(features),
        }

# Training with multiple losses
outputs = model(inputs)
loss_cls = criterion_cls(outputs["class"], class_labels)
loss_reg = criterion_reg(outputs["value"], values)
loss = loss_cls + 0.5 * loss_reg
```
