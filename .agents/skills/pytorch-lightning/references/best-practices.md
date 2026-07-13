# PyTorch Lightning Best Practices

Comprehensive guide to data management, experiment tracking, debugging, and production patterns.

## Data Management

### DataLoader Configuration

```python
from torch.utils.data import DataLoader
import os

def create_dataloader(dataset, batch_size=32, shuffle=True, training=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=min(os.cpu_count(), 8),  # Start with CPU count, cap at 8
        pin_memory=True,                      # Faster GPU transfer
        prefetch_factor=2,                    # Samples to prefetch per worker
        persistent_workers=True,              # Keep workers alive between epochs
        drop_last=training,                   # Drop incomplete batch during training
    )
```

### num_workers Guidelines

| Workers | Behavior |
|---------|----------|
| 0 | Main process loads data (bottleneck!) |
| 1 | Single worker, still slow |
| 4-8 | Sweet spot for most systems |
| os.cpu_count() | Good starting point |

**Tuning Process:**
1. Start with `num_workers = 4`
2. Increase until training speed plateaus
3. Watch for CPU memory overflow (more workers = more RAM)
4. If GPU utilization < 90%, you likely need more workers

### LightningDataModule Best Practices

```python
class MyDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32):
        super().__init__()
        self.save_hyperparameters()  # Track for reproducibility

    def prepare_data(self):
        # ONLY download here - runs on single process
        # DO NOT assign self.x = ... (other processes won't see it)
        download_dataset(self.hparams.data_dir)

    def setup(self, stage: str):
        # Runs on ALL processes - safe to assign self.x
        if stage == "fit":
            full_dataset = MyDataset(self.hparams.data_dir)
            # Use generator for reproducible splits
            generator = torch.Generator().manual_seed(42)
            self.train_data, self.val_data = random_split(
                full_dataset, [0.8, 0.2], generator=generator
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )
```

### Data Loading Anti-Patterns

| DON'T | DO |
|-------|-----|
| Load entire dataset in `__init__` | Defer to `setup()` |
| Assign `self.x` in `prepare_data()` | Only download/preprocess |
| Use `num_workers=0` with GPU | Use at least 4 workers |
| Forget `pin_memory=True` for GPU | Always enable for GPU training |
| Create new workers each epoch | Use `persistent_workers=True` |

---

## Experiment Management

### Logging Best Practices

```python
class LitModel(L.LightningModule):
    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)

        # Basic logging
        self.log("train/loss", loss, prog_bar=True)

        # Batch-level AND epoch-level
        self.log("train/loss_step", loss, on_step=True, on_epoch=False)
        self.log("train/loss_epoch", loss, on_step=False, on_epoch=True)

        # Multiple metrics
        self.log_dict({
            "train/loss": loss,
            "train/acc": accuracy,
        }, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.compute_metrics(batch)

        # For distributed training, sync across GPUs
        self.log_dict({
            "val/loss": loss,
            "val/acc": acc,
        }, sync_dist=True, prog_bar=True)
```

### Metric Naming Conventions

```python
# Use hierarchical naming for organized dashboards
self.log("train/loss", ...)      # Training metrics
self.log("val/loss", ...)        # Validation metrics
self.log("test/acc", ...)        # Test metrics

# Group related metrics
self.log("loss/reconstruction", ...)
self.log("loss/kl_divergence", ...)
self.log("loss/total", ...)

# Track learning rate
self.log("optim/lr", self.trainer.optimizers[0].param_groups[0]["lr"])
```

### Weights & Biases Integration

```python
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

# Initialize logger
wandb_logger = WandbLogger(
    project="my-project",
    name="experiment-name",
    log_model="all",           # Log all checkpoints as artifacts
    save_dir="logs/",
    tags=["baseline", "v1"],
)

# Track model architecture and gradients
wandb_logger.watch(model, log="all", log_freq=100)

# Checkpoint callback
checkpoint = ModelCheckpoint(
    monitor="val/loss",
    mode="min",
    save_top_k=3,
    filename="{epoch}-{val_loss:.4f}",
)

trainer = L.Trainer(
    logger=wandb_logger,
    callbacks=[checkpoint],
)
```

### TensorBoard Integration

```python
from lightning.pytorch.loggers import TensorBoardLogger

logger = TensorBoardLogger(
    save_dir="logs/",
    name="my_model",
    version="v1",
    log_graph=True,  # Log model graph
)

# Log images, histograms in training_step
def training_step(self, batch, batch_idx):
    if batch_idx % 100 == 0:
        tensorboard = self.logger.experiment
        tensorboard.add_image("input", batch[0][0], self.global_step)
        tensorboard.add_histogram("weights", self.fc.weight, self.global_step)
```

### Multiple Loggers

```python
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger, CSVLogger

loggers = [
    TensorBoardLogger("logs/", name="tb"),
    WandbLogger(project="my-project"),
    CSVLogger("logs/", name="csv"),  # Always have CSV as backup
]

trainer = L.Trainer(logger=loggers)
```

### Hyperparameter Tracking

```python
class LitModel(L.LightningModule):
    def __init__(self, lr: float, hidden_dim: int, dropout: float):
        super().__init__()
        # Automatically saved to checkpoint AND logged to experiment tracker
        self.save_hyperparameters()

        # Access via self.hparams.lr, self.hparams.hidden_dim, etc.
        self.model = nn.Sequential(
            nn.Linear(784, self.hparams.hidden_dim),
            nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.hidden_dim, 10),
        )
```

---

## Debugging Techniques

### Quick Sanity Checks

```python
# Run 5 batches of train/val/test to catch bugs fast
trainer = L.Trainer(fast_dev_run=True)

# Or specify number of batches
trainer = L.Trainer(fast_dev_run=5)
```

### Overfit Test

```python
# If model can't overfit 10 samples, it won't learn from more
trainer = L.Trainer(
    overfit_batches=10,  # Use only 10 batches
    max_epochs=100,
)

# Should see training loss → 0
# If not, something is wrong with:
# - Model architecture
# - Loss function
# - Data pipeline
```

### Limit Data for Fast Iteration

```python
trainer = L.Trainer(
    limit_train_batches=0.1,  # Use 10% of training data
    limit_val_batches=0.1,    # Use 10% of validation data
    max_epochs=5,
)
```

### Layer Dimension Debugging

```python
class LitModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(...)

        # Set this to see input/output shapes for all layers
        self.example_input_array = torch.randn(1, 3, 224, 224)
```

### Detect NaN/Inf

```python
# Enable anomaly detection (slower but catches issues)
torch.autograd.set_detect_anomaly(True)

# Check for NaN in training
def training_step(self, batch, batch_idx):
    loss = self.compute_loss(batch)

    if torch.isnan(loss):
        print(f"NaN detected at batch {batch_idx}")
        print(f"Batch stats: {batch[0].mean()}, {batch[0].std()}")
        raise ValueError("NaN loss")

    return loss
```

### Distributed Debugging

```python
# Print from specific rank only
if self.trainer.global_rank == 0:
    print("Debug info from rank 0")

# Synchronize all processes
self.trainer.strategy.barrier()

# Check which rank you're on
print(f"Rank {self.trainer.global_rank} of {self.trainer.world_size}")
```

### Profiling

```python
# Simple profiler (text output)
trainer = L.Trainer(profiler="simple")

# Advanced profiler (detailed breakdown)
trainer = L.Trainer(profiler="advanced")

# PyTorch profiler with TensorBoard trace
from lightning.pytorch.profilers import PyTorchProfiler

profiler = PyTorchProfiler(
    dirpath="profiler_logs",
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler_logs"),
    record_shapes=True,
    with_stack=True,
)
trainer = L.Trainer(profiler=profiler)
```

---

## Performance Optimization

### Speed Checklist

```python
trainer = L.Trainer(
    # Hardware
    accelerator="gpu",
    devices="auto",

    # Precision (up to 3x speedup)
    precision="16-mixed",  # or "bf16-mixed" for A100/H100

    # Compilation (PyTorch 2.0+)
    # Add in configure_model(): self.model = torch.compile(self.model)

    # Reduce validation frequency
    check_val_every_n_epoch=5,
    val_check_interval=0.25,  # Validate 4x per epoch

    # Efficient checkpointing
    enable_checkpointing=True,
)
```

### Things to AVOID

```python
# DON'T: Call .item() in training loop (causes GPU sync)
loss_value = loss.item()  # Bad!

# DO: Log directly
self.log("loss", loss)  # Good!

# DON'T: Move tensors unnecessarily
x = x.cpu().numpy()  # Bad in training loop!

# DON'T: Clear CUDA cache (causes sync)
torch.cuda.empty_cache()  # Bad!

# DON'T: Create tensors on CPU then move
x = torch.tensor([1, 2, 3]).cuda()  # Bad!

# DO: Create directly on device
x = torch.tensor([1, 2, 3], device=self.device)  # Good!
```

### Memory Optimization

```python
# Gradient accumulation (larger effective batch without more memory)
trainer = L.Trainer(accumulate_grad_batches=4)

# Gradient checkpointing
from torch.utils.checkpoint import checkpoint

class LitModel(L.LightningModule):
    def forward(self, x):
        # Recompute during backward to save memory
        x = checkpoint(self.expensive_layer, x, use_reentrant=False)
        return self.output_layer(x)
```

---

## Reproducibility

### Full Reproducibility Setup

```python
import lightning as L

# Seed everything
L.seed_everything(42, workers=True)

trainer = L.Trainer(
    deterministic=True,  # Deterministic algorithms (slower)
)
```

### Checkpoint Everything

```python
class LitModel(L.LightningModule):
    def __init__(self, lr, hidden_dim):
        super().__init__()
        self.save_hyperparameters()  # Saves lr, hidden_dim to checkpoint

# Restore exactly
model = LitModel.load_from_checkpoint("checkpoint.ckpt")
# model.hparams.lr and model.hparams.hidden_dim are restored
```

---

## Common Mistakes & Fixes

### Shape Mismatch Debugging

```python
# Add shape assertions throughout
def forward(self, x):
    assert x.shape[1:] == (3, 224, 224), f"Expected (B, 3, 224, 224), got {x.shape}"

    x = self.encoder(x)
    assert x.shape[1:] == (512,), f"Expected (B, 512), got {x.shape}"

    return self.classifier(x)
```

### Mode Confusion (Train vs Eval)

```python
# Lightning handles this automatically, but for manual inference:
model = LitModel.load_from_checkpoint("checkpoint.ckpt")
model.eval()  # Sets dropout, batchnorm to eval mode
model.freeze()  # Disables gradient computation

with torch.inference_mode():
    predictions = model(inputs)
```

### Forgetting to Return Loss

```python
# DON'T: Forget to return loss
def training_step(self, batch, batch_idx):
    loss = self.compute_loss(batch)
    self.log("loss", loss)
    # Missing return! Training won't work.

# DO: Always return loss
def training_step(self, batch, batch_idx):
    loss = self.compute_loss(batch)
    self.log("loss", loss)
    return loss  # Required!
```

### Incorrect Validation Logging

```python
# DON'T: Forget sync_dist for distributed
def validation_step(self, batch, batch_idx):
    self.log("val_loss", loss)  # May be incorrect in multi-GPU

# DO: Sync across devices
def validation_step(self, batch, batch_idx):
    self.log("val_loss", loss, sync_dist=True)  # Correct!
```

---

## Practical How-Tos

### How to Resume Training

```python
# From checkpoint
trainer.fit(model, datamodule=dm, ckpt_path="path/to/checkpoint.ckpt")

# Find last checkpoint automatically
trainer.fit(model, datamodule=dm, ckpt_path="last")
```

### How to Fine-Tune a Pretrained Model

```python
class FineTunedModel(L.LightningModule):
    def __init__(self, num_classes, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()

        # Load pretrained backbone
        self.backbone = models.resnet50(weights="IMAGENET1K_V2")

        # Freeze backbone initially
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Replace classifier
        self.backbone.fc = nn.Linear(2048, num_classes)

    def configure_optimizers(self):
        # Only train classifier initially
        return torch.optim.Adam(self.backbone.fc.parameters(), lr=self.hparams.lr)


# Later: unfreeze and train everything
def unfreeze_backbone(self):
    for param in self.backbone.parameters():
        param.requires_grad = True
```

### How to Use Learning Rate Finder

```python
from lightning.pytorch.tuner import Tuner

model = LitModel()
trainer = L.Trainer()

tuner = Tuner(trainer)
lr_finder = tuner.lr_find(model, datamodule=dm)

# Plot results
fig = lr_finder.plot(suggest=True)
fig.savefig("lr_finder.png")

# Get suggested LR
suggested_lr = lr_finder.suggestion()
model.hparams.lr = suggested_lr
```

### How to Find Optimal Batch Size

```python
from lightning.pytorch.tuner import Tuner

tuner = Tuner(trainer)
tuner.scale_batch_size(model, datamodule=dm, mode="power")  # or "binsearch"
```

### How to Export for Production

```python
# TorchScript
model = LitModel.load_from_checkpoint("best.ckpt")
model.eval()
script = model.to_torchscript()
torch.jit.save(script, "model.pt")

# ONNX
model.to_onnx(
    "model.onnx",
    input_sample=torch.randn(1, 3, 224, 224),
    export_params=True,
    opset_version=13,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
)
```
