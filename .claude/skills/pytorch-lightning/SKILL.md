---
name: pytorch-lightning
description: PyTorch Lightning development - activates for deep learning training, LightningModule design, Trainer configuration, callbacks, distributed training (DDP/FSDP/DeepSpeed), checkpointing, logging, and converting PyTorch to Lightning. Use when building scalable ML training pipelines.
---

# PyTorch Lightning Development Guide

High-level PyTorch framework that eliminates boilerplate while maintaining flexibility. Scales from laptop to supercomputer with zero code changes.

## Quick Reference

### Installation
```bash
pip install lightning
# or with extras
pip install "lightning[extra]"
```

### Core Pattern
```
LightningModule (what to train)
    +
LightningDataModule (data pipeline)
    +
Trainer (how to train)
    =
Reproducible Training
```

## LightningModule Structure

The fundamental building block - organizes your model, training logic, and optimization.

```python
import lightning as L
import torch
import torch.nn.functional as F

class LitModel(L.LightningModule):
    def __init__(self, hidden_dim=128, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()  # Saves to checkpoint

        self.encoder = torch.nn.Linear(784, hidden_dim)
        self.decoder = torch.nn.Linear(hidden_dim, 10)

    def forward(self, x):
        # Inference path
        return self.decoder(F.relu(self.encoder(x)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # Log metrics (auto-aggregated across batches)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
```

### Essential Methods

| Method | Purpose |
|--------|---------|
| `__init__` | Initialize layers, call `save_hyperparameters()` |
| `forward` | Inference/prediction path |
| `training_step` | Training loop logic, return loss |
| `validation_step` | Validation logic |
| `test_step` | Test logic |
| `predict_step` | Inference logic for `trainer.predict()` |
| `configure_optimizers` | Return optimizer(s) and scheduler(s) |

### Lifecycle Hooks

```
setup() → configure_model() → configure_optimizers()
    ↓
on_train_start() → [epochs] → on_train_end()
    ↓ per epoch
on_train_epoch_start() → [batches] → on_train_epoch_end()
    ↓ per batch
on_before_batch_transfer() → training_step() → on_after_backward()
```

## Trainer Configuration

The Trainer automates 40+ engineering best practices.

```python
trainer = L.Trainer(
    # Duration
    max_epochs=100,
    max_steps=-1,  # -1 = no limit

    # Hardware
    accelerator="auto",  # "cpu", "gpu", "tpu", "mps"
    devices="auto",      # Number or list: 1, [0,1], "auto"
    precision="16-mixed", # "32", "16-mixed", "bf16-mixed"

    # Distributed (see references/distributed.md)
    strategy="auto",  # "ddp", "fsdp", "deepspeed_stage_2"
    num_nodes=1,

    # Validation
    val_check_interval=1.0,  # Check every N epochs (float) or steps (int)
    check_val_every_n_epoch=1,

    # Checkpointing
    enable_checkpointing=True,
    default_root_dir="./lightning_logs",

    # Logging
    logger=True,  # TensorBoard by default
    log_every_n_steps=50,

    # Debugging
    fast_dev_run=False,  # Run 1 batch for sanity check
    overfit_batches=0.0, # Overfit on N batches
    limit_train_batches=1.0,
    limit_val_batches=1.0,

    # Callbacks (see references/callbacks.md)
    callbacks=[],

    # Reproducibility
    deterministic=True,
)
```

### Training Workflow

```python
# Training
trainer.fit(model, train_dataloaders, val_dataloaders)

# Resume from checkpoint
trainer.fit(model, ckpt_path="path/to/checkpoint.ckpt")

# Validation only
trainer.validate(model, dataloaders)

# Testing
trainer.test(model, dataloaders)

# Prediction
predictions = trainer.predict(model, dataloaders)
```

## LightningDataModule

Encapsulates data pipeline for reproducibility and sharing.

```python
class MNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir="./data", batch_size=32, num_workers=4):
        super().__init__()
        self.save_hyperparameters()

    def prepare_data(self):
        # Download (called on 1 GPU only)
        MNIST(self.hparams.data_dir, train=True, download=True)
        MNIST(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Called on every GPU
        if stage == "fit" or stage is None:
            full = MNIST(self.hparams.data_dir, train=True, transform=self.transform)
            self.train_data, self.val_data = random_split(full, [55000, 5000])
        if stage == "test" or stage is None:
            self.test_data = MNIST(self.hparams.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers)

# Usage
dm = MNISTDataModule(batch_size=64)
trainer.fit(model, datamodule=dm)
```

## Logging Metrics

```python
# Single metric
self.log("train_loss", loss)

# Multiple metrics
self.log_dict({"loss": loss, "acc": acc})

# With options
self.log("train_loss", loss,
         on_step=True,      # Log per step
         on_epoch=True,     # Aggregate per epoch
         prog_bar=True,     # Show in progress bar
         logger=True,       # Send to logger
         sync_dist=True)    # Sync across GPUs (for distributed)
```

### Logger Integrations

```python
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

# TensorBoard (default)
logger = TensorBoardLogger("logs", name="my_model")

# Weights & Biases
logger = WandbLogger(project="my_project", name="run_name")

trainer = L.Trainer(logger=logger)
```

## Checkpointing

```python
from lightning.pytorch.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",
    filename="{epoch}-{val_loss:.2f}",
    save_top_k=3,
    monitor="val_loss",
    mode="min",
    save_last=True,
)

trainer = L.Trainer(callbacks=[checkpoint_callback])
```

### Loading Checkpoints

```python
# Load for inference
model = LitModel.load_from_checkpoint("path/to/checkpoint.ckpt")
model.eval()

# Load with modified hparams
model = LitModel.load_from_checkpoint("checkpoint.ckpt", lr=1e-4)

# Resume training
trainer.fit(model, ckpt_path="checkpoint.ckpt")
```

## Common Patterns

### Converting PyTorch to Lightning

**Before (PyTorch):**
```python
model = Model()
optimizer = torch.optim.Adam(model.parameters())
for epoch in range(epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = model.training_step(batch)
        loss.backward()
        optimizer.step()
```

**After (Lightning):**
```python
class LitModel(L.LightningModule):
    def training_step(self, batch, batch_idx):
        return self.model.training_step(batch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

trainer = L.Trainer(max_epochs=epochs)
trainer.fit(model, dataloader)
```

### Gradient Accumulation

```python
# Accumulate gradients over 4 batches (effective batch = batch_size * 4)
trainer = L.Trainer(accumulate_grad_batches=4)
```

### Gradient Clipping

```python
trainer = L.Trainer(gradient_clip_val=0.5)  # Norm clipping
trainer = L.Trainer(gradient_clip_val=0.5, gradient_clip_algorithm="value")  # Value clipping
```

### Mixed Precision Training

```python
# FP16 (V100, older GPUs)
trainer = L.Trainer(precision="16-mixed")

# BF16 (A100/H100, recommended)
trainer = L.Trainer(precision="bf16-mixed")
```

## Anti-Patterns

| DON'T | DO |
|-------|-----|
| `optimizer.zero_grad()` in training_step | Let Trainer handle it |
| `loss.backward()` in training_step | Return loss, Trainer calls backward |
| `model.train()`/`model.eval()` manually | Trainer handles mode switching |
| `torch.cuda.set_device()` | Use `accelerator` and `devices` |
| Global state in `prepare_data()` | Only download, don't assign `self.x` |
| Heavy compute in `__init__` | Defer to `setup()` or `configure_model()` |

## Debugging Tips

```python
# Fast sanity check (1 train + 1 val batch)
trainer = L.Trainer(fast_dev_run=True)

# Overfit on small subset
trainer = L.Trainer(overfit_batches=10)

# Detect anomalies
torch.autograd.set_detect_anomaly(True)

# Profile training
trainer = L.Trainer(profiler="simple")  # or "advanced", "pytorch"
```

## Reference Files

For detailed documentation on specific topics:
- `references/best-practices.md` - Data management, experiment tracking, debugging, practical how-tos
- `references/callbacks.md` - Built-in and custom callbacks
- `references/distributed.md` - DDP, FSDP, DeepSpeed strategies
- `references/advanced.md` - Hyperparameter tuning, CLI, production patterns
