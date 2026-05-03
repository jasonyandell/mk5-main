---
name: wandb
description: Weights & Biases development - activates for experiment tracking, hyperparameter sweeps, artifact versioning, model registry, reports, and ML collaboration. Use when logging training runs, comparing experiments, managing datasets/models, or setting up team workflows.
---

# Weights & Biases (W&B) Development Guide

The AI developer platform for experiment tracking, model management, and team collaboration. Track everything from hyperparameters to model weights, compare experiments, and share results.

## Quick Reference

### Installation & Setup
```bash
pip install wandb
wandb login  # Authenticate with API key
```

Or set environment variable:
```bash
export WANDB_API_KEY=<your_api_key>
```

### Core Pattern
```
wandb.init() → wandb.log() → wandb.finish()
      ↓              ↓              ↓
   Start run    Log metrics    End run
```

## Basic Experiment Tracking

```python
import wandb

# Start a run
with wandb.init(project="my-project", name="experiment-1") as run:
    # Log hyperparameters
    run.config.learning_rate = 0.001
    run.config.epochs = 10
    run.config.batch_size = 32

    # Training loop
    for epoch in range(10):
        loss = train_one_epoch()
        accuracy = evaluate()

        # Log metrics
        run.log({
            "train/loss": loss,
            "val/accuracy": accuracy,
            "epoch": epoch,
        })

    # Log summary metrics
    run.summary["best_accuracy"] = best_acc
```

## wandb.init() Parameters

```python
wandb.init(
    # Identity
    project="my-project",       # Project name (required for organization)
    entity="my-team",           # Username or team name
    name="run-name",            # Display name in UI
    id="unique-id",             # For resuming runs

    # Metadata
    config={                    # Hyperparameters
        "lr": 0.001,
        "epochs": 10,
    },
    tags=["baseline", "v1"],    # For filtering/organizing
    notes="Testing new arch",   # Detailed description
    group="experiment-1",       # Group related runs
    job_type="train",           # Run type (train/eval/preprocess)

    # Behavior
    mode="online",              # "online", "offline", "disabled"
    resume="allow",             # "allow", "never", "must", "auto"
    save_code=True,             # Save git state and code
)
```

## Logging

### Metrics
```python
# Single metric
wandb.log({"loss": 0.5})

# Multiple metrics (recommended - logged to same step)
wandb.log({
    "train/loss": 0.5,
    "train/accuracy": 0.85,
    "learning_rate": 0.001,
})

# Custom x-axis (epoch instead of step)
wandb.define_metric("val/*", step_metric="epoch")
wandb.log({"epoch": 5, "val/loss": 0.3})
```

### Metric Naming Conventions
```python
# Hierarchical naming for organized dashboards
wandb.log({
    "train/loss": train_loss,      # Training metrics
    "val/loss": val_loss,          # Validation metrics
    "test/accuracy": test_acc,     # Test metrics
    "optim/lr": current_lr,        # Optimizer metrics
})
```

**CRITICAL**: Use letters, digits, and underscores only. Avoid hyphens, spaces, or special characters in metric names.

### Summary Metrics
```python
# Track best/final values (shown in run table)
wandb.run.summary["best_val_accuracy"] = 0.95
wandb.run.summary["final_loss"] = 0.01
```

## Configuration

```python
# Method 1: Pass to init
wandb.init(config={"lr": 0.001, "epochs": 10})

# Method 2: Update after init
wandb.config.update({
    "model": "resnet50",
    "optimizer": "adamw",
})

# Method 3: From argparse
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.001)
args = parser.parse_args()
wandb.init(config=args)

# Method 4: From YAML file
wandb.init(config="config.yaml")

# Access config values
lr = wandb.config.lr
# or
lr = wandb.config["lr"]
```

## Tables (Structured Data)

```python
# Create and log a table
table = wandb.Table(
    columns=["image", "prediction", "ground_truth", "confidence"],
    data=[
        [wandb.Image(img1), "cat", "cat", 0.95],
        [wandb.Image(img2), "dog", "cat", 0.60],
    ]
)
wandb.log({"predictions": table})

# From pandas DataFrame
import pandas as pd
df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
wandb.log({"my_table": wandb.Table(dataframe=df)})
```

## Media Logging

```python
# Images
wandb.log({"image": wandb.Image(array)})
wandb.log({"image": wandb.Image("path/to/image.png")})
wandb.log({"images": [wandb.Image(img) for img in batch]})

# Images with masks/boxes
wandb.log({
    "predictions": wandb.Image(image, masks={
        "predictions": {"mask_data": mask_array, "class_labels": class_labels}
    })
})

# Audio
wandb.log({"audio": wandb.Audio(array, sample_rate=44100)})

# Video
wandb.log({"video": wandb.Video(array, fps=4)})

# Plots (matplotlib)
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 9])
wandb.log({"chart": wandb.Image(fig)})
plt.close()

# Histograms
wandb.log({"gradients": wandb.Histogram(gradient_values)})

# 3D point clouds
wandb.log({"point_cloud": wandb.Object3D(points_array)})
```

## PyTorch Lightning Integration

```python
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

# Initialize logger
wandb_logger = WandbLogger(
    project="my-project",
    name="experiment-1",
    log_model="all",          # Log all checkpoints as artifacts
    save_dir="logs/",
    tags=["baseline"],
)

# Optional: watch model for gradient/parameter histograms
wandb_logger.watch(model, log="all", log_freq=100)

# Checkpoint callback
checkpoint = ModelCheckpoint(
    monitor="val/loss",
    mode="min",
    save_top_k=3,
)

# Train
trainer = Trainer(
    logger=wandb_logger,
    callbacks=[checkpoint],
    max_epochs=100,
)
trainer.fit(model, datamodule)

# In LightningModule - metrics auto-logged
def training_step(self, batch, batch_idx):
    loss = self.compute_loss(batch)
    self.log("train/loss", loss)  # Auto-synced to W&B
    return loss
```

## Alerts

```python
# Send alert from code
wandb.alert(
    title="Training Complete",
    text=f"Final accuracy: {accuracy:.2%}",
    level=wandb.AlertLevel.INFO,  # INFO, WARN, ERROR
)

# Alert on condition
if loss > threshold:
    wandb.alert(
        title="High Loss Detected",
        text=f"Loss {loss} exceeded threshold {threshold}",
        level=wandb.AlertLevel.WARN,
    )
```

Configure Slack/email alerts in W&B settings.

## Model Checkpointing

```python
# Save model as artifact
artifact = wandb.Artifact("model", type="model")
artifact.add_file("model.pt")
wandb.log_artifact(artifact)

# With aliases
wandb.log_artifact(artifact, aliases=["latest", "best"])

# Load model from artifact
artifact = wandb.use_artifact("my-project/model:best")
artifact_dir = artifact.download()
model.load_state_dict(torch.load(f"{artifact_dir}/model.pt"))
```

## Offline Mode

```python
# Run without internet
wandb.init(mode="offline")

# ... training ...

# Later, sync offline runs
# $ wandb sync ./wandb/offline-run-*
```

## Best Practices

### Project Organization
```
Entity (Team/User)
    └── Project (one per experiment type)
        └── Runs (individual training runs)
            └── Groups (related runs, e.g., hyperparameter search)
```

### Run Naming Convention
```python
wandb.init(
    name=f"{model_name}-lr{lr}-bs{batch_size}",
    tags=["baseline", "resnet", "imagenet"],
    group="architecture-search",
    job_type="train",
)
```

### Efficient Logging
- Log every N steps, not every batch: `if step % 100 == 0: wandb.log(...)`
- Batch related metrics in single log call
- Use `wandb.define_metric()` for custom x-axes
- Keep images under 50 per step

### Common Anti-Patterns

| DON'T | DO |
|-------|-----|
| Log every single batch | Log every N steps |
| Use hyphens in metric names | Use underscores: `train_loss` |
| Forget `wandb.finish()` | Use `with wandb.init()` context |
| Call `wandb.log(step=...)` in Lightning | Let Lightning handle steps |
| Create new runs for resumed training | Use `resume="must"` with run ID |

## Reference Files

For detailed documentation on specific topics:
- `references/sweeps.md` - Hyperparameter tuning with W&B Sweeps
- `references/artifacts.md` - Dataset and model versioning
- `references/reports.md` - Creating and sharing reports
- `references/organization.md` - Teams, projects, registry, governance
