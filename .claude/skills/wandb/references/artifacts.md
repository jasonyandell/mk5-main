# W&B Artifacts: Data & Model Versioning

Track datasets, models, and other files with automatic versioning and lineage tracking.

## Core Concepts

**Artifact**: A versioned collection of files (dataset, model, config, etc.)
**Version**: Immutable snapshot (v0, v1, v2...)
**Alias**: Mutable pointer to a version ("latest", "best", "production")
**Lineage**: Graph showing how artifacts relate to runs

## Common Use Cases

| Scenario | Input Artifact | Output Artifact |
|----------|----------------|-----------------|
| Training | Dataset | Trained Model |
| Preprocessing | Raw Data | Processed Data |
| Evaluation | Model + Test Set | Results Table |
| Inference | Model | Predictions |

## Creating Artifacts

### Basic Pattern
```python
import wandb

with wandb.init(project="my-project") as run:
    # Create artifact
    artifact = wandb.Artifact(
        name="my-dataset",
        type="dataset",  # "model", "dataset", "predictions", etc.
        description="Training dataset v1",
        metadata={"num_samples": 10000, "split": "train"},
    )

    # Add files
    artifact.add_file("data/train.csv")
    artifact.add_dir("data/images/")

    # Log to W&B
    run.log_artifact(artifact)
```

### Adding Files
```python
# Single file
artifact.add_file("model.pt")

# File with different name in artifact
artifact.add_file("local/path/model.pt", name="model.pt")

# Entire directory
artifact.add_dir("data/images/")

# Directory with prefix
artifact.add_dir("data/images/", name="train_images")

# Reference external file (S3, GCS, etc.)
artifact.add_reference("s3://my-bucket/large-file.h5")
```

### Model Artifacts
```python
import torch

with wandb.init() as run:
    # Save model checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
    }, "checkpoint.pt")

    # Create and log artifact
    artifact = wandb.Artifact(
        name="model",
        type="model",
        metadata={
            "architecture": "resnet50",
            "accuracy": 0.95,
            "epoch": epoch,
        },
    )
    artifact.add_file("checkpoint.pt")
    run.log_artifact(artifact, aliases=["latest", "best"])
```

## Using Artifacts

### Download and Use
```python
with wandb.init() as run:
    # Declare input artifact (tracks lineage)
    artifact = run.use_artifact("my-dataset:latest")

    # Download to local directory
    artifact_dir = artifact.download()

    # Use files
    data = pd.read_csv(f"{artifact_dir}/train.csv")
```

### Download Specific Version
```python
# By version number
artifact = run.use_artifact("my-dataset:v3")

# By alias
artifact = run.use_artifact("my-dataset:latest")
artifact = run.use_artifact("my-dataset:production")

# Full path with entity/project
artifact = run.use_artifact("my-team/my-project/my-dataset:v3")
```

### Download Without a Run
```python
# For inference or standalone scripts
api = wandb.Api()
artifact = api.artifact("my-team/my-project/model:best")
artifact_dir = artifact.download()
```

### Selective Download
```python
# Download specific file only
path = artifact.get_path("model.pt").download()

# Download to custom directory
artifact.download(root="./models/")
```

## Versioning

### Automatic Versioning
Each `log_artifact()` creates a new version (v0, v1, v2...) if content changes.

```python
# First log: creates v0
run.log_artifact(artifact)

# Modify content and log again: creates v1
artifact.add_file("updated_data.csv")
run.log_artifact(artifact)
```

### Aliases
```python
# Add aliases when logging
run.log_artifact(artifact, aliases=["latest", "best"])

# Update alias programmatically
api = wandb.Api()
artifact = api.artifact("my-project/model:v5")
artifact.aliases.append("production")
artifact.save()
```

## Lineage Tracking

W&B automatically tracks:
- **Input artifacts**: Artifacts used by a run (`use_artifact()`)
- **Output artifacts**: Artifacts produced by a run (`log_artifact()`)

View the lineage graph in the W&B UI under the artifact's "Graph" tab.

```python
with wandb.init() as run:
    # Input: dataset artifact
    dataset = run.use_artifact("training-data:latest")
    dataset_dir = dataset.download()

    # ... train model ...

    # Output: model artifact
    model_artifact = wandb.Artifact("trained-model", type="model")
    model_artifact.add_file("model.pt")
    run.log_artifact(model_artifact)
```

## Lightning Integration

```python
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

wandb_logger = WandbLogger(
    project="my-project",
    log_model="all",  # Log all checkpoints as artifacts
)

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    mode="min",
    save_top_k=3,
    filename="{epoch}-{val_loss:.2f}",
)

trainer = Trainer(
    logger=wandb_logger,
    callbacks=[checkpoint_callback],
)

# Checkpoints automatically saved as artifacts with "latest" and "best" aliases
```

### Load Lightning Checkpoint from Artifact
```python
# Download best checkpoint
api = wandb.Api()
artifact = api.artifact("my-team/my-project/model-xxx:best")
artifact_dir = artifact.download()

# Load model
model = MyLightningModule.load_from_checkpoint(
    f"{artifact_dir}/model.ckpt"
)
```

## Artifact Types

Common artifact types (you can use any string):

| Type | Description |
|------|-------------|
| `dataset` | Training/validation/test data |
| `model` | Model weights and checkpoints |
| `predictions` | Model outputs |
| `config` | Configuration files |
| `code` | Source code snapshots |

```python
wandb.Artifact(name="my-artifact", type="dataset")
```

## Metadata and Description

```python
artifact = wandb.Artifact(
    name="my-model",
    type="model",
    description="ResNet-50 trained on ImageNet",
    metadata={
        "architecture": "resnet50",
        "framework": "pytorch",
        "accuracy": 0.95,
        "num_params": 25_000_000,
        "training_data": "imagenet-1k",
    },
)
```

Access metadata later:
```python
artifact = run.use_artifact("my-model:latest")
print(artifact.metadata["accuracy"])
print(artifact.description)
```

## External Storage (References)

For large files stored externally (S3, GCS, Azure):

```python
artifact = wandb.Artifact("large-dataset", type="dataset")

# Reference without copying
artifact.add_reference("s3://my-bucket/data/large-file.h5")
artifact.add_reference("gs://my-bucket/images/", max_objects=10000)

run.log_artifact(artifact)
```

Download resolves the reference:
```python
artifact = run.use_artifact("large-dataset:latest")
artifact.download()  # Fetches from S3/GCS
```

## Data Governance

### TTL (Time to Live)
```python
# Set artifact to expire after 30 days
artifact.ttl = timedelta(days=30)
```

### Delete Artifacts
```python
api = wandb.Api()
artifact = api.artifact("my-project/old-model:v0")
artifact.delete()

# Delete all versions
for version in artifact.versions():
    version.delete()
```

## Best Practices

### Naming Conventions
```
<project>/<artifact-name>:<version-or-alias>

Examples:
- my-project/training-data:latest
- my-project/resnet50-model:v3
- my-project/resnet50-model:production
```

### Use Aliases for Deployment
```python
# Tag production-ready versions
run.log_artifact(artifact, aliases=["latest", "production"])

# In deployment, always use alias
artifact = api.artifact("my-project/model:production")
```

### Track All Pipeline Stages
```python
# Stage 1: Raw data
raw = wandb.Artifact("raw-data", type="dataset")
run.log_artifact(raw)

# Stage 2: Processed data (references raw)
processed_run.use_artifact("raw-data:latest")
processed = wandb.Artifact("processed-data", type="dataset")
processed_run.log_artifact(processed)

# Stage 3: Model (references processed)
train_run.use_artifact("processed-data:latest")
model = wandb.Artifact("model", type="model")
train_run.log_artifact(model)
```

### Common Anti-Patterns

| DON'T | DO |
|-------|-----|
| Use version numbers in names | Let W&B auto-version |
| Skip `use_artifact()` for inputs | Always declare to track lineage |
| Store huge files inline | Use `add_reference()` for external storage |
| Forget aliases | Use "latest", "best", "production" |
| Manual file management | Let artifacts handle versioning |
