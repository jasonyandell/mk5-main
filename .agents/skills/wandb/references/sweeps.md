# W&B Sweeps: Hyperparameter Tuning

Automated hyperparameter search with Bayesian optimization, grid search, and random search.

## Quick Start

```bash
# 1. Create sweep config
# 2. Initialize sweep
wandb sweep --project my-project sweep.yaml

# 3. Run agent(s)
wandb agent <sweep-id>
```

## Sweep Configuration

### YAML Format (sweep.yaml)
```yaml
program: train.py
method: bayes
metric:
  name: val_loss
  goal: minimize

parameters:
  learning_rate:
    distribution: log_uniform_values
    min: 0.0001
    max: 0.1
  batch_size:
    values: [16, 32, 64, 128]
  epochs:
    value: 50
  dropout:
    distribution: uniform
    min: 0.1
    max: 0.5
  optimizer:
    values: ["adam", "sgd", "adamw"]

early_terminate:
  type: hyperband
  min_iter: 3
```

### Python Dictionary Format
```python
sweep_config = {
    "method": "bayes",
    "metric": {
        "name": "val_loss",
        "goal": "minimize"
    },
    "parameters": {
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 0.0001,
            "max": 0.1,
        },
        "batch_size": {
            "values": [16, 32, 64, 128]
        },
        "epochs": {
            "value": 50
        },
    }
}

sweep_id = wandb.sweep(sweep_config, project="my-project")
wandb.agent(sweep_id, function=train)
```

## Search Methods

### Bayesian Optimization (`bayes`)
Best for continuous parameters. Uses Gaussian processes to model objective function.
```yaml
method: bayes
```

### Random Search (`random`)
Simple but effective. Good baseline for comparison.
```yaml
method: random
```

### Grid Search (`grid`)
Exhaustive search over all combinations. Only works with discrete values.
```yaml
method: grid
```

## Parameter Distributions

### Discrete Values
```yaml
# Explicit list
optimizer:
  values: ["adam", "sgd", "adamw"]

# Single fixed value
epochs:
  value: 100
```

### Continuous Distributions
```yaml
# Uniform distribution
dropout:
  distribution: uniform
  min: 0.0
  max: 0.5

# Log-uniform (for learning rates)
learning_rate:
  distribution: log_uniform_values
  min: 0.00001
  max: 0.1

# Normal distribution
weight_decay:
  distribution: normal
  mu: 0.01
  sigma: 0.005

# Quantized (discrete steps)
hidden_dim:
  distribution: q_uniform
  min: 64
  max: 512
  q: 64  # Steps of 64
```

### Integer Parameters
```yaml
num_layers:
  distribution: int_uniform
  min: 1
  max: 10
```

## Metric Configuration

```yaml
metric:
  name: val_accuracy    # Metric name from wandb.log()
  goal: maximize        # "minimize" or "maximize"
```

## Early Termination

### Hyperband
Aggressively terminates underperforming runs.
```yaml
early_terminate:
  type: hyperband
  min_iter: 3       # Minimum epochs before termination
  eta: 3            # Reduction factor
  s: 2              # Bracket configuration
```

### No Early Stopping
```yaml
# Omit early_terminate key entirely
```

## Training Script Setup

```python
import wandb

def train():
    # Initialize run - sweep provides config
    with wandb.init() as run:
        # Access sweep parameters
        lr = wandb.config.learning_rate
        batch_size = wandb.config.batch_size
        epochs = wandb.config.epochs

        model = create_model(...)
        optimizer = create_optimizer(lr=lr)

        for epoch in range(epochs):
            train_loss = train_epoch(model, optimizer, batch_size)
            val_loss, val_acc = validate(model)

            # Log metrics - sweep uses these
            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "epoch": epoch,
            })

# Run as sweep agent
if __name__ == "__main__":
    train()
```

## Running Sweeps

### CLI Commands
```bash
# Create sweep
wandb sweep sweep.yaml

# Run agent (single machine)
wandb agent <entity>/<project>/<sweep-id>

# Run multiple agents (parallel)
wandb agent <sweep-id> &
wandb agent <sweep-id> &
wandb agent <sweep-id> &

# Limit runs per agent
wandb agent --count 10 <sweep-id>
```

### Python API
```python
import wandb

sweep_id = wandb.sweep(sweep_config, project="my-project")

# Run agent
wandb.agent(sweep_id, function=train, count=10)
```

## Sweep Management

### Pause/Resume
```bash
# Pause sweep (stops new runs)
wandb sweep --pause <sweep-id>

# Resume sweep
wandb sweep --resume <sweep-id>
```

### Cancel Sweep
```bash
wandb sweep --cancel <sweep-id>
```

## Nested Parameters

For hierarchical configs:
```yaml
parameters:
  model:
    parameters:
      num_layers:
        values: [2, 3, 4]
      hidden_dim:
        values: [128, 256, 512]
  optimizer:
    parameters:
      name:
        values: ["adam", "sgd"]
      lr:
        distribution: log_uniform_values
        min: 0.0001
        max: 0.01
```

Access in code:
```python
num_layers = wandb.config.model.num_layers
optimizer_name = wandb.config.optimizer.name
```

## Best Practices

### Start with Random Search
```yaml
method: random
# Run 20-50 trials to understand parameter sensitivity
```

### Use Log Scale for Learning Rates
```yaml
learning_rate:
  distribution: log_uniform_values
  min: 0.00001
  max: 0.1
# NOT uniform - LR varies over orders of magnitude
```

### Enable Early Termination
```yaml
early_terminate:
  type: hyperband
  min_iter: 5  # At least 5 epochs before killing
```

### Parallelize When Possible
```bash
# Run on multiple machines/GPUs
for i in {1..4}; do
    wandb agent <sweep-id> &
done
```

### Common Anti-Patterns

| DON'T | DO |
|-------|-----|
| Grid search with continuous params | Use Bayesian or random |
| Uniform for learning rate | Use log_uniform_values |
| Too few random trials | Run at least 20-50 |
| Forget to log metric | Ensure `wandb.log({metric_name: value})` |
| Hardcode hyperparameters | Read from `wandb.config` |

## Debugging Sweeps

```bash
# Check sweep status
wandb sweep --print <sweep-id>

# View runs in browser
# https://wandb.ai/<entity>/<project>/sweeps/<sweep-id>
```

## Integration with Lightning

```python
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer

def train():
    with wandb.init() as run:
        # Create logger from existing run
        wandb_logger = WandbLogger(experiment=run)

        model = LitModel(
            lr=wandb.config.learning_rate,
            hidden_dim=wandb.config.hidden_dim,
        )

        trainer = Trainer(
            logger=wandb_logger,
            max_epochs=wandb.config.epochs,
        )

        trainer.fit(model, datamodule)
```
