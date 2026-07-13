# PyTorch Lightning Advanced Patterns

Production patterns, CLI, hyperparameter tuning, and optimization techniques.

## Lightning CLI

Auto-generate CLI from LightningModule and LightningDataModule.

```python
# train.py
from lightning.pytorch.cli import LightningCLI

def cli_main():
    cli = LightningCLI(MyModel, MyDataModule)

if __name__ == "__main__":
    cli_main()
```

```bash
# Run with CLI
python train.py fit --model.lr=1e-3 --data.batch_size=64 --trainer.max_epochs=100

# Generate config file
python train.py fit --print_config > config.yaml

# Run from config
python train.py fit --config config.yaml

# Override config values
python train.py fit --config config.yaml --trainer.max_epochs=200
```

### Config File Structure

```yaml
# config.yaml
seed_everything: 42
trainer:
  accelerator: gpu
  devices: 4
  max_epochs: 100
  precision: 16-mixed
  callbacks:
    - class_path: lightning.pytorch.callbacks.EarlyStopping
      init_args:
        monitor: val_loss
        patience: 5
model:
  hidden_dim: 256
  lr: 1e-3
data:
  batch_size: 32
  num_workers: 8
```

## Hyperparameter Tuning

### With Optuna

```python
import optuna
from optuna.integration import PyTorchLightningPruningCallback

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    hidden_dim = trial.suggest_int("hidden_dim", 64, 512)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)

    model = LitModel(lr=lr, hidden_dim=hidden_dim, dropout=dropout)
    dm = MyDataModule()

    trainer = L.Trainer(
        max_epochs=50,
        callbacks=[
            PyTorchLightningPruningCallback(trial, monitor="val_loss"),
            EarlyStopping(monitor="val_loss", patience=3),
        ],
        enable_progress_bar=False,
    )

    trainer.fit(model, datamodule=dm)
    return trainer.callback_metrics["val_loss"].item()

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

print(f"Best trial: {study.best_trial.params}")
```

### With Ray Tune

```python
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback

def train_fn(config):
    model = LitModel(lr=config["lr"], hidden_dim=config["hidden_dim"])
    dm = MyDataModule(batch_size=config["batch_size"])

    trainer = L.Trainer(
        max_epochs=50,
        callbacks=[TuneReportCallback({"loss": "val_loss"}, on="validation_end")],
    )
    trainer.fit(model, datamodule=dm)

analysis = tune.run(
    train_fn,
    config={
        "lr": tune.loguniform(1e-5, 1e-2),
        "hidden_dim": tune.choice([64, 128, 256, 512]),
        "batch_size": tune.choice([16, 32, 64]),
    },
    num_samples=50,
    resources_per_trial={"gpu": 1},
)
```

## Production Patterns

### Model Export

```python
# TorchScript
model = LitModel.load_from_checkpoint("best.ckpt")
script = model.to_torchscript()
torch.jit.save(script, "model.pt")

# ONNX
model.to_onnx("model.onnx", input_sample=torch.randn(1, 784))
```

### Inference Optimization

```python
class LitModel(L.LightningModule):
    def predict_step(self, batch, batch_idx):
        x, _ = batch
        with torch.inference_mode():
            return self(x)

# Batch inference
trainer = L.Trainer(accelerator="gpu", devices=1)
predictions = trainer.predict(model, dataloaders=test_loader)
```

### Compile with torch.compile

```python
class LitModel(L.LightningModule):
    def configure_model(self):
        # Compile after model setup
        self.model = torch.compile(self.model)
```

## Memory Optimization

### Activation Checkpointing

Trade compute for memory by recomputing activations during backward.

```python
from torch.utils.checkpoint import checkpoint

class LitModel(L.LightningModule):
    def forward(self, x):
        # Checkpoint expensive layers
        x = checkpoint(self.encoder, x, use_reentrant=False)
        return self.decoder(x)
```

### Gradient Checkpointing for Transformers

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("bert-base-uncased")
model.gradient_checkpointing_enable()
```

### Memory-Efficient Attention

```python
# Use Flash Attention (auto-enabled in PyTorch 2.0+)
import torch.nn.functional as F

attn_output = F.scaled_dot_product_attention(
    query, key, value,
    is_causal=True,  # For autoregressive models
)
```

## Fault Tolerance

### Auto-Resume from Failure

```python
from lightning.pytorch.callbacks import OnExceptionCheckpoint

trainer = L.Trainer(
    callbacks=[OnExceptionCheckpoint("checkpoints/")],
)

# Resume after crash
trainer.fit(model, ckpt_path="checkpoints/on_exception.ckpt")
```

### Elastic Training

```python
# Handles node failures in multi-node training
trainer = L.Trainer(
    plugins=[TorchElasticEnvironment()],
)
```

## Custom Training Loops

### Manual Optimization

For GANs, RL, or complex optimization patterns:

```python
class GAN(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.automatic_optimization = False  # Disable auto optimization
        self.generator = Generator()
        self.discriminator = Discriminator()

    def training_step(self, batch, batch_idx):
        g_opt, d_opt = self.optimizers()

        # Train discriminator
        d_opt.zero_grad()
        d_loss = self.discriminator_loss(batch)
        self.manual_backward(d_loss)
        d_opt.step()

        # Train generator
        g_opt.zero_grad()
        g_loss = self.generator_loss(batch)
        self.manual_backward(g_loss)
        g_opt.step()

        self.log_dict({"g_loss": g_loss, "d_loss": d_loss})

    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.generator.parameters(), lr=1e-4)
        d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4)
        return [g_opt, d_opt], []
```

### Multiple Optimizers with Schedulers

```python
def configure_optimizers(self):
    opt1 = torch.optim.Adam(self.encoder.parameters(), lr=1e-3)
    opt2 = torch.optim.Adam(self.decoder.parameters(), lr=1e-4)

    sched1 = torch.optim.lr_scheduler.StepLR(opt1, step_size=10)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=100)

    return (
        {"optimizer": opt1, "lr_scheduler": {"scheduler": sched1, "interval": "epoch"}},
        {"optimizer": opt2, "lr_scheduler": {"scheduler": sched2, "interval": "step"}},
    )
```

## Testing Patterns

### Unit Testing LightningModules

```python
import pytest

def test_model_forward():
    model = LitModel()
    x = torch.randn(4, 784)
    out = model(x)
    assert out.shape == (4, 10)

def test_training_step():
    model = LitModel()
    batch = (torch.randn(4, 784), torch.randint(0, 10, (4,)))
    loss = model.training_step(batch, 0)
    assert loss.ndim == 0  # Scalar loss

def test_full_training():
    model = LitModel()
    dm = MNISTDataModule()
    trainer = L.Trainer(fast_dev_run=True)
    trainer.fit(model, datamodule=dm)
```

### Integration Testing

```python
def test_overfitting():
    """Model should overfit small dataset."""
    model = LitModel()
    dm = MNISTDataModule(batch_size=8)

    trainer = L.Trainer(
        max_epochs=50,
        overfit_batches=1,
        enable_progress_bar=False,
    )
    trainer.fit(model, datamodule=dm)

    assert trainer.callback_metrics["train_loss"] < 0.01

def test_checkpointing(tmp_path):
    """Checkpoint should restore correctly."""
    model = LitModel()
    trainer = L.Trainer(max_epochs=2, default_root_dir=tmp_path)
    trainer.fit(model, train_dataloaders=...)

    ckpt_path = list(tmp_path.glob("**/*.ckpt"))[0]
    loaded = LitModel.load_from_checkpoint(ckpt_path)

    assert loaded.hparams == model.hparams
```

## Profiling

```python
# Simple profiler (text output)
trainer = L.Trainer(profiler="simple")

# Advanced profiler (detailed breakdown)
trainer = L.Trainer(profiler="advanced")

# PyTorch profiler (with trace)
from lightning.pytorch.profilers import PyTorchProfiler

profiler = PyTorchProfiler(
    dirpath="profiler_logs",
    filename="perf_logs",
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),
)
trainer = L.Trainer(profiler=profiler)
```

## Common Integrations

### Hugging Face Transformers

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class LitTransformer(L.LightningModule):
    def __init__(self, model_name="bert-base-uncased", num_labels=2, lr=2e-5):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        self.log("train_loss", outputs.loss)
        return outputs.loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
```

### Weights & Biases

```python
from lightning.pytorch.loggers import WandbLogger

wandb_logger = WandbLogger(
    project="my-project",
    name="run-name",
    log_model="all",  # Log checkpoints
)

trainer = L.Trainer(logger=wandb_logger)

# Log additional artifacts
wandb_logger.experiment.log({"confusion_matrix": wandb.plot.confusion_matrix(...)})
```
