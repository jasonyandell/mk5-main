# PyTorch Lightning Callbacks Reference

Callbacks inject custom logic at any point in the training loop without cluttering model code.

## Built-in Callbacks

### ModelCheckpoint

Save model checkpoints based on monitored metrics.

```python
from lightning.pytorch.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints/",
    filename="{epoch:02d}-{val_loss:.4f}",
    save_top_k=3,           # Keep top 3 checkpoints
    monitor="val_loss",     # Metric to monitor
    mode="min",             # "min" or "max"
    save_last=True,         # Always save last checkpoint
    save_weights_only=False, # Save full state or just weights
    every_n_epochs=1,
    every_n_train_steps=None,
    train_time_interval=None,  # timedelta for time-based saving
    save_on_train_epoch_end=False,
)
```

### EarlyStopping

Stop training when metric plateaus.

```python
from lightning.pytorch.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor="val_loss",
    min_delta=0.001,    # Minimum change to qualify as improvement
    patience=5,         # Epochs to wait before stopping
    verbose=True,
    mode="min",
    strict=True,        # Crash if metric not found
    check_finite=True,  # Stop on NaN/inf
    stopping_threshold=None,  # Stop immediately if metric reaches this
    divergence_threshold=None,  # Stop if metric goes above this
)
```

### LearningRateMonitor

Log learning rate to logger.

```python
from lightning.pytorch.callbacks import LearningRateMonitor

lr_monitor = LearningRateMonitor(
    logging_interval="step",  # "step" or "epoch"
    log_momentum=True,
    log_weight_decay=False,
)
```

### TQDMProgressBar / RichProgressBar

Custom progress display.

```python
from lightning.pytorch.callbacks import TQDMProgressBar, RichProgressBar

# TQDM (default)
progress = TQDMProgressBar(refresh_rate=10)

# Rich (prettier)
progress = RichProgressBar()
```

### DeviceStatsMonitor

Log GPU utilization.

```python
from lightning.pytorch.callbacks import DeviceStatsMonitor

device_stats = DeviceStatsMonitor()
```

### BatchSizeFinder

Find optimal batch size before OOM.

```python
from lightning.pytorch.callbacks import BatchSizeFinder

batch_finder = BatchSizeFinder(
    mode="power",       # "power" (2^n) or "binsearch"
    steps_per_trial=3,
    init_val=2,
    max_trials=25,
    batch_arg_name="batch_size",
)
```

### LearningRateFinder

Find optimal learning rate.

```python
from lightning.pytorch.callbacks import LearningRateFinder

lr_finder = LearningRateFinder(
    min_lr=1e-8,
    max_lr=1,
    num_training_steps=100,
    mode="exponential",
    early_stop_threshold=4.0,
)
```

### StochasticWeightAveraging

Improve generalization with SWA.

```python
from lightning.pytorch.callbacks import StochasticWeightAveraging

swa = StochasticWeightAveraging(
    swa_lrs=1e-2,           # LR for SWA phase
    swa_epoch_start=0.8,    # Start SWA at 80% of training
    annealing_epochs=10,
    annealing_strategy="cos",
)
```

### Timer

Track and limit training time.

```python
from lightning.pytorch.callbacks import Timer
from datetime import timedelta

timer = Timer(
    duration=timedelta(hours=12),  # Stop after 12 hours
    interval="step",               # Check every step
    verbose=True,
)

# Access elapsed time
elapsed = timer.time_elapsed("train")
remaining = timer.time_remaining("train")
```

### GradientAccumulationScheduler

Dynamic gradient accumulation.

```python
from lightning.pytorch.callbacks import GradientAccumulationScheduler

# Accumulate 8 batches until epoch 5, then 4 batches
accumulator = GradientAccumulationScheduler(
    scheduling={0: 8, 5: 4}
)
```

## Writing Custom Callbacks

```python
from lightning.pytorch.callbacks import Callback

class MyCallback(Callback):
    def __init__(self, threshold=0.1):
        super().__init__()
        self.threshold = threshold

    def on_train_start(self, trainer, pl_module):
        print("Training started!")

    def on_train_epoch_end(self, trainer, pl_module):
        # Access logged metrics
        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is not None and val_loss < self.threshold:
            trainer.should_stop = True

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Access batch outputs
        loss = outputs["loss"]

    def on_validation_epoch_end(self, trainer, pl_module):
        pass

    # State persistence for resuming
    def state_dict(self):
        return {"threshold": self.threshold}

    def load_state_dict(self, state_dict):
        self.threshold = state_dict["threshold"]
```

## All Available Hooks

### Setup/Teardown
- `setup(trainer, pl_module, stage)` - Called at start of fit/validate/test/predict
- `teardown(trainer, pl_module, stage)` - Called at end of fit/validate/test/predict

### Training
- `on_train_start(trainer, pl_module)`
- `on_train_end(trainer, pl_module)`
- `on_train_epoch_start(trainer, pl_module)`
- `on_train_epoch_end(trainer, pl_module)`
- `on_train_batch_start(trainer, pl_module, batch, batch_idx)`
- `on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)`

### Validation
- `on_validation_start(trainer, pl_module)`
- `on_validation_end(trainer, pl_module)`
- `on_validation_epoch_start(trainer, pl_module)`
- `on_validation_epoch_end(trainer, pl_module)`
- `on_validation_batch_start(trainer, pl_module, batch, batch_idx, dataloader_idx)`
- `on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)`

### Testing
- `on_test_start/end/epoch_start/epoch_end/batch_start/batch_end` (same pattern)

### Prediction
- `on_predict_start/end/epoch_start/epoch_end/batch_start/batch_end` (same pattern)

### Optimization
- `on_before_optimizer_step(trainer, pl_module, optimizer)`
- `on_before_zero_grad(trainer, pl_module, optimizer)`

### Checkpointing
- `on_save_checkpoint(trainer, pl_module, checkpoint)`
- `on_load_checkpoint(trainer, pl_module, checkpoint)`

### Exception Handling
- `on_exception(trainer, pl_module, exception)`

## Callback Order

Multiple callbacks execute in registration order. Don't rely on specific ordering between callbacks - keep them independent.

```python
trainer = L.Trainer(callbacks=[
    callback_1,  # Runs first
    callback_2,  # Runs second
    callback_3,  # Runs third
])
```
