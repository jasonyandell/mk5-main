#!/usr/bin/env python3
"""Train DominoTransformer with Lightning."""
import argparse

import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from forge.ml.data import DominoDataModule
from forge.ml.module import DominoLightningModule

# RichProgressBar is optional (requires `rich` package)
try:
    import rich  # noqa: F401 - Check if rich is available
    from lightning.pytorch.callbacks import RichProgressBar
    HAS_RICH = True
except (ImportError, ModuleNotFoundError):
    HAS_RICH = False
    RichProgressBar = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/tokenized', help='Path to tokenized data')
    parser.add_argument('--run-dir', default='runs', help='Output directory')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=3e-4)

    # Architecture hyperparameters
    parser.add_argument('--embed-dim', type=int, default=64, help='Transformer embedding dimension')
    parser.add_argument('--n-heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--n-layers', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--ff-dim', type=int, default=128, help='Feed-forward dimension')
    parser.add_argument('--value-weight', type=float, default=0.5, help='Weight for value head loss')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=None, help='Dataloader workers (default: auto-detect)')
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--wandb-group', type=str, default=None, help='Wandb group name for organizing runs')
    parser.add_argument('--fast-dev-run', action='store_true', help='Quick sanity check')
    parser.add_argument('--limit-batches', type=int, default=None, help='Limit train/val batches (for quick testing)')
    parser.add_argument('--compile', action=argparse.BooleanOptionalAction, default=True, help='Use torch.compile')
    parser.add_argument('--precision', default='16-mixed', help='Training precision (32, 16-mixed, bf16-mixed for A100/H100)')
    parser.add_argument('--strategy', default='auto', help='DDP strategy (auto, ddp, fsdp, deepspeed_stage_2)')
    parser.add_argument('--deterministic', action=argparse.BooleanOptionalAction, default=False, help='Deterministic mode (slower)')
    parser.add_argument('--log-every-n-steps', type=int, default=25, help='Log metrics every N steps (~5s with default batch)')
    args = parser.parse_args()

    # Reproducibility
    L.seed_everything(args.seed, workers=True)

    import os
    import torch

    # Performance optimizations for Tensor Cores (A100/H100)
    torch.set_float32_matmul_precision("high")
    if not args.deterministic:
        torch.backends.cudnn.benchmark = True  # Auto-tune conv algorithms

    # Auto-detect num_workers: 4 per GPU, capped at CPU count
    if args.num_workers is None:
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        num_workers = min(4 * num_gpus, os.cpu_count() or 8)
    else:
        num_workers = args.num_workers

    # Model and data
    model = DominoLightningModule(
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ff_dim=args.ff_dim,
        lr=args.lr,
        value_weight=args.value_weight,
    )

    # torch.compile for PyTorch 2.0+ JIT optimization
    # Compile the inner model, not the LightningModule (required for DDP)
    if args.compile:
        model.model = torch.compile(model.model)

    data = DominoDataModule(
        args.data,
        batch_size=args.batch_size,
        num_workers=num_workers,
    )

    # Compute model size for wandb naming/config
    total_params = sum(p.numel() for p in model.parameters())
    if total_params >= 1_000_000:
        model_size = f"{total_params / 1_000_000:.1f}M"
    else:
        model_size = f"{total_params // 1000}k"

    # Loggers
    loggers = [CSVLogger(args.run_dir, name='domino')]
    if args.wandb:
        # Build run name with model size and architecture
        run_name = f"train-{model_size}-{args.n_layers}L-{args.n_heads}H-d{args.embed_dim}"

        # Build tags
        tags = ["train", model_size]
        if args.wandb_group:
            group_root = args.wandb_group.split("/")[0]
            tags.append(group_root)

        loggers.append(WandbLogger(
            project='crystal-forge',
            name=run_name,
            group=args.wandb_group,
            tags=tags,
            save_dir=args.run_dir,
            log_model=False,  # Don't upload checkpoints (too large)
            config={
                "total_params": total_params,
                "model_size": model_size,
                "embed_dim": args.embed_dim,
                "n_heads": args.n_heads,
                "n_layers": args.n_layers,
                "ff_dim": args.ff_dim,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "precision": args.precision,
                "value_weight": args.value_weight,
            },
        ))

    # Callbacks (following best practices)
    # Note: dirpath=None lets Trainer use logger's log_dir/checkpoints
    callbacks = [
        ModelCheckpoint(
            dirpath=None,  # Use logger's directory (runs/domino/version_X/checkpoints)
            monitor='val/q_gap',
            mode='min',
            save_top_k=1,
            save_last=True,
            filename='{epoch}-{val_q_gap:.2f}',
        ),
        EarlyStopping(
            monitor='val/q_gap',
            mode='min',
            patience=5,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval='epoch'),
    ]
    if HAS_RICH:
        callbacks.append(RichProgressBar())

    # Trainer with best practices
    trainer = L.Trainer(
        max_epochs=args.epochs,
        logger=loggers,
        callbacks=callbacks,
        default_root_dir=args.run_dir,
        log_every_n_steps=args.log_every_n_steps,

        # Reproducibility (off by default for performance)
        deterministic=args.deterministic,

        # Training stability
        gradient_clip_val=1.0,
        gradient_clip_algorithm='norm',

        # Development helpers
        fast_dev_run=args.fast_dev_run,
        limit_train_batches=args.limit_batches,
        limit_val_batches=args.limit_batches,

        # Hardware - auto-detect accelerator and devices
        accelerator='auto',
        devices='auto',
        strategy=args.strategy,

        # Mixed precision (bf16 for A100/H100, 16-mixed for older)
        precision=args.precision,
    )

    trainer.fit(model, data)

    # Print best checkpoint path
    print(f'Best checkpoint: {trainer.checkpoint_callback.best_model_path}')


if __name__ == '__main__':
    main()
