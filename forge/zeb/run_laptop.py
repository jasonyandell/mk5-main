"""Laptop overnight training for Zeb self-play."""
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from pathlib import Path
import argparse

from .types import TrainingConfig
from .model import ZebModel, get_model_config
from .module import ZebLightningModule
from .self_play import play_games_batched, trajectories_to_batch
from .evaluate import evaluate_vs_random, evaluate_vs_heuristic


class SelfPlayDataModule(L.LightningDataModule):
    """Data module that generates training data via self-play."""

    def __init__(
        self,
        model: ZebModel,
        games_per_epoch: int = 500,
        batch_size: int = 128,
        temperature: float = 1.0,
        device: str = 'cuda',
    ):
        super().__init__()
        self.model = model
        self.games_per_epoch = games_per_epoch
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.epoch = 0

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        """Generate new games each epoch."""
        # Play games with current model
        trajectories = play_games_batched(
            self.model,
            n_games=self.games_per_epoch,
            temperature=self.temperature,
            device=self.device,
            base_seed=self.epoch * self.games_per_epoch,
        )
        self.epoch += 1

        # Convert to batch
        batch_data = trajectories_to_batch(trajectories)

        # Create simple dataset
        dataset = torch.utils.data.TensorDataset(*batch_data)

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )


def main():
    parser = argparse.ArgumentParser(description='Zeb laptop training')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--games-per-epoch', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--model-size', choices=['small', 'medium'], default='small')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='forge/zeb/runs')
    args = parser.parse_args()

    L.seed_everything(args.seed)

    # Create model
    config = get_model_config(args.model_size)
    module = ZebLightningModule(
        **config,
        lr=args.lr,
    )

    # Create data module
    data = SelfPlayDataModule(
        model=module.model,
        games_per_epoch=args.games_per_epoch,
        batch_size=args.batch_size,
        temperature=args.temperature,
    )

    # Callbacks
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / 'checkpoints',
            save_top_k=3,
            monitor='train/loss',
            mode='min',
            save_last=True,
        ),
    ]

    # Trainer (laptop settings)
    trainer = L.Trainer(
        max_epochs=args.epochs,
        callbacks=callbacks,
        default_root_dir=str(output_dir),
        accelerator='auto',
        devices=1,
        precision='16-mixed',  # Save memory
        gradient_clip_val=1.0,
        log_every_n_steps=10,
    )

    # Train
    trainer.fit(module, data)

    # Evaluate
    print("\n=== Evaluation ===")
    vs_random = evaluate_vs_random(module.model, n_games=100)
    print(f"vs Random: {vs_random['team0_win_rate']:.1%} win rate")

    vs_heuristic = evaluate_vs_heuristic(module.model, n_games=100)
    print(f"vs Heuristic: {vs_heuristic['team0_win_rate']:.1%} win rate")


if __name__ == '__main__':
    main()
