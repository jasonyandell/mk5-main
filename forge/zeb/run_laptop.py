"""Laptop overnight training for Zeb self-play."""
import argparse
from datetime import datetime
from pathlib import Path

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from .model import ZebModel, get_model_config
from .module import ZebLightningModule
from .self_play import play_games_batched, play_games_vs_heuristic, trajectories_to_batch
from .evaluate import evaluate_vs_random, evaluate_vs_heuristic


class EvaluationCallback(Callback):
    """Callback to evaluate model against baselines periodically."""

    def __init__(self, eval_every_n_epochs: int = 5, n_games: int = 100, device: str = 'cuda'):
        super().__init__()
        self.eval_every_n_epochs = eval_every_n_epochs
        self.n_games = n_games
        self.device = device

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: ZebLightningModule):
        epoch = trainer.current_epoch
        if epoch % self.eval_every_n_epochs != 0:
            return

        # Evaluate vs baselines
        model = pl_module.model
        model.eval()

        vs_random = evaluate_vs_random(model, n_games=self.n_games, device=self.device)
        vs_heuristic = evaluate_vs_heuristic(model, n_games=self.n_games, device=self.device)

        # Log metrics
        pl_module.log('eval/vs_random_win_rate', vs_random['team0_win_rate'], prog_bar=True)
        pl_module.log('eval/vs_heuristic_win_rate', vs_heuristic['team0_win_rate'], prog_bar=True)

        print(f"\n[Epoch {epoch}] vs Random: {vs_random['team0_win_rate']:.1%}, "
              f"vs Heuristic: {vs_heuristic['team0_win_rate']:.1%}")


class SelfPlayDataModule(L.LightningDataModule):
    """Data module that generates training data via self-play or vs heuristic."""

    def __init__(
        self,
        model: ZebModel,
        games_per_epoch: int = 500,
        batch_size: int = 128,
        temperature: float = 1.0,
        device: str = 'cuda',
        vs_heuristic: bool = False,
    ):
        super().__init__()
        self.model = model
        self.games_per_epoch = games_per_epoch
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.vs_heuristic = vs_heuristic
        self.epoch = 0

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        """Generate new games each epoch."""
        # Play games with current model
        if self.vs_heuristic:
            trajectories = play_games_vs_heuristic(
                self.model,
                n_games=self.games_per_epoch,
                temperature=self.temperature,
                device=self.device,
                base_seed=self.epoch * self.games_per_epoch,
            )
        else:
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
    parser.add_argument('--eval-every', type=int, default=1, help='Evaluate every N epochs')
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--wandb-project', type=str, default='zeb-training')
    parser.add_argument('--vs-heuristic', action='store_true',
                        help='Train against fixed heuristic instead of self-play')
    args = parser.parse_args()

    L.seed_everything(args.seed)

    # Create model
    config = get_model_config(args.model_size)
    module = ZebLightningModule(
        **config,
        lr=args.lr,
    )

    # Compute model size for naming
    total_params = sum(p.numel() for p in module.parameters())
    if total_params >= 1_000_000:
        model_size_str = f"{total_params / 1_000_000:.1f}M"
    else:
        model_size_str = f"{total_params // 1000}k"

    # Create data module
    data = SelfPlayDataModule(
        model=module.model,
        games_per_epoch=args.games_per_epoch,
        batch_size=args.batch_size,
        temperature=args.temperature,
        vs_heuristic=args.vs_heuristic,
    )

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Loggers
    loggers = [CSVLogger(str(output_dir), name='zeb')]
    if args.wandb:
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        mode_tag = 'vs-heuristic' if args.vs_heuristic else 'self-play'
        run_name = f"zeb-{mode_tag}-{timestamp}-{model_size_str}"
        loggers.append(WandbLogger(
            project=args.wandb_project,
            name=run_name,
            tags=['laptop', mode_tag, args.model_size],
            save_dir=str(output_dir),
            log_model=False,  # Don't upload checkpoints
            config={
                'total_params': total_params,
                'model_size': args.model_size,
                'model_size_str': model_size_str,
                'epochs': args.epochs,
                'games_per_epoch': args.games_per_epoch,
                'batch_size': args.batch_size,
                'temperature': args.temperature,
                'lr': args.lr,
                'seed': args.seed,
                'vs_heuristic': args.vs_heuristic,
            },
        ))

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / 'checkpoints',
            save_top_k=3,
            monitor='train/loss',
            mode='min',
            save_last=True,
        ),
        EvaluationCallback(
            eval_every_n_epochs=args.eval_every,
            n_games=100,
        ),
    ]

    # Trainer (laptop settings)
    trainer = L.Trainer(
        max_epochs=args.epochs,
        callbacks=callbacks,
        logger=loggers,
        default_root_dir=str(output_dir),
        accelerator='auto',
        devices=1,
        precision='16-mixed',  # Save memory
        gradient_clip_val=1.0,
        log_every_n_steps=10,
    )

    # Train
    trainer.fit(module, data)

    # Final evaluation
    print("\n=== Final Evaluation ===")
    vs_random = evaluate_vs_random(module.model, n_games=100)
    print(f"vs Random: {vs_random['team0_win_rate']:.1%} win rate")

    vs_heuristic = evaluate_vs_heuristic(module.model, n_games=100)
    print(f"vs Heuristic: {vs_heuristic['team0_win_rate']:.1%} win rate")

    # Log final metrics to W&B
    if args.wandb:
        import wandb
        wandb.log({
            'final/vs_random_win_rate': vs_random['team0_win_rate'],
            'final/vs_heuristic_win_rate': vs_heuristic['team0_win_rate'],
        })
        wandb.finish()


if __name__ == '__main__':
    main()
