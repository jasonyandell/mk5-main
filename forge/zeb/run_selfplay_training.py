"""True AlphaZero self-play training for Zeb.

Uses the model's value head for MCTS leaf evaluation instead of the oracle.
This enables the model to bootstrap from itself through self-play.

Key difference from oracle training:
- Oracle training: Perfect oracle evaluates leaves -> model learns to mimic
- Self-play training: Model evaluates its own leaves -> bootstrap improvement

The model's value head is now IN the MCTS loop, so it must learn to provide
useful evaluations for MCTS to work.
"""
import argparse
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from .gpu_training_pipeline import create_selfplay_pipeline, GPUTrainingPipeline
from .model import ZebModel, get_model_config
from .run_mcts_training import (
    evaluate_vs_random,
    DEFAULT_CHECKPOINT_DIR,
)


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: str = 'cuda',
) -> tuple[ZebModel, dict]:
    """Load model from checkpoint.

    Returns:
        model: Loaded ZebModel
        metadata: Checkpoint metadata (epoch, config, etc.)
    """
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Get model config - handle different checkpoint formats
    if 'model_config' in ckpt:
        model_config = ckpt['model_config']
    elif 'config' in ckpt and 'model_config' in ckpt['config']:
        # Nested format from run_mcts_training
        model_config = ckpt['config']['model_config']
    elif 'config' in ckpt:
        # Try to extract model-specific keys
        config = ckpt['config']
        model_config = {k: v for k, v in config.items()
                       if k in ('embed_dim', 'n_heads', 'n_layers', 'ff_dim', 'dropout', 'max_tokens')}
    else:
        raise ValueError("Checkpoint missing model config")

    # Create and load model
    model = ZebModel(**model_config)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)

    # Compute total games from epoch and games_per_epoch if not stored
    total_games = ckpt.get('total_games', 0)
    if total_games == 0 and 'config' in ckpt:
        epoch = ckpt.get('epoch', 0)
        games_per_epoch = ckpt['config'].get('games_per_epoch', 0)
        total_games = (epoch + 1) * games_per_epoch

    metadata = {
        'epoch': ckpt.get('epoch', 0),
        'model_config': model_config,
        'training_config': ckpt.get('config', {}),
        'total_games': total_games,
        'wandb_run_id': ckpt.get('wandb_run_id'),
    }

    return model, metadata


def main():
    parser = argparse.ArgumentParser(description='Self-play Zeb training (AlphaZero style)')

    # Source checkpoint (required for self-play from plateau)
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='Checkpoint to start from (e.g., checkpoints/zeb-medium-epoch0102.pt)')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--games-per-epoch', type=int, default=512,
                        help='Games per epoch (smaller = fresher policy)')
    parser.add_argument('--n-simulations', type=int, default=100,
                        help='MCTS simulations (more helps with weak evaluator)')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (lower for fine-tuning)')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--eval-every', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda')

    # GPU MCTS parameters
    parser.add_argument('--n-parallel-games', type=int, default=512,
                        help='Concurrent games (can be lower for self-play)')
    parser.add_argument('--max-mcts-nodes', type=int, default=256,
                        help='Max nodes per MCTS tree')

    # W&B parameters
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--wandb-project', type=str, default='zeb-mcts')
    parser.add_argument('--run-name', type=str, default=None)

    # Output
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_CHECKPOINT_DIR,
                        help='Directory for output checkpoints')
    parser.add_argument('--save-every', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--keep-checkpoints', type=int, default=0,
                        help='Keep only last N checkpoints (0 = keep all)')

    args = parser.parse_args()

    # Load model from checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    model, metadata = load_model_from_checkpoint(args.checkpoint, args.device)
    model_config = metadata['model_config']

    total_params = sum(p.numel() for p in model.parameters())
    start_epoch = metadata['epoch'] + 1  # Continue from next epoch
    print(f"  Model: {total_params:,} parameters")
    print(f"  From epoch: {metadata['epoch']} (will start at {start_epoch})")
    print(f"  Prior games: {metadata['total_games']:,}")

    # Initial evaluation
    print("\nInitial evaluation...")
    initial_win_rate = evaluate_vs_random(model, n_games=100, device=args.device)
    print(f"  vs Random: {initial_win_rate:.1%}")

    # Initialize W&B (resume if checkpoint has run ID)
    use_wandb = args.wandb and WANDB_AVAILABLE
    wandb_run_id = metadata.get('wandb_run_id')
    if use_wandb:
        if wandb_run_id:
            # Resume existing W&B run
            wandb.init(
                project=args.wandb_project,
                id=wandb_run_id,
                resume='must',
            )
            print(f"W&B run resumed: {wandb.run.url}")
        else:
            # New W&B run
            timestamp = datetime.now().strftime("%Y%m%d%H%M")
            run_name = args.run_name or f"selfplay-{timestamp}"
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                tags=['selfplay', 'alphazero', 'zeb'],
                config={
                    'mode': 'selfplay',
                    'source_checkpoint': str(args.checkpoint),
                    'source_epoch': metadata['epoch'],
                    'initial_win_rate': initial_win_rate,
                    'epochs': args.epochs,
                    'games_per_epoch': args.games_per_epoch,
                    'n_simulations': args.n_simulations,
                    'batch_size': args.batch_size,
                    'lr': args.lr,
                    'temperature': args.temperature,
                    'n_parallel_games': args.n_parallel_games,
                    'max_mcts_nodes': args.max_mcts_nodes,
                    'model_config': model_config,
                },
            )
            print(f"W&B run: {wandb.run.url}")

    print("\n=== Self-Play AlphaZero Training ===")
    print(f"Epochs: {start_epoch} to {start_epoch + args.epochs - 1} ({args.epochs} total)")
    print(f"Games/epoch: {args.games_per_epoch}")
    print(f"MCTS simulations: {args.n_simulations}")
    print(f"Parallel games: {args.n_parallel_games}")
    print(f"Max MCTS nodes: {args.max_mcts_nodes}")
    print(f"Learning rate: {args.lr}")
    print()

    # Create self-play pipeline with the loaded model
    print("Creating self-play pipeline...")
    t0 = time.time()
    pipeline = create_selfplay_pipeline(
        model=model,
        device=args.device,
        n_parallel_games=args.n_parallel_games,
        n_simulations=args.n_simulations,
        max_mcts_nodes=args.max_mcts_nodes,
        temperature=args.temperature,
    )
    print(f"Pipeline created in {time.time() - t0:.1f}s")
    print()

    # Optimizer (fresh optimizer for self-play phase)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Output directory for self-play checkpoints
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop - continue from checkpoint's epoch
    total_games = metadata['total_games']

    for epoch in range(start_epoch, start_epoch + args.epochs):
        t0 = time.time()

        # Generate games with current model evaluating leaves
        model.eval()  # Eval mode for game generation
        examples = pipeline.generate_games_gpu(n_games=args.games_per_epoch)
        gen_time = time.time() - t0

        total_games += args.games_per_epoch
        n_examples = examples.n_examples
        games_per_sec = args.games_per_epoch / gen_time

        # Train on generated examples
        t1 = time.time()
        model.train()  # Training mode
        metrics = pipeline.train_epoch_gpu(
            model=model,
            optimizer=optimizer,
            examples=examples,
            batch_size=args.batch_size,
        )
        train_time = time.time() - t1

        # Update pipeline's model reference (model improved)
        pipeline.set_model(model)

        # Epoch summary
        epoch_str = f"Epoch {epoch:3d}: policy_loss={metrics['policy_loss']:.4f}, value_loss={metrics['value_loss']:.4f}"
        epoch_str += f" (gen={gen_time:.1f}s, train={train_time:.2f}s, {games_per_sec:.2f} games/s)"
        epoch_str += f" [model queries: {pipeline.total_model_queries:,}]"
        print(epoch_str)

        # Log to W&B
        if use_wandb:
            wandb.log({
                'epoch': epoch,
                'train/policy_loss': metrics['policy_loss'],
                'train/value_loss': metrics['value_loss'],
                'train/total_loss': metrics['policy_loss'] + metrics['value_loss'],
                'perf/gen_time_s': gen_time,
                'perf/train_time_s': train_time,
                'perf/games_per_sec': games_per_sec,
                'perf/examples': n_examples,
                'stats/total_games': total_games,
                'stats/total_model_queries': pipeline.total_model_queries,
            })

        # Periodic evaluation
        if (epoch + 1) % args.eval_every == 0:
            model.eval()
            win_rate = evaluate_vs_random(model, n_games=100, device=args.device)
            print(f"         -> vs Random: {win_rate:.1%}")
            if use_wandb:
                wandb.log({'eval/vs_random_win_rate': win_rate, 'epoch': epoch})

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = args.output_dir / f"selfplay-epoch{epoch:04d}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'model_config': model_config,
                'training_config': {
                    'mode': 'selfplay',
                    'source_checkpoint': str(args.checkpoint),
                    'games_per_epoch': args.games_per_epoch,
                    'n_simulations': args.n_simulations,
                    'lr': args.lr,
                },
                'total_games': total_games,
                'wandb_run_id': wandb.run.id if use_wandb else None,
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path.name}")

            # Cleanup old checkpoints if requested (sort by mtime, keep newest)
            if args.keep_checkpoints > 0:
                all_ckpts = sorted(
                    args.output_dir.glob("selfplay-epoch*.pt"),
                    key=lambda p: p.stat().st_mtime,
                )
                if len(all_ckpts) > args.keep_checkpoints:
                    for old_ckpt in all_ckpts[:-args.keep_checkpoints]:
                        old_ckpt.unlink()
                        print(f"  Removed old checkpoint: {old_ckpt.name}")

    # Final evaluation
    print("\n=== Final Evaluation ===")
    model.eval()
    win_rate = evaluate_vs_random(model, n_games=200, device=args.device)
    print(f"vs Random: {win_rate:.1%}")
    print(f"Improvement: {initial_win_rate:.1%} -> {win_rate:.1%} ({win_rate - initial_win_rate:+.1%})")

    print(f"\n=== Cumulative Stats ===")
    print(f"Total games (including prior): {total_games:,}")
    print(f"Total model queries: {pipeline.total_model_queries:,}")

    if use_wandb:
        wandb.log({
            'final/vs_random_win_rate': win_rate,
            'final/improvement': win_rate - initial_win_rate,
        })
        wandb.finish()
        print("W&B run finished.")


if __name__ == '__main__':
    main()
