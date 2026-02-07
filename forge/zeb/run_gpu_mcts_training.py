"""GPU-native MCTS AlphaZero-style training for Zeb.

Uses the GPU-native MCTS pipeline (GPUTrainingPipeline) for zero-copy training:
- Deals generated on GPU
- MCTS tree operations on GPU
- Oracle evaluation with GPU tensors
- Training with GPU-native data

No CPU<->GPU transfers in the hot path - ideal for B200 deployment.

This script mirrors run_mcts_training.py for fair benchmarking.
"""
import argparse
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from forge.zeb.cuda_only import require_cuda

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Import checkpoint utilities from run_mcts_training
from .run_mcts_training import (
    save_checkpoint,
    load_checkpoint,
    find_latest_checkpoint,
    cleanup_old_checkpoints,
    checkpoint_path,
    evaluate_vs_random,
    DEFAULT_CHECKPOINT_DIR,
)
from .gpu_training_pipeline import create_gpu_pipeline, GPUTrainingPipeline
from .model import ZebModel, get_model_config


def main():
    parser = argparse.ArgumentParser(description='GPU-native MCTS Zeb training')

    # Training parameters (mirror run_mcts_training.py)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--games-per-epoch', type=int, default=100)
    parser.add_argument('--n-simulations', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--eval-every', type=int, default=5)
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda'],
        help='CUDA-only (must be cuda)',
    )
    parser.add_argument('--model-size', type=str, default='small',
                        choices=['small', 'medium', 'large'],
                        help='ZebModel size configuration')

    # GPU MCTS parameters
    parser.add_argument('--n-parallel-games', type=int, default=16,
                        help='Number of concurrent MCTS games')
    parser.add_argument('--wave-size', type=int, default=1,
                        help='Number of simulations per evaluation wave (larger batches, fewer oracle/model calls)')
    parser.add_argument('--max-mcts-nodes', type=int, default=1024,
                        help='Maximum nodes per MCTS tree')

    # W&B parameters
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=True,
                        help='Enable W&B logging')
    parser.add_argument('--wandb-project', type=str, default='zeb-mcts',
                        help='W&B project name')
    parser.add_argument('--run-name', type=str, default=None,
                        help='W&B run name (auto-generated if not provided)')

    # Checkpoint and resume parameters
    parser.add_argument('--resume', action='store_true',
                        help='Resume from latest checkpoint (auto-detects by model size)')
    parser.add_argument('--checkpoint-dir', type=Path, default=DEFAULT_CHECKPOINT_DIR,
                        help='Directory for saving/loading checkpoints')
    parser.add_argument('--checkpoint-every', type=int, default=1,
                        help='Save checkpoint every N epochs (default: every epoch)')
    parser.add_argument('--keep-checkpoints', type=int, default=3,
                        help='Keep only the last N checkpoints (default: 3)')

    args = parser.parse_args()

    _ = require_cuda(args.device, where="run_gpu_mcts_training")

    # Check for resume checkpoint
    start_epoch = 0
    resume_checkpoint = None
    wandb_run_id = None

    if args.resume:
        resume_checkpoint = find_latest_checkpoint(args.checkpoint_dir, args.model_size)
        if resume_checkpoint:
            # Peek at checkpoint to get wandb_run_id before initializing wandb
            ckpt_data = torch.load(resume_checkpoint, map_location='cpu', weights_only=False)
            wandb_run_id = ckpt_data.get('wandb_run_id')
            start_epoch = ckpt_data['epoch'] + 1  # Resume from next epoch
            print(f"Found checkpoint: {resume_checkpoint}")
            print(f"  Will resume from epoch {start_epoch}")
        else:
            print(f"No checkpoint found for model size '{args.model_size}' in {args.checkpoint_dir}")
            print("  Starting fresh training")

    # Initialize W&B (with resume if we have a run ID)
    use_wandb = args.wandb and WANDB_AVAILABLE
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
            run_name = args.run_name or f"gpu-mcts-zeb-{args.model_size}-{timestamp}"
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                tags=['gpu-mcts', 'zeb', args.model_size, 'oracle'],
                config={
                    'epochs': args.epochs,
                    'games_per_epoch': args.games_per_epoch,
                    'n_simulations': args.n_simulations,
                    'batch_size': args.batch_size,
                    'lr': args.lr,
                    'temperature': args.temperature,
                    'model_size': args.model_size,
                    'device': args.device,
                    'n_parallel_games': args.n_parallel_games,
                    'wave_size': args.wave_size,
                    'max_mcts_nodes': args.max_mcts_nodes,
                    'pipeline': 'gpu-native',
                },
            )
            print(f"W&B run: {wandb.run.url}")

    print("=== GPU-Native MCTS AlphaZero Training (ZebModel) ===")
    print(f"Epochs: {args.epochs}")
    print(f"Games/epoch: {args.games_per_epoch}")
    print(f"MCTS simulations: {args.n_simulations}")
    print(f"Parallel games: {args.n_parallel_games}")
    print(f"Wave size: {args.wave_size}")
    print(f"Max MCTS nodes: {args.max_mcts_nodes}")
    print(f"Learning rate: {args.lr}")
    print(f"Model size: {args.model_size}")
    print(f"Device: {args.device}")
    print()

    # Initialize GPU training pipeline
    print("Creating GPU training pipeline...")
    t0 = time.time()
    pipeline = create_gpu_pipeline(
        oracle_device=args.device,
        n_parallel_games=args.n_parallel_games,
        n_simulations=args.n_simulations,
        wave_size=args.wave_size,
        max_mcts_nodes=args.max_mcts_nodes,
        temperature=args.temperature,
    )
    print(f"Pipeline created in {time.time() - t0:.1f}s")
    print()

    # Model: ZebModel transformer
    model_config = get_model_config(args.model_size)
    model = ZebModel(**model_config)
    model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Build training config for checkpoint saving
    training_config = {
        'epochs': args.epochs,
        'games_per_epoch': args.games_per_epoch,
        'n_simulations': args.n_simulations,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'temperature': args.temperature,
        'model_size': args.model_size,
        'model_config': model_config,
        'device': args.device,
        'n_parallel_games': args.n_parallel_games,
        'wave_size': args.wave_size,
        'max_mcts_nodes': args.max_mcts_nodes,
        'pipeline': 'gpu-native',
    }

    # Load checkpoint if resuming
    if resume_checkpoint:
        print(f"Loading checkpoint: {resume_checkpoint}")
        load_checkpoint(resume_checkpoint, model, optimizer)
        # Move optimizer state to correct device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(args.device)
        print(f"  Restored model, optimizer, and RNG states")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"ZebModel ({args.model_size}): {total_params:,} parameters")
    print(f"  embed_dim={model_config['embed_dim']}, n_layers={model_config['n_layers']}, n_heads={model_config['n_heads']}")

    # Initial evaluation
    print("\nInitial evaluation...")
    win_rate = evaluate_vs_random(model, n_games=100, device=args.device)
    print(f"  vs Random: {win_rate:.1%}")
    if use_wandb:
        wandb.log({'eval/vs_random_win_rate': win_rate, 'epoch': -1})

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # Record oracle queries before generation
        oracle_queries_before = pipeline.total_oracle_queries

        # Generate games with GPU-native MCTS
        examples = pipeline.generate_games_gpu(n_games=args.games_per_epoch)
        gen_time = time.time() - t0

        # Compute generation stats
        oracle_queries = pipeline.total_oracle_queries - oracle_queries_before
        n_examples = examples.n_examples
        games_per_sec = args.games_per_epoch / gen_time

        # Train on generated examples
        t1 = time.time()
        metrics = pipeline.train_epoch_gpu(
            model=model,
            optimizer=optimizer,
            examples=examples,
            batch_size=args.batch_size,
        )
        train_time = time.time() - t1

        # Build epoch summary
        epoch_str = f"Epoch {epoch:3d}: policy_loss={metrics['policy_loss']:.4f}, value_loss={metrics['value_loss']:.4f}"
        epoch_str += f" (gen={gen_time:.1f}s, train={train_time:.2f}s, {games_per_sec:.2f} games/s)"
        epoch_str += f" [oracle: {oracle_queries:,}]"
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
                'perf/oracle_queries': oracle_queries,
                'stats/total_games_generated': pipeline.total_games_generated,
                'stats/total_oracle_queries': pipeline.total_oracle_queries,
            })

        # Save checkpoint
        if (epoch + 1) % args.checkpoint_every == 0:
            ckpt_path = checkpoint_path(args.checkpoint_dir, args.model_size, epoch)
            current_wandb_id = wandb.run.id if use_wandb else None
            save_checkpoint(ckpt_path, model, optimizer, epoch, training_config, current_wandb_id)
            cleanup_old_checkpoints(args.checkpoint_dir, args.model_size, args.keep_checkpoints)
            if use_wandb:
                wandb.log({'checkpoint/saved': 1, 'checkpoint/epoch': epoch})

        # Periodic evaluation
        if (epoch + 1) % args.eval_every == 0:
            win_rate = evaluate_vs_random(model, n_games=100, device=args.device)
            print(f"         -> vs Random: {win_rate:.1%}")
            if use_wandb:
                wandb.log({'eval/vs_random_win_rate': win_rate, 'epoch': epoch})

    # Final evaluation
    print("\n=== Final Evaluation ===")
    win_rate = evaluate_vs_random(model, n_games=200, device=args.device)
    print(f"vs Random: {win_rate:.1%}")

    # Print cumulative stats
    print(f"\n=== Cumulative Stats ===")
    print(f"Total games generated: {pipeline.total_games_generated:,}")
    print(f"Total oracle queries: {pipeline.total_oracle_queries:,}")

    if use_wandb:
        wandb.log({'final/vs_random_win_rate': win_rate})
        wandb.finish()
        print("W&B run finished.")


if __name__ == '__main__':
    main()
