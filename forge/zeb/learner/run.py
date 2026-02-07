"""Distributed learner for Zeb self-play training.

Consumes training examples written by workers, trains the model on batches
from a GPU replay buffer, pushes updated weights to HuggingFace, and logs
to W&B.

This is the training loop from run_selfplay_training.py, decoupled from
game generation: workers produce examples on disk, the learner ingests them.

Usage:
    python -u -m forge.zeb.learner.run \
        --repo-id username/zeb-42 \
        --input-dir /shared/examples \
        --checkpoint forge/zeb/checkpoints/selfplay-epoch2499.pt \
        --lr 1e-4 \
        --batch-size 64 \
        --replay-buffer-size 50000 \
        --training-steps-per-cycle 100 \
        --push-every 10 \
        --save-every 50 \
        --eval-every 50 \
        --keep-checkpoints 3 \
        --wandb
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import torch

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from forge.zeb.example_store import load_examples, scan_pending
from forge.zeb.gpu_training_pipeline import GPUReplayBuffer, GPUTrainingExample
from forge.zeb.hf import (
    download_example,
    init_examples_repo,
    init_repo,
    list_remote_examples,
    prune_remote_examples,
    push_weights,
)
from forge.zeb.evaluate import evaluate_vs_random
from forge.zeb.run_selfplay_training import load_model_from_checkpoint
from forge.zeb.run_mcts_training import DEFAULT_CHECKPOINT_DIR


def train_n_steps_from_buffer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    replay_buffer: GPUReplayBuffer,
    n_steps: int,
    batch_size: int,
) -> dict[str, float]:
    """Train N gradient steps sampling from GPU replay buffer.

    Mirrors GPUTrainingPipeline.train_n_steps_from_buffer but without
    requiring a full pipeline instance.
    """
    import torch.nn.functional as F

    model.train()
    device = replay_buffer.device
    buffer_size = len(replay_buffer)

    total_policy_loss = torch.tensor(0.0, device=device)
    total_value_loss = torch.tensor(0.0, device=device)

    buf_obs = replay_buffer.observations
    buf_masks = replay_buffer.masks
    buf_hand_idx = replay_buffer.hand_indices
    buf_hand_masks = replay_buffer.hand_masks
    buf_policy = replay_buffer.policy_targets
    buf_value = replay_buffer.value_targets

    for _ in range(n_steps):
        indices = torch.randint(0, buffer_size, (batch_size,), device=device)

        obs = buf_obs[indices]
        masks = buf_masks[indices]
        hand_idx = buf_hand_idx[indices]
        hand_masks = buf_hand_masks[indices]
        policy_targets = buf_policy[indices]
        value_targets = buf_value[indices]

        policy_logits, value = model(obs.long(), masks, hand_idx, hand_masks)

        policy_logits = policy_logits.masked_fill(~hand_masks, float('-inf'))
        log_policy = F.log_softmax(policy_logits, dim=-1)
        log_policy_safe = log_policy.clamp(min=-100)
        policy_loss = -(policy_targets * log_policy_safe).sum(dim=-1).mean()

        value_loss = F.mse_loss(value, value_targets)

        loss = policy_loss + value_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_policy_loss = total_policy_loss + policy_loss.detach()
        total_value_loss = total_value_loss + value_loss.detach()

    return {
        'policy_loss': (total_policy_loss / n_steps).item(),
        'value_loss': (total_value_loss / n_steps).item(),
    }


def run_learner(args: argparse.Namespace) -> None:
    device = args.device

    # --- Load model from checkpoint (or fail) ---
    print(f"Loading checkpoint: {args.checkpoint}")
    model, metadata = load_model_from_checkpoint(args.checkpoint, device)
    model_config = metadata['model_config']

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {total_params:,} parameters")
    print(f"  Prior games: {metadata['total_games']:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Restore optimizer if checkpoint has it
    if args.checkpoint.exists():
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            print("  Restored optimizer state")
        start_cycle = ckpt.get('cycle', 0)
        del ckpt
    else:
        start_cycle = 0

    use_hf_examples = bool(args.examples_repo_id)

    # --- HuggingFace repos ---
    init_repo(args.repo_id, model_config)
    push_weights(model, args.repo_id, step=start_cycle, total_games=metadata['total_games'])
    print(f"Initial weights pushed to {args.repo_id}")

    # --- Replay buffer ---
    if use_hf_examples:
        # HF mode: the examples repo IS the replay buffer — rebuild from HF
        init_examples_repo(args.examples_repo_id)
        replay_buffer = GPUReplayBuffer(
            capacity=args.replay_buffer_size,
            device=torch.device(device),
        )
        remote_files = list_remote_examples(args.examples_repo_id)
        seen_files: set[str] = set()
        rebuilt = 0
        for remote_name in remote_files:
            local_path = download_example(args.examples_repo_id, remote_name)
            batch = load_examples(local_path)
            gpu_batch = GPUTrainingExample(
                observations=batch.observations.to(device),
                masks=batch.masks.to(device),
                hand_indices=batch.hand_indices.to(device),
                hand_masks=batch.hand_masks.to(device),
                policy_targets=batch.policy_targets.to(device),
                value_targets=batch.value_targets.to(device),
            )
            replay_buffer.add_batch(gpu_batch)
            rebuilt += gpu_batch.n_examples
            seen_files.add(remote_name)
        print(f"Replay buffer (HF): {len(replay_buffer):,}/{args.replay_buffer_size:,} "
              f"examples from {len(remote_files)} files")
    else:
        # Local mode: restore from checkpoint
        saved_buffer = metadata.get('replay_buffer', [])
        replay_buffer = GPUReplayBuffer.from_list(
            saved_buffer,
            capacity=args.replay_buffer_size,
            device=torch.device(device),
        )
        seen_files = set()
        print(f"Replay buffer (GPU): {len(replay_buffer):,}/{args.replay_buffer_size:,} examples")

    # --- W&B ---
    use_wandb = args.wandb and WANDB_AVAILABLE
    wandb_run_id = metadata.get('wandb_run_id')
    if use_wandb:
        if wandb_run_id:
            wandb.init(
                project=args.wandb_project,
                id=wandb_run_id,
                resume='must',
            )
            print(f"W&B run resumed: {wandb.run.url}")
        else:
            timestamp = datetime.now().strftime("%Y%m%d%H%M")
            run_name = args.run_name or f"learner-{timestamp}"
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                tags=['learner', 'distributed', 'selfplay', 'zeb'],
                config={
                    'mode': 'distributed-learner',
                    'source_checkpoint': str(args.checkpoint),
                    'lr': args.lr,
                    'batch_size': args.batch_size,
                    'replay_buffer_size': args.replay_buffer_size,
                    'training_steps_per_cycle': args.training_steps_per_cycle,
                    'model_config': model_config,
                },
            )
            print(f"W&B run: {wandb.run.url}")

    # --- Output directory ---
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Main loop ---
    total_games = metadata['total_games']
    cycle = start_cycle

    print(f"\n=== Distributed Learner ===")
    example_src = args.examples_repo_id if use_hf_examples else str(args.input_dir)
    print(f"Examples: {example_src}")
    print(f"Repo: {args.repo_id}")
    print(f"LR: {args.lr}, batch: {args.batch_size}, steps/cycle: {args.training_steps_per_cycle}")
    print(f"Min buffer: {args.min_buffer_size:,}, buffer capacity: {args.replay_buffer_size:,}")
    print()

    while True:
        # 1. Ingest new example files from workers
        ingested = 0
        n_new_files = 0

        # Ingest from HF Hub (new files only)
        if use_hf_examples:
            all_remote = list_remote_examples(args.examples_repo_id)
            new_remote = [f for f in all_remote if f not in seen_files]
            for remote_name in new_remote:
                local_path = download_example(args.examples_repo_id, remote_name)
                batch = load_examples(local_path)
                gpu_batch = GPUTrainingExample(
                    observations=batch.observations.to(device),
                    masks=batch.masks.to(device),
                    hand_indices=batch.hand_indices.to(device),
                    hand_masks=batch.hand_masks.to(device),
                    policy_targets=batch.policy_targets.to(device),
                    value_targets=batch.value_targets.to(device),
                )
                replay_buffer.add_batch(gpu_batch)
                total_games += batch.metadata.get('n_games', 0)
                ingested += gpu_batch.n_examples
                seen_files.add(remote_name)
            n_new_files += len(new_remote)

        # Ingest from local disk
        if args.input_dir:
            for f in scan_pending(args.input_dir):
                batch = load_examples(f)
                gpu_batch = GPUTrainingExample(
                    observations=batch.observations.to(device),
                    masks=batch.masks.to(device),
                    hand_indices=batch.hand_indices.to(device),
                    hand_masks=batch.hand_masks.to(device),
                    policy_targets=batch.policy_targets.to(device),
                    value_targets=batch.value_targets.to(device),
                )
                replay_buffer.add_batch(gpu_batch)
                total_games += batch.metadata.get('n_games', 0)
                ingested += gpu_batch.n_examples
                n_new_files += 1
                f.unlink()

        if ingested > 0:
            print(f"Ingested {ingested:,} examples from {n_new_files} files "
                  f"[buffer: {len(replay_buffer):,}]")

        # 2. Wait if buffer too small
        if len(replay_buffer) < args.min_buffer_size:
            if ingested == 0:
                print(f"Buffer {len(replay_buffer):,}/{args.min_buffer_size:,} — waiting for workers...")
            time.sleep(5)
            continue

        # 3. Train
        t0 = time.time()
        model.train()
        metrics = train_n_steps_from_buffer(
            model, optimizer, replay_buffer,
            n_steps=args.training_steps_per_cycle,
            batch_size=args.batch_size,
        )
        train_time = time.time() - t0
        cycle += 1

        print(f"Cycle {cycle:4d}: policy_loss={metrics['policy_loss']:.4f}, "
              f"value_loss={metrics['value_loss']:.4f} "
              f"(train={train_time:.2f}s) "
              f"[buffer: {len(replay_buffer):,}, games: {total_games:,}]")

        # 4. W&B logging
        if use_wandb:
            wandb.log({
                'cycle': cycle,
                'train/policy_loss': metrics['policy_loss'],
                'train/value_loss': metrics['value_loss'],
                'train/total_loss': metrics['policy_loss'] + metrics['value_loss'],
                'perf/train_time_s': train_time,
                'stats/total_games': total_games,
                'stats/replay_buffer_size': len(replay_buffer),
            })

        # 5. Push weights to HF periodically
        if cycle % args.push_every == 0:
            push_weights(model, args.repo_id, step=cycle, total_games=total_games)
            print(f"  Pushed weights (cycle {cycle})")

            # Prune old example files from HF
            if use_hf_examples:
                pruned = prune_remote_examples(
                    args.examples_repo_id, args.keep_example_files,
                )
                if pruned:
                    seen_files -= set(pruned)
                    print(f"  Pruned {len(pruned)} old files from HF")

        # 6. Evaluate vs random periodically
        if cycle % args.eval_every == 0:
            model.eval()
            win_rate = evaluate_vs_random(
                model, n_games=args.eval_games, device=device,
            )['team0_win_rate']
            print(f"  Eval vs Random: {win_rate:.1%}")
            if use_wandb:
                wandb.log({'eval/vs_random_win_rate': win_rate, 'cycle': cycle})

        # 7. Local checkpoint periodically
        if cycle % args.save_every == 0:
            ckpt_path = args.output_dir / f"learner-cycle{cycle:06d}.pt"
            ckpt_data = {
                'cycle': cycle,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'model_config': model_config,
                'training_config': {
                    'mode': 'distributed-learner',
                    'source_checkpoint': str(args.checkpoint),
                    'lr': args.lr,
                    'replay_buffer_size': args.replay_buffer_size,
                    'training_steps_per_cycle': args.training_steps_per_cycle,
                },
                'total_games': total_games,
                'wandb_run_id': wandb.run.id if use_wandb else None,
            }
            # HF mode: replay buffer lives on HF, no need to save in checkpoint
            # Local mode: save buffer for crash recovery
            if not use_hf_examples:
                ckpt_data['replay_buffer'] = replay_buffer.to_list()
            torch.save(ckpt_data, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path.name}")

            # Cleanup old checkpoints
            if args.keep_checkpoints > 0:
                all_ckpts = sorted(
                    args.output_dir.glob("learner-cycle*.pt"),
                    key=lambda p: p.stat().st_mtime,
                )
                if len(all_ckpts) > args.keep_checkpoints:
                    for old_ckpt in all_ckpts[:-args.keep_checkpoints]:
                        old_ckpt.unlink()
                        print(f"  Removed old checkpoint: {old_ckpt.name}")


def main():
    parser = argparse.ArgumentParser(description='Distributed learner for Zeb self-play')

    # Required
    parser.add_argument('--repo-id', type=str, required=True,
                        help='HuggingFace repo for weight distribution')
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='Starting checkpoint')

    # Example input (one or both)
    parser.add_argument('--examples-repo-id', type=str, default=None,
                        help='HF repo for example exchange (e.g. username/zeb-42-examples)')
    parser.add_argument('--input-dir', type=Path, default=None,
                        help='Local directory where workers write example files')
    parser.add_argument('--keep-example-files', type=int, default=15,
                        help='Max example files to retain on HF (oldest pruned)')

    # Training
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--replay-buffer-size', type=int, default=50000)
    parser.add_argument('--training-steps-per-cycle', type=int, default=100)
    parser.add_argument('--min-buffer-size', type=int, default=5000,
                        help='Minimum examples before training starts')
    parser.add_argument('--device', type=str, default='cuda')

    # Periodic actions
    parser.add_argument('--push-every', type=int, default=10,
                        help='Push weights to HF every N cycles')
    parser.add_argument('--save-every', type=int, default=50,
                        help='Save local checkpoint every N cycles')
    parser.add_argument('--eval-every', type=int, default=50,
                        help='Evaluate vs random every N cycles')
    parser.add_argument('--eval-games', type=int, default=2000)

    # Output
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument('--keep-checkpoints', type=int, default=3,
                        help='Keep only last N checkpoints (0 = keep all)')

    # W&B
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--wandb-project', type=str, default='zeb-mcts')
    parser.add_argument('--run-name', type=str, default=None)

    args = parser.parse_args()
    if not args.examples_repo_id and not args.input_dir:
        parser.error("Either --examples-repo-id or --input-dir is required")
    run_learner(args)


if __name__ == '__main__':
    main()
