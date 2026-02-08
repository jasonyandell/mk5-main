"""Distributed learner for Zeb self-play training.

Consumes training examples written by workers, trains the model on batches
from a GPU replay buffer, pushes updated weights to HuggingFace, and logs
to W&B.

HuggingFace is the single source of truth for model weights. On startup the
learner pulls the latest weights from HF. The --checkpoint flag is only used
to bootstrap a brand-new HF repo.

Usage:
    python -u -m forge.zeb.learner.run \
        --repo-id username/zeb-42 \
        --examples-repo-id username/zeb-42-examples \
        --checkpoint forge/zeb/models/zeb-557k-1m.pt \
        --lr 1e-4 \
        --batch-size 64 \
        --replay-buffer-size 200000 \
        --training-steps-per-cycle 100 \
        --push-every 25 \
        --eval-every 50 \
        --wandb --run-name zeb-557k-1m
"""
from __future__ import annotations

import argparse
import time
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
    get_remote_training_state,
    download_example,
    init_examples_repo,
    init_repo,
    list_remote_examples,
    pull_weights,
    prune_remote_examples,
    push_weights,
)
from forge.zeb.evaluate import evaluate_vs_random
from forge.zeb.run_selfplay_training import load_model_from_checkpoint


def train_n_steps_from_buffer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    replay_buffer: GPUReplayBuffer,
    n_steps: int,
    batch_size: int,
) -> dict[str, float]:
    """Train N gradient steps sampling from GPU replay buffer."""
    import torch.nn.functional as F

    model.train()
    device = replay_buffer.device
    buffer_size = len(replay_buffer)

    zero = torch.tensor(0.0, device=device)
    total_policy_loss = zero.clone()
    total_value_loss = zero.clone()
    total_entropy = zero.clone()
    total_grad_norm = zero.clone()
    total_grad_norm_vhead = zero.clone()
    total_top1_acc = zero.clone()
    total_kl_div = zero.clone()
    total_value_mean = zero.clone()
    total_value_std = zero.clone()
    total_value_target_mean = zero.clone()

    value_head_params = [p for p in model.value_head.parameters() if p.requires_grad]

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

        # Gradient norms (after backward, before step)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
        vhead_grads = [p.grad for p in value_head_params if p.grad is not None]
        grad_norm_vhead = torch.stack([g.norm() for g in vhead_grads]).norm() if vhead_grads else zero

        optimizer.step()

        total_policy_loss = total_policy_loss + policy_loss.detach()
        total_value_loss = total_value_loss + value_loss.detach()
        total_grad_norm = total_grad_norm + grad_norm.detach()
        total_grad_norm_vhead = total_grad_norm_vhead + grad_norm_vhead.detach()

        # Policy entropy: -sum(p * log p) over valid actions
        policy_probs = log_policy_safe.exp()
        entropy = -(policy_probs * log_policy_safe).sum(dim=-1).mean()
        total_entropy = total_entropy + entropy.detach()

        # Policy top-1 accuracy: does model argmax match MCTS argmax?
        top1_acc = (policy_probs.argmax(dim=-1) == policy_targets.argmax(dim=-1)).float().mean()
        total_top1_acc = total_top1_acc + top1_acc.detach()

        # KL divergence: KL(MCTS_target || model)
        log_targets = (policy_targets + 1e-8).log()
        kl = (policy_targets * (log_targets - log_policy_safe)).sum(dim=-1).mean()
        total_kl_div = total_kl_div + kl.detach()

        # Value diagnostics
        total_value_mean = total_value_mean + value.detach().mean()
        total_value_std = total_value_std + value.detach().std()
        total_value_target_mean = total_value_target_mean + value_targets.mean()

    n = n_steps
    return {
        'policy_loss': (total_policy_loss / n).item(),
        'value_loss': (total_value_loss / n).item(),
        'policy_entropy': (total_entropy / n).item(),
        'grad_norm': (total_grad_norm / n).item(),
        'grad_norm_value_head': (total_grad_norm_vhead / n).item(),
        'policy_top1_accuracy': (total_top1_acc / n).item(),
        'policy_kl_divergence': (total_kl_div / n).item(),
        'value_mean': (total_value_mean / n).item(),
        'value_std': (total_value_std / n).item(),
        'value_target_mean': (total_value_target_mean / n).item(),
    }


def run_learner(args: argparse.Namespace) -> None:
    device = args.device
    weights_name = f"{args.weights_name}.pt" if args.weights_name else 'model.pt'
    # Derive examples namespace: "model" → None (root), anything else → subdirectory
    stem = weights_name.removesuffix('.pt')
    namespace = None if stem == 'model' else stem

    # --- Bootstrap: load from checkpoint to get model config ---
    print(f"Loading bootstrap checkpoint: {args.checkpoint}")
    model, metadata = load_model_from_checkpoint(args.checkpoint, device)
    model_config = metadata['model_config']

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {total_params:,} parameters")

    # --- HF is the single source of truth ---
    init_repo(args.repo_id, model_config, weights_name=weights_name)
    remote_state = get_remote_training_state(args.repo_id, weights_name=weights_name)

    if remote_state is not None:
        remote_step = int(remote_state.get('step', 0))
        total_games = int(remote_state.get('total_games', 0))
        print(f"Pulling weights from HF: {weights_name} (step {remote_step}, {total_games:,} games)")
        remote_state_dict, remote_config = pull_weights(args.repo_id, device=device, weights_name=weights_name)
        if remote_config != model_config:
            raise ValueError("Remote model config differs from bootstrap checkpoint config")
        model.load_state_dict(remote_state_dict)
        cycle = remote_step
    else:
        # First time: push bootstrap weights to HF
        total_games = metadata['total_games']
        cycle = 0
        push_weights(model, args.repo_id, step=0, total_games=total_games, weights_name=weights_name)
        print(f"Bootstrapped HF repo with {total_games:,} games")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # --- Replay buffer from HF examples ---
    use_hf_examples = bool(args.examples_repo_id)

    if use_hf_examples:
        init_examples_repo(args.examples_repo_id)
        replay_buffer = GPUReplayBuffer(
            capacity=args.replay_buffer_size,
            device=torch.device(device),
        )
        remote_files = list_remote_examples(args.examples_repo_id, namespace)
        seen_files: set[str] = set()
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
            seen_files.add(remote_name)
        print(f"Replay buffer (HF): {len(replay_buffer):,}/{args.replay_buffer_size:,} "
              f"examples from {len(remote_files)} files")
    else:
        replay_buffer = GPUReplayBuffer(
            capacity=args.replay_buffer_size,
            device=torch.device(device),
        )
        seen_files = set()
        print(f"Replay buffer (empty): 0/{args.replay_buffer_size:,}")

    # --- W&B ---
    use_wandb = args.wandb and WANDB_AVAILABLE
    if use_wandb:
        from datetime import datetime
        run_name = args.run_name or (
            f"learner-{stem}-{datetime.now().strftime('%Y%m%d%H%M')}" if namespace
            else f"learner-{datetime.now().strftime('%Y%m%d%H%M')}"
        )
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            tags=['learner', 'distributed', 'selfplay', 'zeb'],
            config={
                'mode': 'distributed-learner',
                'bootstrap_checkpoint': str(args.checkpoint),
                'lr': args.lr,
                'batch_size': args.batch_size,
                'replay_buffer_size': args.replay_buffer_size,
                'training_steps_per_cycle': args.training_steps_per_cycle,
                'model_config': model_config,
            },
        )
        print(f"W&B run: {wandb.run.url}")

    # --- Main loop ---
    print(f"\n=== Distributed Learner ===")
    example_src = args.examples_repo_id if use_hf_examples else str(args.input_dir)
    print(f"Examples: {example_src}")
    print(f"Repo: {args.repo_id}")
    print(f"Starting from cycle {cycle}")
    print(f"LR: {args.lr}, batch: {args.batch_size}, steps/cycle: {args.training_steps_per_cycle}")
    print(f"Min buffer: {args.min_buffer_size:,}, buffer capacity: {args.replay_buffer_size:,}")
    print()

    while True:
        # 1. Ingest new example files from workers
        ingested = 0
        n_new_files = 0

        if use_hf_examples:
            try:
                all_remote = list_remote_examples(args.examples_repo_id, namespace)
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
            except Exception as e:
                print(f"  HF ingest error (skipping): {type(e).__name__}: {e}")

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

        # 4. Build W&B log dict (logged once at end of cycle)
        log_dict = {
            'cycle': cycle,
            'train/policy_loss': metrics['policy_loss'],
            'train/value_loss': metrics['value_loss'],
            'train/total_loss': metrics['policy_loss'] + metrics['value_loss'],
            'train/policy_entropy': metrics['policy_entropy'],
            'train/policy_top1_accuracy': metrics['policy_top1_accuracy'],
            'train/policy_kl_divergence': metrics['policy_kl_divergence'],
            'train/grad_norm': metrics['grad_norm'],
            'train/grad_norm_value_head': metrics['grad_norm_value_head'],
            'train/value_mean': metrics['value_mean'],
            'train/value_std': metrics['value_std'],
            'train/value_target_mean': metrics['value_target_mean'],
            'train/lr': optimizer.param_groups[0]['lr'],
            'perf/train_time_s': train_time,
            'stats/total_games': total_games,
            'stats/replay_buffer_size': len(replay_buffer),
        }

        # 5. Push weights to HF periodically
        if cycle % args.push_every == 0:
            try:
                push_weights(model, args.repo_id, step=cycle, total_games=total_games, weights_name=weights_name)
                print(f"  Pushed weights (cycle {cycle})")

                if use_hf_examples:
                    pruned = prune_remote_examples(
                        args.examples_repo_id, args.keep_example_files,
                        namespace=namespace,
                    )
                    if pruned:
                        seen_files -= set(pruned)
                        print(f"  Pruned {len(pruned)} old files from HF")
            except Exception as e:
                print(f"  HF push error (will retry next cycle): {type(e).__name__}: {e}")

        # 6. Evaluate vs random periodically
        if cycle % args.eval_every == 0:
            model.eval()
            win_rate = evaluate_vs_random(
                model, n_games=args.eval_games, device=device,
            )['team0_win_rate']
            print(f"  Eval vs Random: {win_rate:.1%}")
            log_dict['eval/vs_random_win_rate'] = win_rate

        # 7. Log all metrics for this cycle in a single call
        if use_wandb:
            wandb.log(log_dict)


def main():
    parser = argparse.ArgumentParser(description='Distributed learner for Zeb self-play')

    # Required
    parser.add_argument('--repo-id', type=str, required=True,
                        help='HuggingFace repo for weight distribution (single source of truth)')
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='Bootstrap checkpoint (only used if HF repo has no weights yet)')

    # Example input
    parser.add_argument('--examples-repo-id', type=str, default=None,
                        help='HF repo for example exchange (e.g. username/zeb-42-examples)')
    parser.add_argument('--input-dir', type=Path, default=None,
                        help='Local directory where workers write example files')
    parser.add_argument('--keep-example-files', type=int, default=15,
                        help='Max example files to retain on HF (oldest pruned)')

    # Training
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--replay-buffer-size', type=int, default=200000)
    parser.add_argument('--training-steps-per-cycle', type=int, default=100)
    parser.add_argument('--min-buffer-size', type=int, default=5000,
                        help='Minimum examples before training starts')
    parser.add_argument('--device', type=str, default='cuda')

    # Periodic actions
    parser.add_argument('--push-every', type=int, default=25,
                        help='Push weights to HF every N cycles')
    parser.add_argument('--eval-every', type=int, default=50,
                        help='Evaluate vs random every N cycles')
    parser.add_argument('--eval-games', type=int, default=2000)

    # Model namespace
    parser.add_argument('--weights-name', type=str, default=None,
                        help='Weights filename stem on HF (e.g. large → large.pt, large-config.json)')

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
