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
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from forge.zeb.example_store import load_examples, scan_pending
from forge.zeb.gpu_training_pipeline import GPUReplayBuffer, TrainingExamples
from forge.zeb.tokenizer_registry import get_tokenizer_spec
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


SOURCE_SELFPLAY = 'selfplay-mcts'
SOURCE_EVAL_AUX = 'eval-eq-zeb'


@dataclass
class IngestStats:
    """Per-cycle ingest counters, with explicit eval-aux staleness accounting."""

    ingested_examples: int = 0
    ingested_files: int = 0
    selfplay_examples: int = 0
    selfplay_files: int = 0
    eval_examples_seen: int = 0
    eval_files_seen: int = 0
    eval_examples_kept: int = 0
    eval_files_kept: int = 0
    eval_examples_skipped_disabled: int = 0
    eval_examples_skipped_stale: int = 0
    eval_files_skipped_stale: int = 0
    eval_examples_dropped_downweight: int = 0
    eval_files_downweighted: int = 0
    eval_keep_weight_sum: float = 0.0
    eval_keep_weight_count: int = 0
    eval_lag_max: int = 0
    eval_lag_sum: int = 0
    eval_lag_count: int = 0


@dataclass
class TrainWeights:
    policy: float
    value: float
    belief: float


def _new_replay_buffer(
    *,
    capacity: int,
    device: torch.device,
    spec,
    has_belief: bool,
) -> GPUReplayBuffer:
    return GPUReplayBuffer(
        capacity=capacity,
        device=device,
        max_tokens=spec.max_tokens,
        n_features=spec.n_features,
        n_hand_slots=spec.n_hand_slots,
        belief=has_belief,
    )


def _source_from_metadata(metadata: dict | None) -> str:
    source = (metadata or {}).get('source', SOURCE_SELFPLAY)
    if not isinstance(source, str):
        return SOURCE_SELFPLAY
    return source


def _coerce_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _subset_examples(examples: TrainingExamples, keep_mask: torch.Tensor) -> TrainingExamples:
    idx = keep_mask.nonzero(as_tuple=True)[0]
    return TrainingExamples(
        observations=examples.observations[idx],
        masks=examples.masks[idx],
        hand_indices=examples.hand_indices[idx],
        hand_masks=examples.hand_masks[idx],
        policy_targets=examples.policy_targets[idx],
        value_targets=examples.value_targets[idx],
        belief_targets=examples.belief_targets[idx] if examples.belief_targets is not None else None,
        belief_mask=examples.belief_mask[idx] if examples.belief_mask is not None else None,
        metadata=examples.metadata,
    )


def _eval_aux_keep_weight(
    *,
    lag_steps: int,
    half_life: int,
    min_weight: float,
) -> float:
    """Decay stale eval-aux influence with a model-step half-life."""
    if half_life <= 0:
        return 1.0
    weight = math.pow(0.5, max(0.0, lag_steps) / max(1.0, float(half_life)))
    return max(min_weight, min(1.0, weight))


def _sample_batch_from_buffer(replay_buffer: GPUReplayBuffer, n: int) -> TrainingExamples:
    indices = torch.randint(0, len(replay_buffer), (n,), device=replay_buffer.device)
    has_belief = replay_buffer.has_belief
    return TrainingExamples(
        observations=replay_buffer.observations[indices],
        masks=replay_buffer.masks[indices],
        hand_indices=replay_buffer.hand_indices[indices],
        hand_masks=replay_buffer.hand_masks[indices],
        policy_targets=replay_buffer.policy_targets[indices],
        value_targets=replay_buffer.value_targets[indices],
        belief_targets=replay_buffer.belief_targets[indices] if has_belief else None,
        belief_mask=replay_buffer.belief_mask[indices] if has_belief else None,
        metadata=None,
    )


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
    has_belief = replay_buffer.has_belief and getattr(model, 'has_belief_head', False)

    zero = torch.tensor(0.0, device=device)
    total_policy_loss = zero.clone()
    total_value_loss = zero.clone()
    total_belief_loss = zero.clone()
    total_belief_acc = zero.clone()
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
    buf_belief = replay_buffer.belief_targets if has_belief else None
    buf_bmask = replay_buffer.belief_mask if has_belief else None

    for _ in range(n_steps):
        indices = torch.randint(0, buffer_size, (batch_size,), device=device)

        obs = buf_obs[indices]
        masks = buf_masks[indices]
        hand_idx = buf_hand_idx[indices]
        hand_masks = buf_hand_masks[indices]
        policy_targets = buf_policy[indices]
        value_targets = buf_value[indices]

        policy_logits, value, belief_logits = model(obs.long(), masks, hand_idx, hand_masks)

        policy_logits = policy_logits.masked_fill(~hand_masks, float('-inf'))
        log_policy = F.log_softmax(policy_logits, dim=-1)
        log_policy_safe = log_policy.clamp(min=-100)
        policy_loss = -(policy_targets * log_policy_safe).sum(dim=-1).mean()

        value_loss = F.mse_loss(value, value_targets)

        loss = policy_loss + value_loss

        # Belief loss
        if has_belief and belief_logits is not None:
            b_targets = buf_belief[indices]
            b_mask = buf_bmask[indices]
            ce = F.cross_entropy(
                belief_logits.permute(0, 2, 1), b_targets, reduction='none',
            )
            belief_loss = (ce * b_mask.float()).sum() / b_mask.float().sum().clamp(min=1)
            loss = loss + 0.5 * belief_loss
            total_belief_loss = total_belief_loss + belief_loss.detach()

            # Belief accuracy on masked positions
            preds = belief_logits.argmax(dim=-1)  # (B, 28)
            correct = (preds == b_targets) & b_mask
            acc = correct.float().sum() / b_mask.float().sum().clamp(min=1)
            total_belief_acc = total_belief_acc + acc.detach()

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
    result = {
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
    if has_belief:
        result['belief_loss'] = (total_belief_loss / n).item()
        result['belief_accuracy'] = (total_belief_acc / n).item()
    return result


def train_n_steps_source_aware(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    selfplay_buffer: GPUReplayBuffer,
    eval_aux_buffer: GPUReplayBuffer,
    n_steps: int,
    batch_size: int,
    *,
    eval_aux_batch_fraction: float,
    selfplay_weights: TrainWeights,
    eval_aux_weights: TrainWeights,
) -> dict[str, float]:
    """Train with source-aware objectives across self-play and eval-aux buffers."""
    import torch.nn.functional as F

    model.train()
    device = selfplay_buffer.device
    has_belief = bool(getattr(model, 'has_belief_head', False))

    zero = torch.tensor(0.0, device=device)
    total_policy_loss = zero.clone()
    total_value_loss = zero.clone()
    total_belief_loss = zero.clone()
    total_belief_acc = zero.clone()
    total_entropy = zero.clone()
    total_grad_norm = zero.clone()
    total_grad_norm_vhead = zero.clone()
    total_top1_acc = zero.clone()
    total_kl_div = zero.clone()
    total_value_mean = zero.clone()
    total_value_std = zero.clone()
    total_value_target_mean = zero.clone()
    total_objective = zero.clone()

    total_policy_examples = 0
    total_value_examples = 0
    total_belief_examples = 0
    total_selfplay_examples = 0
    total_eval_examples = 0

    total_eval_value_loss = zero.clone()
    total_eval_belief_loss = zero.clone()
    total_eval_policy_loss = zero.clone()
    total_eval_steps = 0

    value_head_params = [p for p in model.value_head.parameters() if p.requires_grad]

    have_self = len(selfplay_buffer) > 0
    have_eval = len(eval_aux_buffer) > 0 and eval_aux_batch_fraction > 0.0

    if not have_self and not have_eval:
        raise ValueError('No samples available for source-aware training')

    for _ in range(n_steps):
        requested_eval = int(round(batch_size * max(0.0, min(1.0, eval_aux_batch_fraction))))
        eval_n = requested_eval if have_eval else 0
        self_n = batch_size - eval_n

        if have_self and self_n <= 0:
            self_n = 1
            eval_n = batch_size - 1
        if not have_self:
            self_n = 0
            eval_n = batch_size
        if not have_eval:
            self_n = batch_size
            eval_n = 0

        sources: list[tuple[str, TrainingExamples, TrainWeights]] = []
        if self_n > 0:
            sources.append((SOURCE_SELFPLAY, _sample_batch_from_buffer(selfplay_buffer, self_n), selfplay_weights))
        if eval_n > 0:
            sources.append((SOURCE_EVAL_AUX, _sample_batch_from_buffer(eval_aux_buffer, eval_n), eval_aux_weights))

        loss = zero.clone()

        optimizer.zero_grad()

        for source_name, batch, weights in sources:
            obs = batch.observations
            masks = batch.masks
            hand_idx = batch.hand_indices
            hand_masks = batch.hand_masks
            policy_targets = batch.policy_targets
            value_targets = batch.value_targets

            policy_logits, value, belief_logits = model(obs.long(), masks, hand_idx, hand_masks)
            policy_logits = policy_logits.masked_fill(~hand_masks, float('-inf'))
            log_policy = F.log_softmax(policy_logits, dim=-1)
            log_policy_safe = log_policy.clamp(min=-100)

            policy_loss = -(policy_targets * log_policy_safe).sum(dim=-1).mean()
            value_loss = F.mse_loss(value, value_targets)

            source_loss = torch.tensor(0.0, device=device)
            source_loss = source_loss + weights.policy * policy_loss
            source_loss = source_loss + weights.value * value_loss

            belief_loss = None
            belief_acc = None
            if has_belief and belief_logits is not None and batch.belief_targets is not None and batch.belief_mask is not None:
                b_targets = batch.belief_targets
                b_mask = batch.belief_mask
                ce = F.cross_entropy(
                    belief_logits.permute(0, 2, 1), b_targets, reduction='none',
                )
                belief_loss = (ce * b_mask.float()).sum() / b_mask.float().sum().clamp(min=1)
                source_loss = source_loss + weights.belief * belief_loss

                preds = belief_logits.argmax(dim=-1)
                correct = (preds == b_targets) & b_mask
                belief_acc = correct.float().sum() / b_mask.float().sum().clamp(min=1)

                total_belief_loss = total_belief_loss + belief_loss.detach() * obs.shape[0]
                total_belief_acc = total_belief_acc + belief_acc.detach() * obs.shape[0]
                total_belief_examples += obs.shape[0]

            loss = loss + source_loss

            total_value_loss = total_value_loss + value_loss.detach() * obs.shape[0]
            total_value_examples += obs.shape[0]

            # Keep policy diagnostics anchored to self-play unless self-play is absent.
            use_for_policy_diag = source_name == SOURCE_SELFPLAY or (not have_self)
            if use_for_policy_diag:
                total_policy_loss = total_policy_loss + policy_loss.detach() * obs.shape[0]
                total_policy_examples += obs.shape[0]

                policy_probs = log_policy_safe.exp()
                entropy = -(policy_probs * log_policy_safe).sum(dim=-1).mean()
                top1_acc = (policy_probs.argmax(dim=-1) == policy_targets.argmax(dim=-1)).float().mean()
                log_targets = (policy_targets + 1e-8).log()
                kl = (policy_targets * (log_targets - log_policy_safe)).sum(dim=-1).mean()
                total_entropy = total_entropy + entropy.detach() * obs.shape[0]
                total_top1_acc = total_top1_acc + top1_acc.detach() * obs.shape[0]
                total_kl_div = total_kl_div + kl.detach() * obs.shape[0]

            total_value_mean = total_value_mean + value.detach().mean() * obs.shape[0]
            total_value_std = total_value_std + value.detach().std() * obs.shape[0]
            total_value_target_mean = total_value_target_mean + value_targets.mean() * obs.shape[0]

            if source_name == SOURCE_SELFPLAY:
                total_selfplay_examples += obs.shape[0]
            else:
                total_eval_examples += obs.shape[0]
                total_eval_steps += 1
                total_eval_policy_loss = total_eval_policy_loss + policy_loss.detach()
                total_eval_value_loss = total_eval_value_loss + value_loss.detach()
                if belief_loss is not None:
                    total_eval_belief_loss = total_eval_belief_loss + belief_loss.detach()

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
        vhead_grads = [p.grad for p in value_head_params if p.grad is not None]
        grad_norm_vhead = torch.stack([g.norm() for g in vhead_grads]).norm() if vhead_grads else zero

        optimizer.step()

        total_grad_norm = total_grad_norm + grad_norm.detach()
        total_grad_norm_vhead = total_grad_norm_vhead + grad_norm_vhead.detach()
        total_objective = total_objective + loss.detach()

    policy_den = max(total_policy_examples, 1)
    value_den = max(total_value_examples, 1)
    belief_den = max(total_belief_examples, 1)

    result: dict[str, float] = {
        'policy_loss': (total_policy_loss / policy_den).item(),
        'value_loss': (total_value_loss / value_den).item(),
        'policy_entropy': (total_entropy / policy_den).item(),
        'grad_norm': (total_grad_norm / n_steps).item(),
        'grad_norm_value_head': (total_grad_norm_vhead / n_steps).item(),
        'policy_top1_accuracy': (total_top1_acc / policy_den).item(),
        'policy_kl_divergence': (total_kl_div / policy_den).item(),
        'value_mean': (total_value_mean / value_den).item(),
        'value_std': (total_value_std / value_den).item(),
        'value_target_mean': (total_value_target_mean / value_den).item(),
        'objective_loss': (total_objective / n_steps).item(),
        'selfplay_examples_per_step': float(total_selfplay_examples) / max(n_steps, 1),
        'eval_aux_examples_per_step': float(total_eval_examples) / max(n_steps, 1),
    }
    if total_belief_examples > 0:
        result['belief_loss'] = (total_belief_loss / belief_den).item()
        result['belief_accuracy'] = (total_belief_acc / belief_den).item()

    if total_eval_steps > 0:
        result['eval_aux_policy_loss'] = (total_eval_policy_loss / total_eval_steps).item()
        result['eval_aux_value_loss'] = (total_eval_value_loss / total_eval_steps).item()
        if total_belief_examples > 0:
            result['eval_aux_belief_loss'] = (total_eval_belief_loss / total_eval_steps).item()
    return result


def _route_batch_into_replay(
    *,
    batch: TrainingExamples,
    cycle: int,
    args: argparse.Namespace,
    device: str,
    selfplay_buffer: GPUReplayBuffer,
    eval_aux_buffer: GPUReplayBuffer,
    stats: IngestStats,
) -> tuple[int, int]:
    """Route a batch by source, applying eval-aux kill-switch and staleness logic."""
    source = _source_from_metadata(batch.metadata)
    n_examples = batch.n_examples

    if source == SOURCE_EVAL_AUX:
        stats.eval_examples_seen += n_examples
        stats.eval_files_seen += 1

        if not args.eval_aux_enabled:
            stats.eval_examples_skipped_disabled += n_examples
            return 0, 0

        model_step = _coerce_int((batch.metadata or {}).get('model_step'))
        lag_steps = None
        if model_step is not None:
            lag_steps = max(0, cycle - model_step)
            stats.eval_lag_max = max(stats.eval_lag_max, lag_steps)
            stats.eval_lag_sum += lag_steps
            stats.eval_lag_count += 1

        if args.eval_aux_max_model_lag >= 0 and lag_steps is not None and lag_steps > args.eval_aux_max_model_lag:
            stats.eval_examples_skipped_stale += n_examples
            stats.eval_files_skipped_stale += 1
            return 0, 0

        keep_weight = 1.0
        if lag_steps is not None:
            keep_weight = _eval_aux_keep_weight(
                lag_steps=lag_steps,
                half_life=args.eval_aux_lag_half_life,
                min_weight=args.eval_aux_min_keep_weight,
            )

        if keep_weight < 0.999999:
            keep_mask = torch.rand(n_examples) < keep_weight
            kept = int(keep_mask.sum().item())
            dropped = n_examples - kept
            stats.eval_examples_dropped_downweight += dropped
            if dropped > 0:
                stats.eval_files_downweighted += 1
            if kept == 0:
                return 0, 0
            batch = _subset_examples(batch, keep_mask)
            stats.eval_keep_weight_sum += keep_weight
            stats.eval_keep_weight_count += 1

        eval_aux_buffer.add_batch(batch.to(device))
        stats.eval_examples_kept += batch.n_examples
        stats.eval_files_kept += 1
        stats.ingested_examples += batch.n_examples
        stats.ingested_files += 1
        added_games = int((batch.metadata or {}).get('n_games', 0))
        return batch.n_examples, added_games

    # Default path: treat unknown source as self-play MCTS.
    selfplay_buffer.add_batch(batch.to(device))
    stats.selfplay_examples += batch.n_examples
    stats.selfplay_files += 1
    stats.ingested_examples += batch.n_examples
    stats.ingested_files += 1
    added_games = int((batch.metadata or {}).get('n_games', 0))
    return batch.n_examples, added_games


def _format_eval_aux_ingest(stats: IngestStats) -> str:
    if stats.eval_files_seen == 0:
        return ""
    parts = [
        f"eval_aux seen={stats.eval_examples_seen:,}/{stats.eval_files_seen} files",
        f"kept={stats.eval_examples_kept:,}/{stats.eval_files_kept} files",
    ]
    if stats.eval_examples_skipped_disabled:
        parts.append(f"disabled_skip={stats.eval_examples_skipped_disabled:,}")
    if stats.eval_examples_skipped_stale:
        parts.append(
            f"stale_skip={stats.eval_examples_skipped_stale:,}/{stats.eval_files_skipped_stale} files"
        )
    if stats.eval_examples_dropped_downweight:
        parts.append(f"downweight_drop={stats.eval_examples_dropped_downweight:,}")
    if stats.eval_lag_count:
        avg_lag = stats.eval_lag_sum / stats.eval_lag_count
        parts.append(f"lag(avg/max)={avg_lag:.1f}/{stats.eval_lag_max}")
    if stats.eval_keep_weight_count:
        avg_keep = stats.eval_keep_weight_sum / stats.eval_keep_weight_count
        parts.append(f"avg_keep_weight={avg_keep:.3f}")
    return " | ".join(parts)


def run_learner(args: argparse.Namespace) -> None:
    device = args.device
    torch_device = torch.device(device)
    weights_name = f"{args.weights_name}.pt" if args.weights_name else 'model.pt'
    # Derive examples namespace: "model" -> None (root), anything else -> subdirectory
    stem = weights_name.removesuffix('.pt')
    namespace = None if stem == 'model' else stem

    # --- Bootstrap: load from checkpoint to get model config ---
    print(f"Loading bootstrap checkpoint: {args.checkpoint}")
    model, metadata = load_model_from_checkpoint(args.checkpoint, device)
    model_config = metadata['model_config']

    total_params = sum(p.numel() for p in model.parameters())
    tokenizer_name = model_config.get('tokenizer', 'v1')
    has_belief = model_config.get('belief_head', False)
    spec = get_tokenizer_spec(tokenizer_name)
    print(f"  Model: {total_params:,} parameters, tokenizer={tokenizer_name}"
          + (", belief_head=True" if has_belief else ""))

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
        if has_belief:
            missing, _ = model.load_state_dict(remote_state_dict, strict=False)
            if missing:
                print(f"  Initializing missing belief head weights: {missing}")
        else:
            model.load_state_dict(remote_state_dict)
        cycle = remote_step
    else:
        # First time: push bootstrap weights to HF
        total_games = metadata['total_games']
        cycle = 0
        push_weights(model, args.repo_id, step=0, total_games=total_games, weights_name=weights_name)
        print(f"Bootstrapped HF repo with {total_games:,} games")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # --- Replay buffers from examples ---
    use_hf_examples = bool(args.examples_repo_id)
    selfplay_buffer = _new_replay_buffer(
        capacity=args.replay_buffer_size,
        device=torch_device,
        spec=spec,
        has_belief=has_belief,
    )
    eval_aux_buffer = _new_replay_buffer(
        capacity=args.eval_aux_replay_buffer_size,
        device=torch_device,
        spec=spec,
        has_belief=has_belief,
    )

    seen_files: set[str] = set()

    if use_hf_examples:
        init_examples_repo(args.examples_repo_id)
        remote_files = list_remote_examples(args.examples_repo_id, namespace)
        bootstrap_stats = IngestStats()
        for remote_name in remote_files:
            local_path = download_example(args.examples_repo_id, remote_name)
            batch = load_examples(local_path)
            _route_batch_into_replay(
                batch=batch,
                cycle=cycle,
                args=args,
                device=device,
                selfplay_buffer=selfplay_buffer,
                eval_aux_buffer=eval_aux_buffer,
                stats=bootstrap_stats,
            )
            seen_files.add(remote_name)
        print(
            "Replay buffers (HF): "
            f"selfplay={len(selfplay_buffer):,}/{args.replay_buffer_size:,}, "
            f"eval_aux={len(eval_aux_buffer):,}/{args.eval_aux_replay_buffer_size:,} "
            f"from {len(remote_files)} files"
        )
        msg = _format_eval_aux_ingest(bootstrap_stats)
        if msg:
            print(f"  Bootstrap eval-aux ingest: {msg}")
    else:
        print(
            "Replay buffers (empty): "
            f"selfplay=0/{args.replay_buffer_size:,}, "
            f"eval_aux=0/{args.eval_aux_replay_buffer_size:,}"
        )

    # --- W&B ---
    use_wandb = args.wandb and WANDB_AVAILABLE
    if use_wandb:
        wandb_run_id = remote_state.get('wandb_run_id') if remote_state else None
        run_name = args.run_name or (
            f"learner-{stem}" if namespace else "learner"
        )
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            id=wandb_run_id,
            resume="allow",
            tags=['learner', 'distributed', 'selfplay', 'zeb'],
            config={
                'mode': 'distributed-learner',
                'bootstrap_checkpoint': str(args.checkpoint),
                'lr': args.lr,
                'batch_size': args.batch_size,
                'replay_buffer_size': args.replay_buffer_size,
                'eval_aux_replay_buffer_size': args.eval_aux_replay_buffer_size,
                'training_steps_per_cycle': args.training_steps_per_cycle,
                'model_config': model_config,
                'eval_aux_enabled': args.eval_aux_enabled,
                'eval_aux_batch_fraction': args.eval_aux_batch_fraction,
                'eval_aux_policy_weight': args.eval_aux_policy_weight,
                'eval_aux_value_weight': args.eval_aux_value_weight,
                'eval_aux_belief_weight': args.eval_aux_belief_weight,
                'eval_aux_max_model_lag': args.eval_aux_max_model_lag,
                'eval_aux_lag_half_life': args.eval_aux_lag_half_life,
            },
        )
        if wandb.run.id != wandb_run_id:
            wandb_run_id = wandb.run.id
        print(f"W&B run: {wandb.run.url}")

    # --- Main loop ---
    print("\n=== Distributed Learner ===")
    example_src = args.examples_repo_id if use_hf_examples else str(args.input_dir)
    print(f"Examples: {example_src}")
    print(f"Repo: {args.repo_id}")
    print(f"Starting from cycle {cycle}")
    print(f"LR: {args.lr}, batch: {args.batch_size}, steps/cycle: {args.training_steps_per_cycle}")
    print(
        f"Buffers: selfplay min={args.min_buffer_size:,}/{args.replay_buffer_size:,}, "
        f"eval_aux={args.eval_aux_replay_buffer_size:,}"
    )
    print(
        "Source routing: "
        f"{SOURCE_SELFPLAY}=policy+value+belief, "
        f"{SOURCE_EVAL_AUX}=policy*{args.eval_aux_policy_weight:g} "
        f"+ value*{args.eval_aux_value_weight:g} + belief*{args.eval_aux_belief_weight:g}"
    )
    print(
        f"Eval-aux kill-switch: {'enabled' if args.eval_aux_enabled else 'disabled'}, "
        f"max_lag={args.eval_aux_max_model_lag}, half_life={args.eval_aux_lag_half_life}, "
        f"min_keep={args.eval_aux_min_keep_weight:.3f}"
    )
    print()

    last_ingest_time = time.time()

    selfplay_weights = TrainWeights(policy=1.0, value=1.0, belief=0.5)
    eval_aux_weights = TrainWeights(
        policy=args.eval_aux_policy_weight,
        value=args.eval_aux_value_weight,
        belief=args.eval_aux_belief_weight,
    )

    while True:
        # 1. Ingest new example files from workers
        ingest_stats = IngestStats()

        if use_hf_examples:
            try:
                all_remote = list_remote_examples(args.examples_repo_id, namespace)
                new_remote = [f for f in all_remote if f not in seen_files]
                for remote_name in new_remote:
                    local_path = download_example(args.examples_repo_id, remote_name)
                    batch = load_examples(local_path)
                    _, added_games = _route_batch_into_replay(
                        batch=batch,
                        cycle=cycle,
                        args=args,
                        device=device,
                        selfplay_buffer=selfplay_buffer,
                        eval_aux_buffer=eval_aux_buffer,
                        stats=ingest_stats,
                    )
                    total_games += added_games
                    seen_files.add(remote_name)
            except Exception as e:
                print(f"  HF ingest error (skipping): {type(e).__name__}: {e}")

        if args.input_dir:
            for f in scan_pending(args.input_dir):
                batch = load_examples(f)
                _, added_games = _route_batch_into_replay(
                    batch=batch,
                    cycle=cycle,
                    args=args,
                    device=device,
                    selfplay_buffer=selfplay_buffer,
                    eval_aux_buffer=eval_aux_buffer,
                    stats=ingest_stats,
                )
                total_games += added_games
                f.unlink()

        if ingest_stats.ingested_examples > 0:
            last_ingest_time = time.time()
            print(
                f"Ingested {ingest_stats.ingested_examples:,} examples "
                f"from {ingest_stats.ingested_files} files "
                f"[selfplay: {len(selfplay_buffer):,}, eval_aux: {len(eval_aux_buffer):,}]"
            )

        eval_ingest_msg = _format_eval_aux_ingest(ingest_stats)
        if eval_ingest_msg:
            print(f"  {eval_ingest_msg}")

        # 2. Freshness check: pause if workers have stopped producing
        if args.max_example_age > 0:
            stale_seconds = time.time() - last_ingest_time
            if stale_seconds > args.max_example_age:
                print(
                    f"No new examples for {stale_seconds:.0f}s "
                    f"(limit: {args.max_example_age}s) — pausing training"
                )
                time.sleep(10)
                continue

        # 3. Wait if self-play buffer too small
        if len(selfplay_buffer) < args.min_buffer_size:
            if ingest_stats.ingested_examples == 0:
                print(
                    f"Self-play buffer {len(selfplay_buffer):,}/{args.min_buffer_size:,} "
                    f"(eval_aux={len(eval_aux_buffer):,}) — waiting for workers..."
                )
            time.sleep(5)
            continue

        # 4. Train
        t0 = time.time()
        model.train()

        use_source_aware = (
            args.eval_aux_enabled
            and args.eval_aux_batch_fraction > 0.0
            and len(eval_aux_buffer) > 0
        )
        if use_source_aware:
            metrics = train_n_steps_source_aware(
                model,
                optimizer,
                selfplay_buffer,
                eval_aux_buffer,
                n_steps=args.training_steps_per_cycle,
                batch_size=args.batch_size,
                eval_aux_batch_fraction=args.eval_aux_batch_fraction,
                selfplay_weights=selfplay_weights,
                eval_aux_weights=eval_aux_weights,
            )
        else:
            metrics = train_n_steps_from_buffer(
                model,
                optimizer,
                selfplay_buffer,
                n_steps=args.training_steps_per_cycle,
                batch_size=args.batch_size,
            )
            metrics['selfplay_examples_per_step'] = float(args.batch_size)
            metrics['eval_aux_examples_per_step'] = 0.0
            metrics['objective_loss'] = (
                metrics['policy_loss']
                + metrics['value_loss']
                + (0.5 * metrics['belief_loss'] if 'belief_loss' in metrics else 0.0)
            )

        train_time = time.time() - t0
        cycle += 1

        cycle_str = (
            f"Cycle {cycle:4d}: policy_loss={metrics['policy_loss']:.4f}, "
            f"value_loss={metrics['value_loss']:.4f}"
        )
        if 'belief_loss' in metrics:
            cycle_str += f", belief_loss={metrics['belief_loss']:.4f}"
            cycle_str += f", belief_acc={metrics['belief_accuracy']:.3f}"
        if use_source_aware:
            cycle_str += (
                f", eval_aux_examples/step={metrics['eval_aux_examples_per_step']:.1f}"
                f", eval_aux_policy_w={args.eval_aux_policy_weight:g}"
            )
        cycle_str += (
            f" (train={train_time:.2f}s) "
            f"[selfplay: {len(selfplay_buffer):,}, eval_aux: {len(eval_aux_buffer):,}, games: {total_games:,}]"
        )
        print(cycle_str)

        # 5. Build W&B log dict (single log per cycle)
        total_loss = metrics['objective_loss']
        log_dict = {
            'cycle': cycle,
            'train/policy_loss': metrics['policy_loss'],
            'train/value_loss': metrics['value_loss'],
            'train/total_loss': total_loss,
            'train/objective_loss': metrics['objective_loss'],
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
            'stats/replay_buffer_selfplay': len(selfplay_buffer),
            'stats/replay_buffer_eval_aux': len(eval_aux_buffer),
            'stats/selfplay_examples_per_step': metrics['selfplay_examples_per_step'],
            'stats/eval_aux_examples_per_step': metrics['eval_aux_examples_per_step'],
            'ingest/eval_aux_seen_examples': ingest_stats.eval_examples_seen,
            'ingest/eval_aux_kept_examples': ingest_stats.eval_examples_kept,
            'ingest/eval_aux_skipped_disabled_examples': ingest_stats.eval_examples_skipped_disabled,
            'ingest/eval_aux_skipped_stale_examples': ingest_stats.eval_examples_skipped_stale,
            'ingest/eval_aux_downweight_dropped_examples': ingest_stats.eval_examples_dropped_downweight,
        }
        if ingest_stats.eval_lag_count:
            log_dict['ingest/eval_aux_lag_avg'] = ingest_stats.eval_lag_sum / ingest_stats.eval_lag_count
            log_dict['ingest/eval_aux_lag_max'] = ingest_stats.eval_lag_max
        if ingest_stats.eval_keep_weight_count:
            log_dict['ingest/eval_aux_keep_weight_avg'] = (
                ingest_stats.eval_keep_weight_sum / ingest_stats.eval_keep_weight_count
            )

        if 'belief_loss' in metrics:
            log_dict['train/belief_loss'] = metrics['belief_loss']
            log_dict['train/belief_accuracy'] = metrics['belief_accuracy']
        if 'eval_aux_policy_loss' in metrics:
            log_dict['train/eval_aux_policy_loss'] = metrics['eval_aux_policy_loss']
            log_dict['train/eval_aux_value_loss'] = metrics['eval_aux_value_loss']
        if 'eval_aux_belief_loss' in metrics:
            log_dict['train/eval_aux_belief_loss'] = metrics['eval_aux_belief_loss']

        # 6. Push weights to HF periodically
        if cycle % args.push_every == 0:
            try:
                extra = {'wandb_run_id': wandb.run.id} if use_wandb else None
                push_weights(
                    model,
                    args.repo_id,
                    step=cycle,
                    total_games=total_games,
                    extra_metadata=extra,
                    weights_name=weights_name,
                )
                print(f"  Pushed weights (cycle {cycle})")

                if use_hf_examples:
                    pruned = prune_remote_examples(
                        args.examples_repo_id,
                        args.keep_example_files,
                        namespace=namespace,
                    )
                    if pruned:
                        seen_files -= set(pruned)
                        print(f"  Pruned {len(pruned)} old files from HF")
            except Exception as e:
                print(f"  HF push error (will retry next cycle): {type(e).__name__}: {e}")

        # 7. Evaluate vs random periodically
        if cycle % args.eval_every == 0:
            model.eval()
            win_rate = evaluate_vs_random(
                model,
                n_games=args.eval_games,
                device=device,
            )['team0_win_rate']
            print(f"  Eval vs Random: {win_rate:.1%}")
            log_dict['eval/vs_random_win_rate'] = win_rate

        # 8. Log metrics
        if use_wandb:
            wandb.log(log_dict, step=cycle)


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
                        help='Minimum self-play examples before training starts')
    parser.add_argument('--device', type=str, default='cuda')

    # Eval-aux source-aware controls
    parser.add_argument('--eval-aux-enabled', action=argparse.BooleanOptionalAction, default=True,
                        help='Enable eval-eq-zeb ingest/training path (kill-switch via --no-eval-aux-enabled)')
    parser.add_argument('--eval-aux-replay-buffer-size', type=int, default=100000,
                        help='Replay buffer capacity for eval-eq-zeb examples')
    parser.add_argument('--eval-aux-batch-fraction', type=float, default=0.20,
                        help='Fraction of each train batch drawn from eval-aux buffer (0 disables eval-aux training)')
    parser.add_argument('--eval-aux-policy-weight', type=float, default=0.0,
                        help='Policy loss weight for eval-aux samples (default 0)')
    parser.add_argument('--eval-aux-value-weight', type=float, default=1.0,
                        help='Value loss weight for eval-aux samples')
    parser.add_argument('--eval-aux-belief-weight', type=float, default=0.5,
                        help='Belief loss weight for eval-aux samples')
    parser.add_argument('--eval-aux-max-model-lag', type=int, default=400,
                        help='Max allowed (cycle - model_step) for eval-aux batches; -1 disables staleness filter')
    parser.add_argument('--eval-aux-lag-half-life', type=int, default=200,
                        help='Model-step half-life for eval-aux keep-probability downweighting (0 disables)')
    parser.add_argument('--eval-aux-min-keep-weight', type=float, default=0.10,
                        help='Floor keep weight after lag decay for eval-aux batches')

    # Periodic actions
    parser.add_argument('--push-every', type=int, default=25,
                        help='Push weights to HF every N cycles')
    parser.add_argument('--eval-every', type=int, default=50,
                        help='Evaluate vs random every N cycles')
    parser.add_argument('--eval-games', type=int, default=2000)

    # Model namespace
    parser.add_argument('--weights-name', type=str, default=None,
                        help='Weights filename stem on HF (e.g. large -> large.pt, large-config.json)')

    # W&B + learner freshness
    parser.add_argument('--max-example-age', type=int, default=300,
                        help='Pause training if no new examples for this many seconds (0=disable)')

    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--wandb-project', type=str, default='zeb-mcts')
    parser.add_argument('--run-name', type=str, default=None)

    args = parser.parse_args()
    if not args.examples_repo_id and not args.input_dir:
        parser.error("Either --examples-repo-id or --input-dir is required")
    if args.eval_aux_batch_fraction < 0 or args.eval_aux_batch_fraction > 1:
        parser.error('--eval-aux-batch-fraction must be in [0, 1]')
    if args.eval_aux_min_keep_weight < 0 or args.eval_aux_min_keep_weight > 1:
        parser.error('--eval-aux-min-keep-weight must be in [0, 1]')
    run_learner(args)


if __name__ == '__main__':
    main()
