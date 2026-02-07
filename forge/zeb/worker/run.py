"""Self-play worker for distributed training.

Generates training examples via MCTS self-play and writes them to a shared
directory. Periodically pulls fresh model weights from HuggingFace.

Usage:
    python -u -m forge.zeb.worker.run \
        --repo-id username/zeb-42 \
        --output-dir /shared/examples \
        --device cuda
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from forge.zeb.gpu_training_pipeline import (
    GPUTrainingExample,
    create_selfplay_pipeline,
)
from forge.zeb.model import ZebModel
from forge.zeb.example_store import ExampleBatch, save_examples
from forge.zeb.hf import get_remote_step, pull_weights, pull_weights_if_new


def gpu_examples_to_batch(
    examples: GPUTrainingExample,
    worker_id: str,
    model_step: int,
    n_games: int,
) -> ExampleBatch:
    """Convert GPU training examples to a CPU ExampleBatch for serialization."""
    return ExampleBatch(
        observations=examples.observations.cpu(),
        masks=examples.masks.cpu(),
        hand_indices=examples.hand_indices.cpu(),
        hand_masks=examples.hand_masks.cpu(),
        policy_targets=examples.policy_targets.cpu(),
        value_targets=examples.value_targets.cpu(),
        metadata={
            'worker_id': worker_id,
            'model_step': model_step,
            'n_games': n_games,
            'timestamp': time.time(),
        },
    )


def run_worker(args: argparse.Namespace) -> None:
    """Main worker loop: generate games, save examples, sync weights."""
    device = torch.device(args.device)

    # 1. Pull initial model from HuggingFace
    print(f"Pulling initial weights from {args.repo_id}...")
    state_dict, config = pull_weights(args.repo_id, device=device)
    model = ZebModel(**config).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    current_step = get_remote_step(args.repo_id)
    print(f"  Model loaded (step {current_step}), {sum(p.numel() for p in model.parameters()):,} params")

    # 2. Create GPU self-play pipeline
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

    # 3. Main loop
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_count = 0
    total_games = 0
    print(f"\nWorker {args.worker_id} starting (output: {output_dir})")
    print(f"  games_per_batch={args.games_per_batch}, weight_sync_interval={args.weight_sync_interval}")

    while True:
        t0 = time.time()

        # Generate games
        with torch.no_grad():
            examples = pipeline.generate_games_gpu(n_games=args.games_per_batch)

        gen_time = time.time() - t0
        total_games += args.games_per_batch

        # Convert to CPU batch and save
        batch = gpu_examples_to_batch(
            examples, args.worker_id, current_step, n_games=args.games_per_batch,
        )
        save_examples(batch, output_dir, args.worker_id)
        batch_count += 1

        games_per_sec = args.games_per_batch / gen_time
        print(
            f"[{args.worker_id}] batch {batch_count}: "
            f"{examples.n_examples} examples, "
            f"{games_per_sec:.1f} games/s, "
            f"step={current_step}, "
            f"total_games={total_games}"
        )

        # Periodic weight sync
        if batch_count % args.weight_sync_interval == 0:
            result = pull_weights_if_new(args.repo_id, current_step, device=device)
            if result is not None:
                state_dict, _, new_step = result
                model.load_state_dict(state_dict)
                print(f"[{args.worker_id}] Weights updated: step {current_step} -> {new_step}")
                current_step = new_step
            else:
                print(f"[{args.worker_id}] No new weights (still step {current_step})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Self-play worker: generate training examples via MCTS"
    )

    # Required
    parser.add_argument(
        "--repo-id", type=str, required=True,
        help="HuggingFace repo for weight sync (e.g. username/zeb-42)",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Shared directory for writing example batches",
    )

    # Worker identity
    parser.add_argument(
        "--worker-id", type=str, default="worker-0",
        help="Unique identifier for this worker",
    )

    # MCTS parameters
    parser.add_argument("--n-parallel-games", type=int, default=128)
    parser.add_argument("--n-simulations", type=int, default=200)
    parser.add_argument("--max-mcts-nodes", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)

    # Batching
    parser.add_argument(
        "--games-per-batch", type=int, default=256,
        help="Games to generate per batch before saving",
    )
    parser.add_argument(
        "--weight-sync-interval", type=int, default=10,
        help="Pull new weights every N batches",
    )

    # Device
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    run_worker(args)


if __name__ == "__main__":
    main()
