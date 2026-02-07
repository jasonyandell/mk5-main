"""CLI for E[Q] generation."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from forge.eq.types import ExplorationPolicy

from .pipeline import generate_eq_games_gpu
from .types import AdaptiveConfig, PosteriorConfig


def main() -> int:
    """CLI for E[Q] generation."""
    parser = argparse.ArgumentParser(
        description="Generate E[Q] training data using GPU pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 5 games starting at seed 1000 with 1000 samples
  python -m forge.eq.generate --start-seed 1000 --n-games 5 --samples 1000

  # Generate to specific output file
  python -m forge.eq.generate --start-seed 0 --n-games 100 --samples 500 -o data.pt

  # Use exact enumeration for late-game positions
  python -m forge.eq.generate --start-seed 0 --n-games 10 --enumerate

  # Enable posterior weighting
  python -m forge.eq.generate --start-seed 0 --n-games 10 --posterior

  # Use adaptive convergence-based sampling (samples until SEM < 0.5)
  python -m forge.eq.generate --start-seed 0 --n-games 10 --adaptive

  # Adaptive with custom thresholds (tighter convergence)
  python -m forge.eq.generate --start-seed 0 --n-games 10 --adaptive \\
      --min-samples 100 --max-samples 5000 --sem-threshold 0.3
""",
    )

    # Required arguments
    parser.add_argument(
        "--start-seed", type=int, required=True,
        help="Starting seed for deal generation"
    )
    parser.add_argument(
        "--n-games", type=int, required=True,
        help="Number of games to generate"
    )

    # Quality parameters
    parser.add_argument(
        "--samples", type=int, default=1000,
        help="Number of world samples per decision (default: 1000)"
    )

    # Output
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output file path. Default: forge/data/eq_pdf_{start}-{end}_{samples}s.pt"
    )

    # Model
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint. Default: auto-detect best model"
    )

    # Optional features
    parser.add_argument(
        "--enumerate", action="store_true",
        help="Use exact enumeration for late-game positions (slower but exact)"
    )
    parser.add_argument(
        "--enum-threshold", type=int, default=100_000,
        help="Max worlds to enumerate before falling back to sampling (default: 100000)"
    )
    parser.add_argument(
        "--posterior", action="store_true",
        help="Enable posterior weighting using past play history"
    )
    parser.add_argument(
        "--posterior-k", type=int, default=4,
        help="Window size for posterior weighting (default: 4)"
    )

    # Exploration
    parser.add_argument(
        "--exploration", type=str, choices=["none", "boltzmann", "epsilon"],
        default="none", help="Exploration policy (default: none = greedy)"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Temperature for Boltzmann exploration (default: 1.0)"
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.1,
        help="Epsilon for epsilon-greedy exploration (default: 0.1)"
    )

    # Adaptive sampling
    parser.add_argument(
        "--adaptive", action="store_true",
        help="Use adaptive convergence-based sampling instead of fixed sample count"
    )
    parser.add_argument(
        "--min-samples", type=int, default=50,
        help="Minimum samples before checking convergence (default: 50)"
    )
    parser.add_argument(
        "--max-samples", type=int, default=2000,
        help="Maximum samples (hard cap) for adaptive sampling (default: 2000)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=50,
        help="Samples to add per iteration in adaptive mode (default: 50)"
    )
    parser.add_argument(
        "--sem-threshold", type=float, default=0.5,
        help="SEM threshold for convergence in Q-value points (default: 0.5)"
    )

    # Device
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to run on (default: cuda)"
    )

    args = parser.parse_args()

    # Resolve device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("Error: CUDA not available. GPU-only pipeline requires CUDA.", flush=True)
        return 1

    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        # Auto-detect best model
        model_dir = Path(__file__).parent.parent.parent / "models"
        candidates = [
            model_dir / "domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt",
            model_dir / "domino-large-817k-valuehead-acc97.8-qgap0.07.ckpt",
            Path("checkpoints/stage1/best.ckpt"),
        ]
        checkpoint_path = None
        for path in candidates:
            if path.exists():
                checkpoint_path = str(path)
                break
        if checkpoint_path is None:
            print("Error: No model checkpoint found. Use --checkpoint to specify.", flush=True)
            return 1

    # Resolve output path
    if args.output:
        output_path = args.output
    else:
        end_seed = args.start_seed + args.n_games - 1
        output_path = f"forge/data/eq_pdf_{args.start_seed}-{end_seed}_{args.samples}s.pt"

    # Load model
    from forge.eq.oracle import Stage1Oracle
    print(f"Loading model from {checkpoint_path}...", flush=True)
    oracle = Stage1Oracle(checkpoint_path, device=device, compile=False)

    # Generate deals
    from forge.oracle.rng import deal_from_seed
    hands = [deal_from_seed(args.start_seed + i) for i in range(args.n_games)]
    decl_ids = [i % 10 for i in range(args.n_games)]

    # Configure posterior
    posterior_config = None
    if args.posterior:
        posterior_config = PosteriorConfig(
            enabled=True,
            window_k=args.posterior_k,
            tau=0.1,
            uniform_mix=0.1,
        )

    # Configure exploration
    exploration_policy = None
    if args.exploration == "boltzmann":
        exploration_policy = ExplorationPolicy.boltzmann(temperature=args.temperature)
    elif args.exploration == "epsilon":
        exploration_policy = ExplorationPolicy.epsilon_greedy(epsilon=args.epsilon)

    # Configure adaptive sampling
    adaptive_config = None
    if args.adaptive:
        adaptive_config = AdaptiveConfig(
            enabled=True,
            min_samples=args.min_samples,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            sem_threshold=args.sem_threshold,
        )

    # Run generation
    print(f"Generating {args.n_games} games (seeds {args.start_seed}-{args.start_seed + args.n_games - 1})", flush=True)
    if args.adaptive:
        print(f"  Adaptive: enabled (min={args.min_samples}, max={args.max_samples}, "
              f"batch={args.batch_size}, SEM<{args.sem_threshold})", flush=True)
    else:
        print(f"  Samples: {args.samples}", flush=True)
    print(f"  Device: {device}", flush=True)
    if args.enumerate:
        print(f"  Enumeration: enabled (threshold={args.enum_threshold})", flush=True)
    if args.posterior:
        print(f"  Posterior: enabled (k={args.posterior_k})", flush=True)
    if exploration_policy:
        print(f"  Exploration: {args.exploration}", flush=True)

    t0 = time.perf_counter()
    results = generate_eq_games_gpu(
        model=oracle.model,
        hands=hands,
        decl_ids=decl_ids,
        n_samples=args.samples,
        device=device,
        greedy=(exploration_policy is None),
        exploration_policy=exploration_policy,
        posterior_config=posterior_config,
        use_enumeration=args.enumerate,
        enumeration_threshold=args.enum_threshold,
        adaptive_config=adaptive_config,
    )
    elapsed = time.perf_counter() - t0

    print(f"Generated {len(results)} games in {elapsed:.1f}s ({len(results)/elapsed:.2f} games/s)", flush=True)

    # Save results
    save_dict = {
        'results': results,
        'seeds': list(range(args.start_seed, args.start_seed + args.n_games)),
        'checkpoint': checkpoint_path,
        'enumerate': args.enumerate,
        'posterior': args.posterior,
    }
    if args.adaptive:
        save_dict['adaptive'] = True
        save_dict['adaptive_config'] = {
            'min_samples': args.min_samples,
            'max_samples': args.max_samples,
            'batch_size': args.batch_size,
            'sem_threshold': args.sem_threshold,
        }
    else:
        save_dict['n_samples'] = args.samples
    torch.save(save_dict, output_path)
    print(f"Saved to {output_path}", flush=True)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
