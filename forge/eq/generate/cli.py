"""CLI for E[Q] generation."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from forge.eq.types import ExplorationPolicy

from .pipeline import generate_eq_games_gpu
from .types import AdaptiveConfig, PosteriorConfig


def _parse_bid_values(raw: str | None, *, fallback: int, n_games: int) -> list[int]:
    """Parse one or N comma-separated bid values."""
    if raw:
        values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    else:
        values = [fallback]
    if len(values) == 1:
        values = values * n_games
    if len(values) != n_games:
        raise ValueError(f"Expected 1 or {n_games} bid values, got {len(values)}")
    for value in values:
        if value == 84:
            continue
        if value < 30 or value > 42:
            raise ValueError(f"Bid values must be 30..42, or 84 for take-all contracts (got {value})")
    return values


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

  # Diverse-seed recipe: 10 declarations per seed (matches oracle training).
  # 100 games = 10 seeds x 10 decls.
  python -m forge.eq.generate --start-seed 0 --n-games 100 --n-decl-per-seed 10 \\
      --samples 500
""",
    )

    # Required arguments
    parser.add_argument(
        "--start-seed", type=int, required=True,
        help="Starting seed for deal generation"
    )
    parser.add_argument(
        "--n-games", type=int, required=True,
        help="Total number of (seed, decl) games to generate. "
             "With --n-decl-per-seed N, this must be divisible by N; "
             "the first n_games/N seeds are expanded across the first N decls."
    )
    parser.add_argument(
        "--n-decl-per-seed", type=int, default=1,
        help="Declarations generated per seed (default: 1, backwards-compat). "
             "When >1, each seed is expanded across decl_ids 0..N-1. "
             "n-games must be divisible by this value. Max 10."
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

    # Joint-world tensor (distillation tire-kick; gus/)
    parser.add_argument(
        "--save-joint-worlds", action="store_true",
        help="Save per-world hand layouts and Q-values on each decision record. "
             "Enables distillation of the joint (belief, Q) distribution. "
             "Fixed-sampling path only; skipped in adaptive mode."
    )

    # Schema version
    parser.add_argument(
        "--schema", type=str, choices=["v1", "v2"], default="v1",
        help="Output schema version (default: v1). "
             "v2 adds: bid_value, oracle_softmax_per_seat [4,7], "
             "legal_mask_per_seat [4,7], voids_per_seat [4,3,8]. "
             "Requires --save-joint-worlds (per-seat softmax needs world tensors)."
    )

    # Bid value (used in Schema v2 for per-seat oracle threshold)
    parser.add_argument(
        "--bid-value", type=int, default=30,
        help="Bid value for all games (default: 30 = minimum bid). "
             "Used for p_make thresholds and recorded in Schema v2."
    )
    parser.add_argument(
        "--bid-values", type=str, default=None,
        help="Comma-separated per-game bid values. Length must be 1 or n-games. "
             "Overrides --bid-value. Values must be 30..42, or 84 for take-all."
    )

    # Device
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to run on (default: cuda)"
    )

    args = parser.parse_args()

    # Validate decl-per-seed expansion
    if args.n_decl_per_seed < 1:
        print(f"Error: --n-decl-per-seed must be >= 1 (got {args.n_decl_per_seed}).", flush=True)
        return 1
    if args.n_decl_per_seed > 10:
        print(
            f"Error: --n-decl-per-seed must be <= 10 (got {args.n_decl_per_seed}); "
            f"only 10 distinct declarations exist.",
            flush=True,
        )
        return 1
    if args.n_games % args.n_decl_per_seed != 0:
        print(
            f"Error: --n-games ({args.n_games}) must be divisible by "
            f"--n-decl-per-seed ({args.n_decl_per_seed}).",
            flush=True,
        )
        return 1
    n_seeds = args.n_games // args.n_decl_per_seed

    try:
        parsed_bid_values = _parse_bid_values(
            args.bid_values,
            fallback=args.bid_value,
            n_games=args.n_games,
        )
    except ValueError as exc:
        print(f"Error: {exc}", flush=True)
        return 1

    # Validate schema v2 requires joint worlds
    schema_v2 = args.schema == "v2"
    if schema_v2 and not args.save_joint_worlds:
        print(
            "Warning: --schema v2 requires --save-joint-worlds for per-seat softmax. "
            "Enabling --save-joint-worlds automatically.",
            flush=True,
        )
        args.save_joint_worlds = True

    # Resolve device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        # Tire-kick fallback: try MPS (Apple Silicon), else CPU.
        if torch.backends.mps.is_available():
            print("Warning: CUDA unavailable; falling back to MPS.", flush=True)
            device = "mps"
        else:
            print("Warning: CUDA unavailable; falling back to CPU (slow).", flush=True)
            device = "cpu"

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
        # Default path mirrors seed range actually covered. When expanding
        # across decls, include the decl-per-seed count in the filename.
        end_seed = args.start_seed + n_seeds - 1
        if args.n_decl_per_seed > 1:
            output_path = (
                f"forge/data/eq_pdf_s{args.start_seed}-{end_seed}"
                f"_d{args.n_decl_per_seed}_{args.samples}s.pt"
            )
        else:
            output_path = (
                f"forge/data/eq_pdf_{args.start_seed}-{end_seed}_{args.samples}s.pt"
            )

    # Load model
    from forge.eq.oracle import Stage1Oracle
    print(f"Loading model from {checkpoint_path}...", flush=True)
    oracle = Stage1Oracle(checkpoint_path, device=device, compile=False)

    # Generate deals.
    #
    # Default (n_decl_per_seed=1): one game per seed, decl_ids rotate 0..9.
    #     hands  = [deal(s0), deal(s0+1), ...]
    #     decls  = [0, 1, 2, ...]
    #
    # Expanded (n_decl_per_seed=N): n_seeds seeds, each expanded across N
    # declarations 0..N-1. Matches the oracle's training recipe for state
    # diversity (see gus/PRACTICALITIES.md #8).
    #     hands  = [deal(s0), deal(s0), ..., deal(s0+1), deal(s0+1), ...]
    #     decls  = [0, 1, ..., N-1,       0, 1, ..., N-1,       ...]
    from forge.oracle.rng import deal_from_seed
    if args.n_decl_per_seed == 1:
        hands = [deal_from_seed(args.start_seed + i) for i in range(args.n_games)]
        decl_ids = [i % 10 for i in range(args.n_games)]
    else:
        hands = []
        decl_ids = []
        for seed_offset in range(n_seeds):
            deal = deal_from_seed(args.start_seed + seed_offset)
            for decl in range(args.n_decl_per_seed):
                hands.append(deal)
                decl_ids.append(decl)
        assert len(hands) == args.n_games
        assert len(decl_ids) == args.n_games

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
    if args.n_decl_per_seed > 1:
        print(
            f"Generating {args.n_games} games = "
            f"{n_seeds} seeds ({args.start_seed}..{args.start_seed + n_seeds - 1}) "
            f"x {args.n_decl_per_seed} decls (0..{args.n_decl_per_seed - 1})",
            flush=True,
        )
    else:
        print(
            f"Generating {args.n_games} games "
            f"(seeds {args.start_seed}-{args.start_seed + args.n_games - 1})",
            flush=True,
        )
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
    if args.save_joint_worlds:
        print(f"  Joint-world tensor: saving per-decision (world_hands, q_per_world)", flush=True)
    if schema_v2:
        unique_bids = sorted(set(parsed_bid_values))
        bid_label = unique_bids[0] if len(unique_bids) == 1 else unique_bids
        print(f"  Schema: v2 (bid_value={bid_label}, per-seat oracle softmax)", flush=True)

    # Build per-game bid values when they are recorded (schema v2) or when the
    # user explicitly requested non-default bid-aware action selection.
    bid_values = None
    if schema_v2 or args.bid_values is not None or args.bid_value != 30:
        bid_values = parsed_bid_values

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
        save_joint_worlds=args.save_joint_worlds,
        schema_v2=schema_v2,
        bid_values=bid_values,
    )
    elapsed = time.perf_counter() - t0

    print(f"Generated {len(results)} games in {elapsed:.1f}s ({len(results)/elapsed:.2f} games/s)", flush=True)

    # Save results. `seeds` matches `results` 1:1 — when a seed is reused
    # across decls it appears multiple times, mirroring `hands`/`decl_ids`.
    per_game_seeds = [args.start_seed + (i // args.n_decl_per_seed) for i in range(args.n_games)]
    save_dict = {
        'results': results,
        'seeds': per_game_seeds,
        'decl_ids': decl_ids,
        'start_seed': args.start_seed,
        'n_seeds': n_seeds,
        'n_decl_per_seed': args.n_decl_per_seed,
        'checkpoint': checkpoint_path,
        'enumerate': args.enumerate,
        'posterior': args.posterior,
        'schema': args.schema,
    }
    if bid_values is not None:
        save_dict['bid_values'] = bid_values
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
