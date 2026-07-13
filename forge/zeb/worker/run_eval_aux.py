"""Eval-aux worker for distributed training.

Generates TrainingExamples from E[Q] vs Zeb games and writes them through the
same worker->HF example exchange path used by self-play workers.

Usage:
    # Remote (HF Hub for examples)
    python -u -m forge.zeb.worker.run_eval_aux \
        --repo-id username/zeb-42 \
        --examples-repo-id username/zeb-42-examples \
        --checkpoint forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt \
        --device cuda

    # Local (shared filesystem)
    python -u -m forge.zeb.worker.run_eval_aux \
        --repo-id username/zeb-42 \
        --output-dir /shared/examples \
        --checkpoint forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt \
        --device cuda
"""
from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import torch

from forge.zeb.eq_player import generate_eq_vs_zeb_training_examples
from forge.zeb.eval.loading import DEFAULT_ORACLE, load_oracle
from forge.zeb.example_store import save_examples
from forge.zeb.hf import (
    get_remote_step,
    init_examples_repo,
    pull_weights,
    pull_weights_if_new,
    upload_examples_folder,
)
from forge.zeb.model import ZebModel


def run_worker(args: argparse.Namespace) -> None:
    """Main eval-aux worker loop: generate E[Q] vs Zeb examples, sync weights."""
    device = torch.device(args.device)
    use_hf_examples = bool(args.examples_repo_id)

    # 1) Load oracle model once (fixed during run)
    print(f"Loading oracle checkpoint: {args.checkpoint}")
    oracle = load_oracle(args.checkpoint, args.device)

    # 2) Pull initial Zeb weights from HF
    weights_name = f"{args.weights_name}.pt" if args.weights_name else 'model.pt'
    stem = weights_name.removesuffix('.pt')
    namespace = None if stem == 'model' else stem

    print(f"Pulling initial Zeb weights from {args.repo_id}/{weights_name}...")
    state_dict, config = pull_weights(args.repo_id, device=device, weights_name=weights_name)
    zeb = ZebModel(**config).to(device)
    zeb.load_state_dict(state_dict)
    zeb.eval()
    current_step = get_remote_step(args.repo_id, weights_name=weights_name)
    tokenizer_name = config.get('tokenizer', 'v1')
    print(
        f"  Zeb loaded (step {current_step}), "
        f"{sum(p.numel() for p in zeb.parameters()):,} params, tokenizer={tokenizer_name}"
    )

    # 3) Init examples repo (if using HF)
    if use_hf_examples:
        init_examples_repo(args.examples_repo_id)
        print(f"  Examples repo: {args.examples_repo_id}")

    # 4) Prepare output staging
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    staging_dir = Path(tempfile.mkdtemp(prefix='zeb-evalaux-staging-')) if use_hf_examples else None
    last_upload_time = time.time()
    staged_count = 0
    upload_interval_sec = args.upload_interval

    batch_count = 0
    total_games = 0
    seed_cursor = args.base_seed
    dest = args.examples_repo_id if use_hf_examples else str(output_dir)
    ns_label = f" (namespace: {namespace})" if namespace else ""
    print(f"\nEval-aux worker {args.worker_id} starting (output: {dest}{ns_label})")
    print(
        f"  games_per_batch={args.games_per_batch}, n_samples={args.n_samples}, "
        f"weight_sync_interval={args.weight_sync_interval}"
    )
    if use_hf_examples:
        print(f"  upload_interval={upload_interval_sec}s (folder upload)")

    while True:
        t0 = time.time()

        # Generate E[Q] vs Zeb games, convert to TrainingExamples
        with torch.no_grad():
            examples, eval_stats = generate_eq_vs_zeb_training_examples(
                oracle,
                zeb,
                n_games=args.games_per_batch,
                n_samples=args.n_samples,
                device=args.device,
                zeb_temperature=args.zeb_temperature,
                base_seed=seed_cursor,
                batch_size=args.eval_batch_size,
                include_eq_pdf_policy=args.eq_policy_targets,
            )

        seed_cursor += args.games_per_batch
        gen_time = time.time() - t0
        total_games += args.games_per_batch

        batch = examples.cpu()
        batch.metadata = {
            'worker_id': args.worker_id,
            'model_step': current_step,
            'n_games': args.games_per_batch,
            'timestamp': time.time(),
            'source': 'eval-eq-zeb',
            'model_checkpoint': str(args.repo_id),
            'oracle_checkpoint': str(args.checkpoint),
            'eval_config': {
                'n_samples': args.n_samples,
                'zeb_temperature': args.zeb_temperature,
                'eq_policy_targets': bool(args.eq_policy_targets),
                'policy_target_kind': 'eq-pdf' if args.eq_policy_targets else 'actor-onehot',
            },
            'eval_stats': {
                'eq_win_rate': eval_stats['eq_win_rate'],
                'avg_margin': eval_stats['avg_margin'],
            },
        }

        if use_hf_examples:
            save_examples(batch, staging_dir, args.worker_id)
            staged_count += 1
        else:
            save_examples(batch, output_dir, args.worker_id)

        batch_count += 1
        games_per_sec = args.games_per_batch / max(gen_time, 1e-6)
        print(
            f"[{args.worker_id}] batch {batch_count}: "
            f"{batch.n_examples} examples, "
            f"{games_per_sec:.2f} games/s, "
            f"eq_win={eval_stats['eq_win_rate']:.1%}, "
            f"step={current_step}, "
            f"total_games={total_games}"
        )

        # Flush staged examples to HF
        if use_hf_examples and (time.time() - last_upload_time) >= upload_interval_sec:
            print(f"[{args.worker_id}] Uploading {staged_count} batches to HF...")
            try:
                upload_examples_folder(args.examples_repo_id, staging_dir, n_files=staged_count, namespace=namespace)
                for f in staging_dir.iterdir():
                    f.unlink()
                print(f"[{args.worker_id}] Upload complete ({staged_count} batches)")
                staged_count = 0
            except Exception as e:
                print(f"[{args.worker_id}] Upload failed ({e}), will retry next cycle")
            last_upload_time = time.time()

        # Periodic weight sync
        if batch_count % args.weight_sync_interval == 0:
            result = pull_weights_if_new(args.repo_id, current_step, device=device, weights_name=weights_name)
            if result is not None:
                state_dict, _, new_step = result
                zeb.load_state_dict(state_dict)
                zeb.eval()
                print(f"[{args.worker_id}] Zeb weights updated: step {current_step} -> {new_step}")
                current_step = new_step
            else:
                print(f"[{args.worker_id}] No new weights (still step {current_step})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Eval-aux worker: generate E[Q] vs Zeb training examples'
    )

    # Required
    parser.add_argument(
        '--repo-id', type=str, required=True,
        help='HuggingFace repo for Zeb weight sync (e.g. username/zeb-42)',
    )

    # Oracle checkpoint
    parser.add_argument(
        '--checkpoint', type=str, default=DEFAULT_ORACLE,
        help='Stage 1 oracle checkpoint used by E[Q]',
    )

    # Example output (one or both)
    parser.add_argument(
        '--examples-repo-id', type=str, default=None,
        help='HF repo for example exchange (e.g. username/zeb-42-examples)',
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Local directory for writing example batches (fallback if no --examples-repo-id)',
    )

    # Worker identity
    parser.add_argument(
        '--worker-id', type=str, default='eval-worker-0',
        help='Unique identifier for this worker',
    )

    # Eval configuration
    parser.add_argument('--n-samples', type=int, default=100,
                        help='World samples per E[Q] decision')
    parser.add_argument('--games-per-batch', type=int, default=128,
                        help='Games to generate per batch before saving')
    parser.add_argument('--eval-batch-size', type=int, default=0,
                        help='Chunk size within one games batch (0=all-at-once)')
    parser.add_argument('--zeb-temperature', type=float, default=0.1)
    parser.add_argument('--eq-policy-targets', action=argparse.BooleanOptionalAction, default=False,
                        help='Store E[Q] policy PDFs instead of actor one-hot targets')
    parser.add_argument('--base-seed', type=int, default=0,
                        help='Seed offset for deterministic deal generation')

    # Sync and upload cadence
    parser.add_argument(
        '--weight-sync-interval', type=int, default=5,
        help='Pull new Zeb weights every N batches',
    )
    parser.add_argument(
        '--upload-interval', type=int, default=240,
        help='Seconds between HF folder uploads',
    )

    # Model namespace
    parser.add_argument(
        '--weights-name', type=str, default=None,
        help='Weights filename stem on HF (e.g. large -> large.pt)',
    )

    # Device
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()
    if not args.examples_repo_id and not args.output_dir:
        parser.error('Either --examples-repo-id or --output-dir is required')
    run_worker(args)


if __name__ == '__main__':
    main()
