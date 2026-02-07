#!/usr/bin/env python3
"""
Flywheel: Iterative fine-tuning pipeline for Claude Code.

Reads state from forge/flywheel/state.yaml, runs one iteration,
updates state with results. Designed for token-efficient operation
by Claude Code agents.

W&B Integration:
- Each iteration is one W&B run in a group
- Checkpoint artifacts with lineage (baseline → iter-1 → iter-2)
- Custom x-axis: plot metrics by total_seeds_trained
- Summary metrics for easy comparison in runs table

Usage:
    # Run next iteration
    python -m forge.cli.flywheel

    # Check status
    python -m forge.cli.flywheel status

    # Initialize new flywheel
    python -m forge.cli.flywheel init --wandb-group my-exp --start-seed 200

    # Run with specific seed range (overrides state.yaml)
    python -m forge.cli.flywheel --seed-range 210:220
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Paths
FLYWHEEL_DIR = Path(__file__).parent.parent / "flywheel"
STATE_FILE = FLYWHEEL_DIR / "state.yaml"
DEFAULT_BASELINE = "forge/models/domino-large-817k-valuehead-acc97.8-qgap0.07.ckpt"

# Golden val/test seed ranges - MUST match original training data exactly
# Original data/shards/ has: val=900-902 (30 shards), test=950 (10 shards)
# Oracle generation is deterministic, so regenerating produces binary identical data
GOLDEN_VAL_SEEDS = range(900, 903)   # seeds 900-902 → val split (30 shards, ~1.5M samples)
GOLDEN_TEST_SEEDS = range(950, 951)  # seed 950 only → test split (10 shards, ~500K samples)
ALL_DECLS = range(10)                # 0-9: blanks, ones, twos, ..., nines


def load_state() -> dict[str, Any]:
    """Load state from state.yaml."""
    if not STATE_FILE.exists():
        print(f"ERROR: State file not found: {STATE_FILE}")
        print("Run: python -m forge.cli.flywheel init --wandb-group <name>")
        sys.exit(1)
    with open(STATE_FILE) as f:
        return yaml.safe_load(f)


def save_state(state: dict[str, Any]) -> None:
    """Save state to state.yaml."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        yaml.dump(state, f, default_flow_style=False, sort_keys=False)


def parse_seed_range(seed_range: str) -> tuple[int, int]:
    """Parse seed range like '200:210' into (200, 210)."""
    start, end = seed_range.split(":")
    return int(start), int(end)


def next_seed_range(current: str, step: int = 10) -> str:
    """Get next seed range after current."""
    _, end = parse_seed_range(current)
    return f"{end}:{end + step}"


def compute_total_seeds(state: dict[str, Any], current_end: int) -> int:
    """Compute total seeds trained including current iteration."""
    # Start from baseline (200 seeds trained initially)
    baseline_seeds = 200

    # Add seeds from history
    history_seeds = 0
    for h in state.get("history", []):
        start, end = parse_seed_range(h["seed_range"])
        history_seeds += end - start

    # Add current iteration
    current_start, _ = parse_seed_range(state["current"]["seed_range"])
    current_seeds = current_end - current_start

    return baseline_seeds + history_seeds + current_seeds


def run_command(cmd: list[str], desc: str) -> subprocess.CompletedProcess:
    """Run a command with logging."""
    print(f"\n{'='*60}")
    print(f"[FLYWHEEL] {desc}")
    print(f"[FLYWHEEL] Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"{desc} failed with code {result.returncode}")
    return result


def ensure_golden_val_test(shards_dir: Path, verbose: bool = True) -> bool:
    """Ensure golden val/test shards exist, generating any that are missing.

    Golden shards provide a consistent benchmark for comparing flywheel iterations.
    These MUST match the original training data exactly for apples-to-apples comparison.
    Val: seeds 900-902 with all 10 decls (30 shards, ~1.5M samples)
    Test: seed 950 with all 10 decls (10 shards, ~500K samples)

    Args:
        shards_dir: Directory containing parquet shards (e.g., data/flywheel-shards)
        verbose: Print progress messages

    Returns:
        True if all golden shards now exist, False on generation failure
    """
    shards_dir = Path(shards_dir)
    shards_dir.mkdir(parents=True, exist_ok=True)

    # Collect missing shards
    missing = []
    for seed in list(GOLDEN_VAL_SEEDS) + list(GOLDEN_TEST_SEEDS):
        for decl in ALL_DECLS:
            shard_path = shards_dir / f"seed_{seed:08d}_decl_{decl}.parquet"
            if not shard_path.exists():
                missing.append((seed, decl))

    if not missing:
        if verbose:
            print("[FLYWHEEL] Golden val/test shards already exist (40 shards)")
        return True

    # Report what we need to generate
    val_missing = sum(1 for s, _ in missing if s in GOLDEN_VAL_SEEDS)
    test_missing = sum(1 for s, _ in missing if s in GOLDEN_TEST_SEEDS)

    if verbose:
        print(f"\n{'='*60}")
        print("[FLYWHEEL] Generating missing golden val/test shards")
        print(f"           Val shards needed:  {val_missing}/30")
        print(f"           Test shards needed: {test_missing}/10")
        print(f"{'='*60}")

    # Generate missing shards
    for i, (seed, decl) in enumerate(missing):
        split_name = "val" if seed in GOLDEN_VAL_SEEDS else "test"
        try:
            run_command([
                "python", "-m", "forge.oracle.generate",
                "--seed", str(seed),
                "--decl", str(decl),
                "--out", str(shards_dir),
            ], f"Generating golden {split_name} shard [{i+1}/{len(missing)}]: seed {seed} decl {decl}")
        except RuntimeError as e:
            print(f"[FLYWHEEL] ERROR: Failed to generate golden shard: {e}")
            return False

    if verbose:
        print(f"\n[FLYWHEEL] Generated {len(missing)} golden shards successfully")

    return True


def extract_metrics_from_csv(runs_dir: Path) -> dict[str, float | None]:
    """Extract final metrics from Lightning CSV logs."""
    metrics = {"q_gap": None, "accuracy": None, "val_loss": None}

    versions = sorted(runs_dir.glob("version_*"), key=lambda p: p.stat().st_mtime)
    if not versions:
        return metrics

    latest = versions[-1]
    csv_path = latest / "metrics.csv"

    if not csv_path.exists():
        return metrics

    try:
        import csv
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            return metrics

        # Get last row with validation metrics
        for row in reversed(rows):
            if row.get("val/q_gap"):
                metrics["q_gap"] = float(row["val/q_gap"])
            if row.get("val/accuracy"):
                metrics["accuracy"] = float(row["val/accuracy"])
            if row.get("val/loss"):
                metrics["val_loss"] = float(row["val/loss"])
            if metrics["q_gap"] is not None:
                break
    except Exception as e:
        print(f"[FLYWHEEL] Warning: Could not extract metrics from CSV: {e}")

    return metrics


def cmd_status(args: argparse.Namespace) -> int:
    """Show current flywheel status."""
    state = load_state()

    print(f"\n{'='*50}")
    print("FLYWHEEL STATUS")
    print(f"{'='*50}")
    print(f"Status:       {state['status']}")
    print(f"W&B Group:    {state['wandb_group']}")
    print(f"Baseline:     {state['baseline_checkpoint']}")

    if state.get('current'):
        print(f"\nCurrent iteration:")
        print(f"  Seed range: {state['current'].get('seed_range', 'N/A')}")
        print(f"  Checkpoint: {state['current'].get('checkpoint', 'N/A')}")
        print(f"  Started:    {state['current'].get('started_at', 'N/A')}")

    if state.get('last_error'):
        print(f"\nLast error: {state['last_error']}")

    print(f"\nNext action: {state.get('next_action', 'N/A')}")

    if state.get('history'):
        print(f"\nHistory ({len(state['history'])} iterations):")
        for i, h in enumerate(state['history'][-5:]):  # Last 5
            q_gap = h.get('q_gap')
            acc = h.get('accuracy')
            q_str = f"{q_gap:.3f}" if q_gap is not None else "N/A"
            acc_str = f"{acc * 100:.2f}%" if acc is not None else "N/A"
            print(f"  {i+1}. seeds {h['seed_range']}: q_gap={q_str}, acc={acc_str}")

    print(f"{'='*50}\n")
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    """Initialize a new flywheel."""
    if STATE_FILE.exists() and not args.force:
        print(f"State file already exists: {STATE_FILE}")
        print("Use --force to overwrite")
        return 1

    baseline = args.baseline or DEFAULT_BASELINE
    if not Path(baseline).exists():
        print(f"WARNING: Baseline checkpoint not found: {baseline}")
        print("Training will start from scratch if not fixed.")

    state = {
        "status": "ready",
        "wandb_group": args.wandb_group,
        "baseline_checkpoint": baseline,
        "current": {
            "seed_range": f"{args.start_seed}:{args.start_seed + args.seeds_per_iter}",
            "checkpoint": None,
            "started_at": None,
        },
        "last_error": None,
        "next_action": "Run: python -m forge.cli.flywheel",
        "history": [],
    }

    save_state(state)
    print(f"Initialized flywheel state at {STATE_FILE}")
    print(f"  W&B group:    {args.wandb_group}")
    print(f"  Start seed:   {args.start_seed}")
    print(f"  Seeds/iter:   {args.seeds_per_iter}")
    print(f"  Baseline:     {baseline}")
    print(f"\nRun: python -m forge.cli.flywheel")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Run one flywheel iteration with W&B tracking."""
    state = load_state()

    # Check status
    if state["status"] == "running":
        print("ERROR: Flywheel is already running.")
        print("If stuck, set status to 'failed' in state.yaml")
        return 1

    if state["status"] == "done":
        print("Flywheel is done. Nothing to do.")
        return 0

    if state["status"] == "failed":
        print("ERROR: Flywheel is in failed state.")
        print(f"Last error: {state.get('last_error', 'unknown')}")
        print("Fix the issue and set status to 'ready' to retry.")
        return 1

    # Get seed range (from args or state)
    if args.seed_range:
        seed_range = args.seed_range
        state["current"]["seed_range"] = seed_range
    else:
        seed_range = state["current"]["seed_range"]

    start_seed, end_seed = parse_seed_range(seed_range)
    iteration_num = len(state["history"]) + 1
    total_seeds = compute_total_seeds(state, end_seed)

    # Determine checkpoint to resume from
    if state["history"]:
        resume_from = state["history"][-1]["checkpoint"]
        parent_artifact_name = f"{state['wandb_group']}-checkpoint"
    else:
        resume_from = state["baseline_checkpoint"]
        parent_artifact_name = None  # Baseline, no artifact yet

    # Update state to running
    state["status"] = "running"
    state["current"]["started_at"] = datetime.now(timezone.utc).isoformat()
    state["last_error"] = None
    state["next_action"] = "Waiting for iteration to complete..."
    save_state(state)

    # Initialize W&B run for this iteration
    wandb_run = None
    if WANDB_AVAILABLE and not args.no_wandb:
        run_name = f"iter-{iteration_num:03d}-seeds-{start_seed}-{end_seed}"

        wandb_run = wandb.init(
            project="crystal-forge",
            group=state["wandb_group"],
            name=run_name,
            job_type="flywheel",
            tags=["flywheel", state["wandb_group"].split("-")[0]],
            dir="runs",  # Consolidate all wandb logs in runs/wandb/
            config={
                "flywheel_iteration": iteration_num,
                "seed_range": seed_range,
                "start_seed": start_seed,
                "end_seed": end_seed,
                "total_seeds_trained": total_seeds,
                "parent_checkpoint": resume_from,
                "epochs": args.epochs,
                "model_size": "817K",
                "embed_dim": 128,
                "n_heads": 8,
                "n_layers": 4,
                "ff_dim": 512,
            },
        )

        # Define custom x-axis for plotting metrics by total seeds
        wandb.define_metric("total_seeds")
        wandb.define_metric("final/*", step_metric="total_seeds")

        # Log parent artifact for lineage (if not baseline)
        if parent_artifact_name and resume_from:
            try:
                parent_artifact = wandb_run.use_artifact(
                    f"{parent_artifact_name}:latest"
                )
                print(f"[FLYWHEEL] Linked parent artifact: {parent_artifact.name}")
            except wandb.errors.CommError:
                print(f"[FLYWHEEL] Note: No existing artifact for {parent_artifact_name}")

    try:
        # Step 0: Ensure golden val/test shards exist for consistent benchmarking
        if not ensure_golden_val_test(Path("data/flywheel-shards")):
            raise RuntimeError("Failed to generate golden val/test shards")

        if wandb_run:
            wandb.log({"stage": 0, "stage_name": "golden_val_test"})

        # Step 1: Generate oracle shards (1 decl per seed, decl = seed % 10)
        for seed in range(start_seed, end_seed):
            decl = seed % 10
            run_command([
                "python", "-m", "forge.oracle.generate",
                "--seed", str(seed),
                "--decl", str(decl),
                "--out", "data/flywheel-shards",
            ], f"Generating shard for seed {seed} decl {decl}")

        if wandb_run:
            wandb.log({"stage": 1, "stage_name": "generate"})

        # Step 2: Tokenize flywheel shards only
        run_command([
            "python", "-m", "forge.cli.tokenize",
            "--input", "data/flywheel-shards",
            "--output", "data/flywheel-tokenized",
            "--force",  # Re-tokenize all flywheel data
        ], "Tokenizing data")

        if wandb_run:
            wandb.log({"stage": 2, "stage_name": "tokenize"})

        # Step 3: Train (fine-tune from previous checkpoint)
        train_cmd = [
            "python", "-m", "forge.cli.train",
            "--data", "data/flywheel-tokenized",
            "--epochs", str(args.epochs),
            "--embed-dim", "128",
            "--n-heads", "8",
            "--n-layers", "4",
            "--ff-dim", "512",
            "--lr", "3e-5",  # Lower LR for fine-tuning (vs 3e-4 for training from scratch)
            "--wandb",
            "--wandb-group", state["wandb_group"],
        ]
        if resume_from and Path(resume_from).exists():
            train_cmd.extend(["--resume", resume_from])

        run_command(train_cmd, "Training model")

        if wandb_run:
            wandb.log({"stage": 3, "stage_name": "train"})

        # Step 4: Find the new checkpoint
        runs_dir = Path("runs/domino")
        versions = sorted(runs_dir.glob("version_*"), key=lambda p: p.stat().st_mtime)
        new_checkpoint = None
        if versions:
            latest = versions[-1]
            checkpoints = list((latest / "checkpoints").glob("*.ckpt"))
            if checkpoints:
                # Prefer 'last.ckpt' if exists
                last_ckpt = latest / "checkpoints" / "last.ckpt"
                new_checkpoint = str(last_ckpt if last_ckpt.exists() else checkpoints[-1])

        # Step 5: Extract metrics from training
        metrics = extract_metrics_from_csv(runs_dir)

        # Step 6: Log final metrics and artifact to W&B
        if wandb_run:
            # Log final metrics with total_seeds as x-axis
            final_metrics = {
                "total_seeds": total_seeds,
                "final/q_gap": metrics["q_gap"],
                "final/accuracy": metrics["accuracy"],
                "final/val_loss": metrics["val_loss"],
            }
            # Filter out None values
            final_metrics = {k: v for k, v in final_metrics.items() if v is not None}
            wandb.log(final_metrics)

            # Update summary for runs table
            wandb_run.summary["total_seeds"] = total_seeds
            wandb_run.summary["iteration"] = iteration_num
            if metrics["q_gap"] is not None:
                wandb_run.summary["final_q_gap"] = metrics["q_gap"]
            if metrics["accuracy"] is not None:
                wandb_run.summary["final_accuracy"] = metrics["accuracy"]

            # Log checkpoint as artifact with lineage
            if new_checkpoint and Path(new_checkpoint).exists():
                artifact = wandb.Artifact(
                    name=f"{state['wandb_group']}-checkpoint",
                    type="model",
                    description=f"Flywheel iteration {iteration_num}, seeds {seed_range}",
                    metadata={
                        "iteration": iteration_num,
                        "seed_range": seed_range,
                        "total_seeds": total_seeds,
                        "q_gap": metrics["q_gap"],
                        "accuracy": metrics["accuracy"],
                    },
                )
                artifact.add_file(new_checkpoint)
                wandb_run.log_artifact(
                    artifact,
                    aliases=["latest", f"iter-{iteration_num:03d}"]
                )
                print(f"[FLYWHEEL] Logged checkpoint artifact: {artifact.name}")

            wandb.finish()

        # Update state with success
        history_entry = {
            "seed_range": seed_range,
            "checkpoint": new_checkpoint,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "total_seeds": total_seeds,
            **metrics,
        }
        state["history"].append(history_entry)
        state["current"] = {
            "seed_range": next_seed_range(seed_range, args.seeds_per_iter),
            "checkpoint": None,
            "started_at": None,
        }
        state["status"] = "ready"
        state["next_action"] = "Run: python -m forge.cli.flywheel"
        save_state(state)

        # Check if this is a new best
        is_new_best = False
        best_q_gap = None
        baseline_q_gap = 0.072  # Large v2 baseline

        if metrics["q_gap"] is not None:
            # Find best from history (excluding current)
            prev_q_gaps = [h.get("q_gap") for h in state["history"][:-1] if h.get("q_gap") is not None]
            if prev_q_gaps:
                best_q_gap = min(prev_q_gaps)
                is_new_best = metrics["q_gap"] < best_q_gap
            else:
                # First iteration - compare to baseline
                is_new_best = metrics["q_gap"] < baseline_q_gap
                best_q_gap = baseline_q_gap

        print(f"\n{'='*60}")
        print("[FLYWHEEL] Iteration complete!")
        print(f"  Iteration:    {iteration_num}")
        print(f"  Seeds:        {seed_range}")
        print(f"  Total seeds:  {total_seeds}")
        print(f"  Checkpoint:   {new_checkpoint}")
        if metrics["q_gap"] is not None:
            print(f"  Q-gap:        {metrics['q_gap']:.4f}")
        if metrics["accuracy"] is not None:
            print(f"  Accuracy:     {metrics['accuracy'] * 100:.2f}%")
        print(f"  Next:         {state['current']['seed_range']}")
        print(f"{'='*60}\n")

        # Print promotion instructions if new best
        if is_new_best and new_checkpoint:
            q_gap_str = f"{metrics['q_gap']:.3f}" if metrics['q_gap'] else "unknown"
            acc_str = f"{metrics['accuracy']:.1f}" if metrics['accuracy'] else "unknown"
            print(f"{'='*60}")
            print("[FLYWHEEL] NEW BEST MODEL FOUND!")
            print(f"  Previous best q_gap: {best_q_gap:.4f}")
            print(f"  New best q_gap:      {metrics['q_gap']:.4f}")
            print(f"")
            print("To promote this model to forge/models/:")
            print(f"")
            print(f"  cp {new_checkpoint} \\")
            print(f"     forge/models/domino-large-817k-flywheel-qgap{q_gap_str}-acc{acc_str}.ckpt")
            print(f"")
            print("Then update forge/models/README.md with the new model details.")
            print(f"{'='*60}\n")

        return 0

    except Exception as e:
        # Update state with failure
        state["status"] = "failed"
        state["last_error"] = str(e)
        state["next_action"] = f"Fix error and set status to 'ready': {e}"
        save_state(state)

        if wandb_run:
            wandb.finish(exit_code=1)

        print(f"\n{'='*60}")
        print(f"[FLYWHEEL] ERROR: {e}")
        print(f"{'='*60}\n")

        return 1


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Flywheel: Iterative fine-tuning pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # status command
    status_parser = subparsers.add_parser("status", help="Show flywheel status")
    status_parser.set_defaults(func=cmd_status)

    # init command
    init_parser = subparsers.add_parser("init", help="Initialize new flywheel")
    init_parser.add_argument("--wandb-group", required=True, help="W&B group name")
    init_parser.add_argument("--start-seed", type=int, default=200, help="Starting seed")
    init_parser.add_argument("--seeds-per-iter", type=int, default=10, help="Seeds per iteration")
    init_parser.add_argument("--baseline", type=str, help="Baseline checkpoint path")
    init_parser.add_argument("--force", action="store_true", help="Overwrite existing state")
    init_parser.set_defaults(func=cmd_init)

    # Default run command (no subcommand)
    parser.add_argument("--seed-range", type=str, help="Override seed range (e.g., 200:210)")
    parser.add_argument("--epochs", type=int, default=2, help="Epochs per iteration")
    parser.add_argument("--seeds-per-iter", type=int, default=10, help="Seeds per iteration")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--once", action="store_true", help="Run only one iteration (default: run continuously)")

    args = parser.parse_args()

    if args.command:
        return args.func(args)
    else:
        # Run continuously until interrupted or error
        if args.once:
            return cmd_run(args)
        else:
            print("[FLYWHEEL] Running continuously. Press Ctrl+C to stop.")
            iteration = 0
            while True:
                iteration += 1
                print(f"\n{'#'*60}")
                print(f"[FLYWHEEL] Starting iteration {iteration}")
                print(f"{'#'*60}\n")
                result = cmd_run(args)
                if result != 0:
                    print(f"[FLYWHEEL] Iteration failed with code {result}. Stopping.")
                    return result
                # Small delay between iterations
                import time
                time.sleep(2)


if __name__ == "__main__":
    sys.exit(main())
