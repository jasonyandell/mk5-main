#!/usr/bin/env python3
"""
Flywheel: Iterative fine-tuning pipeline for Claude Code.

Reads state from forge/flywheel/state.yaml, runs one iteration,
updates state with results. Designed for token-efficient operation
by Claude Code agents.

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
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

# Paths
FLYWHEEL_DIR = Path(__file__).parent.parent / "flywheel"
STATE_FILE = FLYWHEEL_DIR / "state.yaml"
DEFAULT_BASELINE = "forge/models/domino-large-817k-valuehead-acc97.8-qgap0.07.ckpt"


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


def cmd_status(args: argparse.Namespace) -> int:
    """Show current flywheel status."""
    state = load_state()

    print(f"\n{'='*50}")
    print(f"FLYWHEEL STATUS")
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
            print(f"  {i+1}. seeds {h['seed_range']}: q_gap={h.get('q_gap', 'N/A')}, acc={h.get('accuracy', 'N/A')}")

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
    """Run one flywheel iteration."""
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

    # Determine checkpoint to resume from
    if state["history"]:
        resume_from = state["history"][-1]["checkpoint"]
    else:
        resume_from = state["baseline_checkpoint"]

    # Update state to running
    state["status"] = "running"
    state["current"]["started_at"] = datetime.now(timezone.utc).isoformat()
    state["last_error"] = None
    state["next_action"] = "Waiting for iteration to complete..."
    save_state(state)

    try:
        # Step 1: Generate oracle shards
        run_command([
            "python", "-m", "forge.oracle.generate",
            "--seed-range", f"{start_seed}:{end_seed}",
            "--decl", "auto",  # 1 decl per seed
            "--out", "data/shards",
        ], f"Generating oracle shards for seeds {start_seed}:{end_seed}")

        # Step 2: Tokenize (append to existing)
        run_command([
            "python", "-m", "forge.cli.tokenize",
            "--input", "data/shards",
            "--output", "data/tokenized",
            "--force",  # Re-tokenize all
        ], "Tokenizing data")

        # Step 3: Train
        train_cmd = [
            "python", "-m", "forge.cli.train",
            "--data", "data/tokenized",
            "--epochs", str(args.epochs),
            "--embed-dim", "128",
            "--n-heads", "8",
            "--n-layers", "4",
            "--ff-dim", "512",
            "--wandb",
            "--wandb-group", state["wandb_group"],
        ]
        if resume_from and Path(resume_from).exists():
            # TODO: Add --resume flag to train.py when implemented
            pass

        run_command(train_cmd, "Training model")

        # Step 4: Find the new checkpoint
        runs_dir = Path("runs/domino")
        versions = sorted(runs_dir.glob("version_*"), key=lambda p: p.stat().st_mtime)
        if versions:
            latest = versions[-1]
            checkpoints = list((latest / "checkpoints").glob("*.ckpt"))
            if checkpoints:
                # Prefer 'last.ckpt' if exists
                last_ckpt = latest / "checkpoints" / "last.ckpt"
                new_checkpoint = str(last_ckpt if last_ckpt.exists() else checkpoints[-1])
            else:
                new_checkpoint = None
        else:
            new_checkpoint = None

        # Step 5: Get metrics from latest run (parse from logs or wandb)
        # For now, placeholder values - can be improved
        metrics = {
            "q_gap": None,
            "accuracy": None,
            "val_loss": None,
        }

        # Update state with success
        history_entry = {
            "seed_range": seed_range,
            "checkpoint": new_checkpoint,
            "completed_at": datetime.now(timezone.utc).isoformat(),
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

        print(f"\n{'='*60}")
        print("[FLYWHEEL] Iteration complete!")
        print(f"  Seeds:      {seed_range}")
        print(f"  Checkpoint: {new_checkpoint}")
        print(f"  Next:       {state['current']['seed_range']}")
        print(f"{'='*60}\n")

        return 0

    except Exception as e:
        # Update state with failure
        state["status"] = "failed"
        state["last_error"] = str(e)
        state["next_action"] = f"Fix error and set status to 'ready': {e}"
        save_state(state)

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

    args = parser.parse_args()

    if args.command:
        return args.func(args)
    else:
        return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
