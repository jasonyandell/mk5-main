#!/usr/bin/env python3
"""Fleet launcher for Zeb training runs.

Usage:
    python -m forge.zeb.vast.fleet list              # show all models
    python -m forge.zeb.vast.fleet bootstrap large-belief  # create HF checkpoint
    python -m forge.zeb.vast.fleet learner large-belief    # start learner
    python -m forge.zeb.vast.fleet workers large-belief 4  # start 4 vast workers
"""
import argparse
import json
import os
import subprocess
import sys

import torch

# ── Model registry ──────────────────────────────────────────────────────
# Each entry: base weights on HF, config overrides, learner/worker params
MODELS = {
    "large": {
        "base_weights": "large",        # HF weights_name to fork from
        "config_overrides": {},
        "workers": 8,
        "lr": 1e-4,
        "batch_size": 64,
        "replay_buffer_size": 1_000_000,
        "steps_per_cycle": 1000,
        "push_every": 10,
        "eval_every": 10,
    },
    "large-belief": {
        "base_weights": "large",
        "config_overrides": {"belief_head": True},
        "workers": 4,
        "lr": 1e-4,
        "batch_size": 64,
        "replay_buffer_size": 500_000,
        "steps_per_cycle": 1000,
        "push_every": 10,
        "eval_every": 10,
    },
    "medium-belief": {
        "base_weights": None,           # default model.pt
        "config_overrides": {"belief_head": True},
        "workers": 4,
        "lr": 1e-4,
        "batch_size": 64,
        "replay_buffer_size": 200_000,
        "steps_per_cycle": 100,
    },
}

REPO_ID = "jasonyandell/zeb-42"
EXAMPLES_REPO_ID = "jasonyandell/zeb-42-examples"


def cmd_list():
    print(f"{'Name':<20} {'Base':<15} {'Overrides':<30} {'Workers'}")
    print("─" * 75)
    for name, cfg in MODELS.items():
        base = cfg["base_weights"] or "model"
        overrides = json.dumps(cfg["config_overrides"]) if cfg["config_overrides"] else "—"
        print(f"{name:<20} {base:<15} {overrides:<30} {cfg['workers']}")


def cmd_bootstrap(name: str):
    from forge.zeb.model import ZebModel
    from forge.zeb.hf import pull_weights, init_repo, push_weights

    cfg = MODELS[name]
    base = cfg["base_weights"]
    hf_weights = f"{base}.pt" if base else "model.pt"

    print(f"Pulling base weights: {hf_weights}")
    state_dict, model_config = pull_weights(REPO_ID, device="cpu", weights_name=hf_weights)
    print(f"  Base config: {model_config}")

    # Apply overrides
    new_config = dict(model_config, **cfg["config_overrides"])
    print(f"  New config:  {new_config}")

    # Build model, load with strict=False for new heads
    model = ZebModel(**new_config)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  New params (random init): {missing}")

    total = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total:,}")

    # Push to HF under new name
    weights_name = f"{name}.pt"
    init_repo(REPO_ID, new_config, weights_name=weights_name)
    push_weights(model, REPO_ID, step=0, total_games=0, weights_name=weights_name)
    print(f"  Pushed to HF: {weights_name}")

    # Save local bootstrap checkpoint
    ckpt_path = f"forge/zeb/{name}-bootstrap.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": new_config,
        "epoch": 0,
        "total_games": 0,
    }, ckpt_path)
    print(f"  Local checkpoint: {ckpt_path}")
    print("Done.")


def cmd_learner(name: str):
    cfg = MODELS[name]
    ckpt = f"forge/zeb/{name}-bootstrap.pt"
    if not os.path.exists(ckpt):
        print(f"No bootstrap checkpoint at {ckpt} — run: fleet.py bootstrap {name}")
        sys.exit(1)

    cmd = [
        sys.executable, "-u", "-m", "forge.zeb.learner.run",
        "--repo-id", REPO_ID,
        "--examples-repo-id", EXAMPLES_REPO_ID,
        "--checkpoint", ckpt,
        "--weights-name", name,
        "--lr", str(cfg["lr"]),
        "--batch-size", str(cfg["batch_size"]),
        "--replay-buffer-size", str(cfg["replay_buffer_size"]),
        "--training-steps-per-cycle", str(cfg["steps_per_cycle"]),
        "--push-every", str(cfg.get("push_every", 10)),
        "--eval-every", str(cfg.get("eval_every", 10)),
        "--eval-games", "2000",
        "--keep-example-files", "100",
        "--min-buffer", "250000",
        "--max-example-age", "600",
        "--wandb", "--run-name", name,
    ]
    print(f"Starting learner: {name}")
    print(f"  {' '.join(cmd)}")
    os.execvp(cmd[0], cmd)


def cmd_workers(name: str, n: int):
    env = os.environ.copy()
    env["ZEB_WEIGHTS_NAME"] = name
    env["ZEB_FLEET"] = f"zeb-{name}"
    script = os.path.join(os.path.dirname(__file__), "vast_monitor.sh")
    cmd = [script, str(n)]
    print(f"Starting {n} workers for {name}")
    os.execvp(cmd[0], cmd)


def main():
    parser = argparse.ArgumentParser(description="Zeb fleet manager")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("list", help="List all models")

    p = sub.add_parser("bootstrap", help="Create HF checkpoint from base model")
    p.add_argument("name", choices=list(MODELS.keys()))

    p = sub.add_parser("learner", help="Start learner for a model")
    p.add_argument("name", choices=list(MODELS.keys()))

    p = sub.add_parser("workers", help="Start Vast.ai workers")
    p.add_argument("name", choices=list(MODELS.keys()))
    p.add_argument("n", type=int, nargs="?", default=None, help="Number of workers")

    args = parser.parse_args()
    if args.cmd == "list":
        cmd_list()
    elif args.cmd == "bootstrap":
        cmd_bootstrap(args.name)
    elif args.cmd == "learner":
        cmd_learner(args.name)
    elif args.cmd == "workers":
        n = args.n or MODELS[args.name]["workers"]
        cmd_workers(args.name, n)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
