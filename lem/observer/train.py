# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""Train on observed next tiles. Run with scripts/cpu-slot; see README.md."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import hashlib
from pathlib import Path
import platform
import subprocess
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F

from .data import QUERIES, VERSION, fingerprint, load_deals, prepare
from .model import Config, Observer
from .tracking import Tracking


@torch.inference_mode()
def evaluate(model, data, indices, device, batch_size=128):
    model.eval()
    probabilities = []
    for start in range(0, len(indices), batch_size):
        x = torch.as_tensor(data["x"][indices[start:start + batch_size]], device=device)
        probabilities.append(model(x)[:, QUERIES].softmax(-1).cpu().numpy())
    p = np.concatenate(probabilities)
    target = data["targets"][indices]
    nll = -np.log(np.take_along_axis(p, target[..., None], -1)[..., 0].clip(1e-30))
    hit = p.argmax(-1) == target
    possible, legal = data["possible"][indices], data["legal"][indices]
    own = data["own"][indices]
    groups = {"all": np.ones_like(own), "own": own, "others": ~own,
              "discretionary": legal.sum(-1) > 1, "forced_move": legal.sum(-1) == 1,
              "coverage_forced": data["coverage_forced"][indices],
              "book_declared": ~data["coverage_forced"][indices]}
    result = {}
    for name, mask in groups.items():
        if not mask.any():
            continue
        result[name] = {
            "decisions": int(mask.sum()), "nll": float(nll[mask].mean()),
            "top1": float(hit[mask].mean()),
            "known_impossible_mass": float((p * ~possible).sum(-1)[mask].mean()),
            "visible_uniform_nll": float(np.log(possible.sum(-1))[mask].mean()),
            "visible_uniform_expected_top1": float((1 / possible.sum(-1))[mask].mean()),
        }
    return result


def save_checkpoint(path, model, optimizer, sampler, step, best_loss, history, data_hash, arguments):
    payload = {"config": asdict(model.config), "model": model.state_dict(),
               "optimizer": optimizer.state_dict(), "sampler_state": sampler.get_state(),
               "torch_rng_state": torch.get_rng_state(), "step": step, "best_loss": best_loss,
               "history": history, "data_sha256": data_hash, "tokenizer": VERSION, "arguments": arguments}
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def run(args):
    out = Path(args.out)
    if (out / "latest.pt").exists() and not args.resume:
        raise ValueError("run already exists; choose another directory or --resume")
    out.mkdir(parents=True, exist_ok=True)
    if args.resume and args.init_from:
        raise ValueError("choose resume or init-from, not both")
    if args.init_from and (out / "latest.pt").exists():
        raise ValueError("a continuation needs a new output directory")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS unavailable; choose a device explicitly")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    records, sources = load_deals(args.source, args.deals)
    data, data_hash = prepare(records), fingerprint(records)
    splits = {s: np.flatnonzero(data["split"] == s) for s in ("train", "validation", "test")}
    if any(len(v) == 0 for v in splits.values()):
        raise ValueError("need at least one deal in each split")
    about = {"data_sha256": data_hash, "tokenizer": VERSION, "sources": sources,
             "split_deals": {k: len(v) // 4 for k, v in splits.items()},
             "declarations": {str(d): sum(r["decl"] == d for r in records) for d in (*range(8), 9)},
             "seed": args.seed, "torch": str(torch.__version__), "platform": platform.platform(),
             "device": args.device, "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    config = Config(width=args.width, layers=args.layers, heads=args.heads)
    model = Observer(config).to(args.device)
    # As in nanoGPT: matrices decay; normalization scales and biases do not.
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if p.ndim >= 2], "weight_decay": 0.1},
        {"params": [p for p in model.parameters() if p.ndim < 2], "weight_decay": 0.0},
    ], lr=args.lr, betas=(0.9, 0.95))
    sampler = torch.Generator().manual_seed(args.seed + 1)
    history, start, best_loss = [], 0, float("inf")
    source = out if args.resume else Path(args.init_from) if args.init_from else None
    if source is not None:
        saved = torch.load(source / "latest.pt", map_location="cpu", weights_only=True)
        if saved["data_sha256"] != data_hash or saved["config"] != asdict(config) or saved["tokenizer"] != VERSION:
            raise ValueError("resume requires identical data, tokenizer and model")
        if any(saved["arguments"][k] != vars(args)[k] for k in ("seed", "lr", "batch", "eval_every", "device")):
            raise ValueError("resume requires identical training settings")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        sampler.set_state(saved["sampler_state"])
        torch.set_rng_state(saved["torch_rng_state"])
        start, best_loss, history = saved["step"], saved["best_loss"], saved["history"]
        initial = json.loads((source / "initial.json").read_text())
        if args.init_from:
            about["parent_checkpoint"] = str(source / "latest.pt")
            about["parent_sha256"] = hashlib.sha256((source / "latest.pt").read_bytes()).hexdigest()
            about["parent_step"] = start
            for name in ("initial.pt", "initial.json", "best.pt", "latest.pt"):
                shutil.copy2(source / name, out / name)
        elif "parent_step" in json.loads((out / "data.json").read_text()):
            previous = json.loads((out / "data.json").read_text())
            about.update({k: previous[k] for k in ("parent_step", "parent_checkpoint", "parent_sha256")})
    else:
        initial = evaluate(model, data, splits["test"], args.device)
        (out / "initial.json").write_text(json.dumps(initial, indent=2) + "\n")
        save_checkpoint(out / "initial.pt", model, optimizer, sampler, 0, best_loss, history, data_hash, vars(args))
    (out / "deals.json").write_text(json.dumps(records))
    (out / "data.json").write_text(json.dumps(about, indent=2) + "\n")
    print(json.dumps({**about, "parameters": sum(p.numel() for p in model.parameters()), "sources": len(sources)}), flush=True)
    train_x = torch.as_tensor(data["x"][splits["train"]], device=args.device)
    train_y = torch.as_tensor(data["targets"][splits["train"]], device=args.device)
    # A fixed probe of training deals makes overfitting visible without changing the sampler.
    probe = splits["train"][:min(1024, len(splits["train"]))]
    if args.init_from:
        history[-1] = {**history[-1], "train_probe": evaluate(model, data, probe, args.device),
                       "continuation_boundary": True}
    with Tracking(args, asdict(config), about, history, len(train_x)) as tracker:
        started = time.monotonic()
        for step in range(start + 1, args.steps + 1):
            model.train()
            indices = torch.randint(len(train_x), (args.batch,), generator=sampler).to(args.device)
            logits = model(train_x[indices])[:, QUERIES]
            loss = F.cross_entropy(logits.reshape(-1, 28), train_y[indices].reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step % args.eval_every == 0 or step == args.steps:
                validation = evaluate(model, data, splits["validation"], args.device)
                train_probe = evaluate(model, data, probe, args.device)
                row = {"step": step, "train_batch_nll": float(loss.detach().cpu()), "train_probe": train_probe,
                       "validation": validation, "segment_seconds": round(time.monotonic() - started, 2)}
                history.append(row)
                improved = validation["all"]["nll"] < best_loss
                best_loss = min(best_loss, validation["all"]["nll"])
                save_checkpoint(out / "latest.pt", model, optimizer, sampler, step, best_loss, history, data_hash, vars(args))
                if step % args.checkpoint_every == 0 or step == args.steps:
                    save_checkpoint(out / f"step-{step:05d}.pt", model, optimizer, sampler, step, best_loss, history, data_hash, vars(args))
                    tracker.checkpoint(step)
                if improved:
                    save_checkpoint(out / "best.pt", model, optimizer, sampler, step, best_loss, history, data_hash, vars(args))
                print(json.dumps({"step": step, "train_nll": row["train_batch_nll"],
                                  "validation": validation["all"], "seconds": row["segment_seconds"]}), flush=True)
                tracker.log(row)
                (out / "progress.json").write_text(json.dumps({"step": step,
                    "epoch_equivalent": step * args.batch / len(train_x), "validation": validation["all"],
                    "train_probe": train_probe["all"]}, indent=2) + "\n")
        final_checkpoint_test = evaluate(model, data, splits["test"], args.device)
        saved = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model"])
        final = evaluate(model, data, splits["test"], args.device)
        report = {**about, "config": asdict(config), "parameters": sum(p.numel() for p in model.parameters()),
                  "best_step": saved["step"], "steps": args.steps, "initial_test": initial,
                  "test": final, "final_test": final_checkpoint_test, "history": history, "uniform_28_nll": float(np.log(28)),
                  "uniform_28_top1": 1 / 28}
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"best_step": saved["step"], "test": final}), flush=True)
        tracker.finish(report)
        return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", action="append", required=True, help="raw pour directory; repeat for pip and dtnt")
    p.add_argument("--out", required=True)
    p.add_argument("--deals", type=int, default=5000)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--init-from", help="branch from an existing run, including optimizer and sampler")
    p.add_argument("--checkpoint-every", type=int, default=250)
    p.add_argument("--wandb-project", help="enable online tracking in this project")
    p.add_argument("--wandb-entity")
    p.add_argument("--wandb-name")
    args = p.parse_args()
    if min(args.deals, args.steps, args.eval_every, args.checkpoint_every, args.batch) < 1 or args.lr <= 0:
        p.error("positive training sizes and learning rate required")
    if args.checkpoint_every % args.eval_every:
        p.error("checkpoint-every must be a multiple of eval-every")
    run(args)


if __name__ == "__main__":
    main()
