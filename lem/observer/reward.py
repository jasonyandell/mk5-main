# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""Resumable, bounded population work units; invoke each through cpu-slot."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import time

import numpy as np
import torch

from .data import VERSION
from .play import ObserverPlay
from .population import (atomic_json, atomic_save, collect, digest, load_model,
                         seed_for, update)


def policies(config, device):
    from arena.walt_play import parse_walt_spec
    if digest(config["zeb_field"]) != config["zeb_field_sha256"]:
        raise ValueError("Zeb field pin changed")
    result = {"zeb": parse_walt_spec(config["zeb_spec"])}
    # The declared cpu-slot threads also bound the Rust planner explicitly.
    result["zeb"].walt.threads = 2
    for name, identity in config["pool"].items():
        if digest(identity["path"]) != identity["sha256"]:
            raise ValueError("frozen population checkpoint changed")
        result[name] = ObserverPlay(identity["path"], device)
    return result


def save_episodes(path, episodes):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as stream:
            for episode in episodes:
                stream.write((json.dumps(episode, sort_keys=True) + "\n").encode())
    temporary.replace(path)


def outcome_summary(episodes):
    wins = np.array([(e["reward"]+1)/2 for e in episodes])
    result = {"hands": len(episodes), "team_win_rate": float(wins.mean()),
              "mean_reward": float(2*wins.mean()-1), "replays_verified": len(episodes)}
    groups = {}
    for i, e in enumerate(episodes):
        seat = e["learner"]
        labels = [f"bidder_role_{(seat-e['bidder'])%4}",
                  "partner_zeb" if e["seats"][(seat+2)%4] == "zeb" else "partner_gus",
                  f"zeb_opponents_{sum(e['seats'][(seat+d)%4]=='zeb' for d in (1,3))}"]
        for label in labels:
            groups.setdefault(label, []).append(wins[i])
    result["groups"] = {k: {"hands": len(v), "win_rate": float(np.mean(v))} for k, v in groups.items()}
    return result


def train_block(run, index, device):
    run = Path(run)
    config = json.loads((run / "config.json").read_text())
    out = run / f"block-{index:02d}"
    out.mkdir(exist_ok=True)
    checkpoint = out / "checkpoint.pt"
    receipt = out / "complete.json"
    if receipt.exists():
        stored = json.loads(receipt.read_text())
        if digest(checkpoint) != stored["checkpoint_sha256"]:
            raise ValueError("completed block checkpoint changed")
        return stored
    if not 0 <= index < config["blocks"]:
        raise ValueError("block outside registered budget")
    previous = run / "initial.pt" if index == 0 else run / f"block-{index-1:02d}/checkpoint.pt"
    if index > 0 and not (previous.parent / "complete.json").exists():
        raise ValueError("previous block incomplete")
    started = time.monotonic()
    torch.set_num_threads(2)
    model, saved = load_model(previous, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], betas=(.9, .95), weight_decay=0.)
    if index:
        optimizer.load_state_dict(saved["optimizer"])
    all_specs = json.loads((run / "specs.json").read_text())
    start, end = index * config["hands_per_block"], (index+1)*config["hands_per_block"]
    specs = all_specs[start:end]
    if len(specs) != config["hands_per_block"]:
        raise ValueError("missing predeclared hands")
    population = policies(config, device)
    before = time.monotonic()
    data, episodes = collect(model, population, specs, device)
    collection_seconds = time.monotonic() - before
    # Keep every trajectory before optimizing. A failure does not erase it.
    save_episodes(out / "hands.jsonl.gz", episodes)
    np.savez_compressed(out / "decisions.npz", **data)
    prior_hands = saved.get("prior_hands", 0) if index else 0
    prior_return = saved.get("prior_return", 0.) if index else 0.
    baseline = prior_return / prior_hands if prior_hands else 0.
    before = time.monotonic()
    metrics = update(model, optimizer, data, baseline,
                     seed_for(config["generation"], index, "update"), device)
    optimizer_seconds = time.monotonic() - before
    outcome = outcome_summary(episodes)
    total_updates = saved.get("rl_updates", 0) + metrics["passes_completed"] * ((len(data["action"])+255)//256)
    payload = {"config": saved["config"], "model": {k:v.detach().cpu() for k,v in model.state_dict().items()},
               "optimizer": optimizer.state_dict(), "tokenizer": VERSION, "step": total_updates,
               "rl_updates": total_updates, "generation": config["generation"], "block": index,
               "prior_hands": prior_hands + len(episodes),
               "prior_return": prior_return + sum(e["reward"] for e in episodes),
               "parent_checkpoint_sha256": digest(previous),
               "data_sha256": digest(out / "hands.jsonl.gz"),
               "pretraining_data_sha256": config["pretraining_data_sha256"],
               "rng_protocol": "per-hand action seed and per-block optimizer seed; no dropout"}
    atomic_save(checkpoint, payload)
    report = {"generation": config["generation"], "block": index,
              "checkpoint_sha256": digest(checkpoint), "decisions_sha256": digest(out / "decisions.npz"),
              "hands_sha256": digest(out / "hands.jsonl.gz"), "outcome": outcome,
              "optimizer": metrics, "collection_seconds": collection_seconds,
              "optimizer_seconds": optimizer_seconds, "wall_seconds": time.monotonic()-started,
              "rl_updates": total_updates}
    atomic_json(receipt, report)
    print(json.dumps(report), flush=True)
    return report


def evaluate_panel(run, checkpoint, spec_path, out, device, reference=None, prediction=None, purpose="development"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((Path(run) / "config.json").read_text())
    started = time.monotonic()
    torch.set_num_threads(2)
    model, _ = load_model(checkpoint, device)
    specs = json.loads(Path(spec_path).read_text())
    _, episodes = collect(model, policies(config, device), specs, device, sample=False)
    save_episodes(out / "hands.jsonl.gz", episodes)
    wins = np.array([(e["reward"]+1)/2 for e in episodes])
    result = {**outcome_summary(episodes), "wins": wins.tolist(),
              "checkpoint_sha256": digest(checkpoint), "specs_sha256": digest(spec_path),
              "scope": "single learner seat, fixed G0/Zeb table; legal argmax; bid30 pip",
              "selection": ("fresh confirmation panel, after all model selection" if purpose == "confirmation"
                            else "development panel; repeated adaptive use is exploratory")}
    if reference:
        ref = json.loads(Path(reference).read_text())
        if ref["specs_sha256"] != result["specs_sha256"]:
            raise ValueError("reference uses different deals or table assignments")
        delta = wins-np.asarray(ref["wins"])
        idx = np.random.default_rng(42).integers(len(delta), size=(2000, len(delta)))
        result["vs_reference"] = {"win_rate_difference": float(delta.mean()),
                                  "ci95": np.quantile(delta[idx].mean(1), [.025,.975]).tolist(),
                                  "method": "whole-hand paired bootstrap; 2000 draws; seed42"}
    if prediction:
        from .train import evaluate
        with np.load(prediction, allow_pickle=False) as source:
            data = {k:source[k] for k in source.files}
        result["prediction"] = evaluate(model, data, np.arange(len(data["x"])), device)
    result["wall_seconds"] = time.monotonic()-started
    atomic_json(out / "report.json", result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("wins","prediction")}), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("block", "panel"))
    p.add_argument("--run", required=True)
    p.add_argument("--device", default="mps", choices=("cpu", "mps", "cuda"))
    p.add_argument("--index", type=int)
    p.add_argument("--checkpoint")
    p.add_argument("--specs")
    p.add_argument("--out")
    p.add_argument("--reference")
    p.add_argument("--prediction")
    p.add_argument("--purpose", choices=("development", "confirmation"), default="development")
    a = p.parse_args()
    if a.command == "block":
        train_block(a.run, a.index, a.device)
    else:
        evaluate_panel(a.run, a.checkpoint, a.specs, a.out, a.device, a.reference, a.prediction, a.purpose)


if __name__ == "__main__":
    main()
