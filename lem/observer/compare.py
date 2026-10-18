# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""Pair two nested-corpus observers on fresh and reused held-out whole deals."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from .data import QUERIES, fingerprint, partition, prepare
from .model import Config, Observer


def cohorts(reference, candidate):
    """Require exact nested recordings and deterministic whole-deal partitions."""
    maps = []
    for records in (reference, candidate):
        mapping = {}
        for row in records:
            key = tuple(row["hands"])
            if key in mapping:
                raise ValueError("duplicate physical deal")
            if row["split"] != partition(row["hands"]):
                raise ValueError("split differs from the whole-deal partition")
            mapping[key] = row
        maps.append(mapping)
    old, new = maps
    if any(new.get(key) != row for key, row in old.items()):
        raise ValueError("candidate must retain every original recording and split exactly")
    result = {
        "fresh": [r for r in candidate if r["split"] == "test" and tuple(r["hands"]) not in old],
        "reused": [r for r in reference if r["split"] == "test"],
    }
    if any(not rows for rows in result.values()):
        raise ValueError("need both fresh and reused test cohorts")
    training = {key for mapping in maps for key, row in mapping.items() if row["split"] == "train"}
    if any(tuple(r["hands"]) in training for rows in result.values() for r in rows):
        raise ValueError("a test deal was used for training")
    return result


def interval(totals, counts, draws):
    """Resample the deal totals and their denominators together, never rows."""
    denominators = counts[draws].sum(-1)
    if counts.sum() == 0 or np.any(denominators == 0):
        raise ValueError("empty decision group in bootstrap sample")
    samples = totals[draws].sum(-1) / denominators
    return {"estimate": float(totals.sum() / counts.sum()),
            "ci95": np.quantile(samples, [.025, .975]).tolist()}


@torch.inference_mode()
def values(checkpoint, data, device):
    model = Observer(Config(**checkpoint["config"])).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    probabilities = []
    for start in range(0, len(data["x"]), 128):
        x = torch.as_tensor(data["x"][start:start + 128], device=device)
        probabilities.append(model(x)[:, QUERIES].softmax(-1).cpu().numpy())
    p = np.concatenate(probabilities)
    actual = np.take_along_axis(p, data["targets"][..., None], -1)[..., 0]
    return {"nll": -np.log(actual.clip(1e-30)),
            "top1": (p.argmax(-1) == data["targets"]).astype(float),
            "known_impossible_mass": (p * ~data["possible"]).sum(-1)}


def compare(reference, candidate, destination, *, step=20000, device="mps", draws=2000):
    torch.set_num_threads(2)
    roots = {"reference": Path(reference), "candidate": Path(candidate)}
    records = {k: json.loads((p / "deals.json").read_text()) for k, p in roots.items()}
    tests = cohorts(records["reference"], records["candidate"])
    initial = {k: torch.load(p / "initial.pt", map_location="cpu", weights_only=True)
               for k, p in roots.items()}
    a, b = initial["reference"], initial["candidate"]
    if (a["config"] != b["config"] or a["model"].keys() != b["model"].keys()
            or any(not torch.equal(a["model"][k], b["model"][k]) for k in a["model"])):
        raise ValueError("initial parameter tensors differ")
    saved, provenance = {"initial": a}, {}
    for name, root in roots.items():
        for kind, filename in (("budget", f"step-{step:05d}.pt"), ("best", "best.pt")):
            path = root / filename
            c = torch.load(path, map_location="cpu", weights_only=True)
            if c["data_sha256"] != fingerprint(records[name]) or c["config"] != a["config"]:
                raise ValueError("checkpoint does not match its recorded corpus or architecture")
            if kind == "budget" and c["step"] != step:
                raise ValueError("checkpoint has a different update budget")
            key = f"{name}_{kind}"
            saved[key] = c
            provenance[key] = {"path": str(path), "step": c["step"],
                               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    controls = ("seed", "batch", "lr", "device")
    control = saved["reference_budget"]
    if any(c["tokenizer"] != control["tokenizer"] or any(
            c["arguments"][k] != control["arguments"][k] for k in controls)
           for name, c in saved.items() if name != "initial"):
        raise ValueError("matched-update comparisons require identical training controls and tokenizer")
    result = {"method": "paired whole-deal percentile bootstrap; 95%; seed 42",
              "draws": draws, "matched_step": step, "initial_tensors_identical": True,
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "training_controls": {k: control["arguments"][k] for k in controls},
              "checkpoints": provenance, "cohorts": {}}
    for cohort, rows in tests.items():
        d, n = prepare(rows), len(rows)
        predictions = {k: values(c, d, device) for k, c in saved.items()}
        support = d["possible"].sum(-1)
        predictions["visible_uniform"] = {"nll": np.log(support), "top1": 1 / support,
                                            "known_impossible_mass": np.zeros_like(support, dtype=float)}
        groups = {"all": np.ones_like(d["own"]), "own": d["own"], "others": ~d["own"],
                  "discretionary": d["legal"].sum(-1) > 1}
        sample = np.random.default_rng(42).integers(0, n, (draws, n))
        output = {"deals": n, "record_sha256": fingerprint(rows), "groups": {}}
        for group, mask in groups.items():
            counts = mask.reshape(n, -1).sum(-1)
            totals = {name: {metric: (value * mask).reshape(n, -1).sum(-1)
                             for metric, value in measures.items()}
                      for name, measures in predictions.items()}
            stats = {name: {m: interval(v, counts, sample) for m, v in measures.items()}
                     for name, measures in totals.items()}
            for kind in ("budget", "best"):
                stats[f"candidate_minus_reference_{kind}"] = {
                    m: interval(totals[f"candidate_{kind}"][m] - totals[f"reference_{kind}"][m],
                                counts, sample) for m in ("nll", "top1", "known_impossible_mass")}
            output["groups"][group] = stats
        result["cohorts"][cohort] = output
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(destination), "fresh_deals": len(tests["fresh"]),
                      "fresh_matched_update_difference": result["cohorts"]["fresh"]["groups"]["all"]
                      ["candidate_minus_reference_budget"]}))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference")
    p.add_argument("candidate")
    p.add_argument("--out", required=True)
    p.add_argument("--step", type=int, default=20000)
    p.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    args = p.parse_args()
    compare(args.reference, args.candidate, args.out, step=args.step, device=args.device)
