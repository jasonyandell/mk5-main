# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""Describe test uncertainty by resampling whole deals, keeping views together.

This is conditional on one trained model and this pour, not training-seed or
new-population uncertainty. No model or checkpoint selection happens here.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import QUERIES, prepare
from .model import Config, Observer


@torch.inference_mode()
def summarize(run, draws=2000):
    run = Path(run)
    torch.set_num_threads(2)
    records = [r for r in json.loads((run / "deals.json").read_text()) if r["split"] == "test"]
    d, n = prepare(records), len(records)
    predictions = {}
    for name in ("initial", "best"):
        saved = torch.load(run / f"{name}.pt", map_location="cpu", weights_only=True)
        model = Observer(Config(**saved["config"])).eval()
        model.load_state_dict(saved["model"])
        predictions[name] = np.concatenate([model(torch.as_tensor(d["x"][i:i+128]))[:, QUERIES]
                                            .softmax(-1).numpy() for i in range(0, len(d["x"]), 128)])
    p, p0 = predictions["best"], predictions["initial"]
    target = d["targets"][..., None]
    loss = -np.log(np.take_along_axis(p, target, -1)[..., 0].clip(1e-30))
    loss0 = -np.log(np.take_along_axis(p0, target, -1)[..., 0].clip(1e-30))
    hit, hit0 = p.argmax(-1) == d["targets"], p0.argmax(-1) == d["targets"]
    values = {"top1": hit, "initial_top1": hit0, "nll": loss, "initial_nll": loss0,
              "known_impossible_mass": (p * ~d["possible"]).sum(-1),
              "visible_uniform_nll": np.log(d["possible"].sum(-1)),
              "visible_uniform_expected_top1": 1 / d["possible"].sum(-1),
              "nll_minus_initial": loss-loss0,
              "nll_minus_visible_uniform": loss-np.log(d["possible"].sum(-1))}
    resamples = np.random.default_rng(42).integers(0, n, (draws, n))
    groups = {"all": np.ones_like(d["own"]), "own": d["own"], "others": ~d["own"],
              "discretionary": d["legal"].sum(-1) > 1}
    result = {"method": "paired whole-deal percentile bootstrap; 95%; seed 42",
              "deals": n, "draws": draws, "groups": {}}
    for group, mask in groups.items():
        counts = mask.reshape(n, -1).sum(-1)
        denominator = counts[resamples].sum(-1)
        result["groups"][group] = {}
        for name, value in values.items():
            totals = (value * mask).reshape(n, -1).sum(-1)
            samples = totals[resamples].sum(-1) / denominator
            result["groups"][group][name] = {"estimate": float(totals.sum()/counts.sum()),
                                            "ci95": np.quantile(samples, [.025, .975]).tolist()}
    (run / "intervals.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run")
    args = p.parse_args()
    print(json.dumps(summarize(args.run)))
