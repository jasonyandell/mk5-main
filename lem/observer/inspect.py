# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""Build a standalone worked-hand lesson from saved training artifacts."""
import argparse
import json
from pathlib import Path

import torch

from .data import QUERIES, label, prepare
from .model import Config, Observer


@torch.inference_mode()
def build(run, destination):
    run, destination = Path(run), Path(destination)
    records = json.loads((run / "deals.json").read_text())
    record = next(r for r in records if r["split"] == "test")
    data = prepare([record])
    x = torch.as_tensor(data["x"])
    payload = {"report": json.loads((run / "report.json").read_text()),
               "seed": record["seed"], "decl": record["decl"],
               "tile_names": [label(t) for t in range(28)], "queries": QUERIES.tolist(),
               "views": []}
    intervals = run / "intervals.json"
    if intervals.exists():
        payload["intervals"] = json.loads(intervals.read_text())
    predictions, traces = {}, None
    for name in ("initial", "best"):
        saved = torch.load(run / f"{name}.pt", map_location="cpu", weights_only=True)
        model = Observer(Config(**saved["config"])).eval()
        model.load_state_dict(saved["model"])
        logits, trace = model(x, inspect=True)
        predictions[name] = logits[:, QUERIES].softmax(-1).numpy().round(7).tolist()
        if name == "best":
            traces = trace
    for viewer in range(4):
        payload["views"].append({
            "tokens": [label(int(t)) for t in data["x"][viewer]],
            "targets": data["targets"][viewer].tolist(),
            "possible": data["possible"][viewer].tolist(),
            "own": data["own"][viewer].tolist(),
            "initial": predictions["initial"][viewer], "trained": predictions["best"][viewer],
            "attention": [t["attention"][viewer, :, QUERIES].numpy().round(5).tolist() for t in traces],
            "units": [t["pre"][viewer, QUERIES].numpy().round(5).tolist() for t in traces],
        })
    template = Path(__file__).with_name("viewer.html").read_text()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(template.replace("__OBSERVER_DATA__", json.dumps(payload).replace("<", "\\u003c")))
    print(destination)
    return payload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    build(args.run, args.out)


if __name__ == "__main__":
    main()
