# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""The probe kit's commands; see the package docstring."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

NET = "zeb/net/margin_d250k_h128x64.fnet"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m lem.probe")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample"); s.add_argument("pour"); s.add_argument("out")
    s.add_argument("--every", type=int, default=10); s.add_argument("--layout", default="field-v2")
    s.add_argument("--offset", type=int, default=0, help="the trainer's deal offset for this pour")
    s = sub.add_parser("merge"); s.add_argument("out"); s.add_argument("parts", nargs="+")
    s = sub.add_parser("concepts"); s.add_argument("sample"); s.add_argument("out")
    s.add_argument("--start", type=int, default=0); s.add_argument("--stop", type=int, default=None)
    s.add_argument("--list", choices=("v1", "v2"), default="v1")
    s = sub.add_parser("concepts-merge"); s.add_argument("out"); s.add_argument("parts", nargs="+")
    s.add_argument("--as-list", choices=("v1", "v2"), default=None, help="v2 labels saved as v1 (its first columns)")
    s = sub.add_parser("audit"); s.add_argument("sample"); s.add_argument("labels"); s.add_argument("out_dir")
    s.add_argument("--net", default=NET)
    s.add_argument("--shuffles", type=int, default=20, help="0 skips the chance baseline")
    s = sub.add_parser("cards"); s.add_argument("sample"); s.add_argument("audit_dir"); s.add_argument("out")
    s.add_argument("--net", default=NET)
    s = sub.add_parser("readers"); s.add_argument("cards"); s.add_argument("out_dir")
    s.add_argument("--workers", type=int, default=6); s.add_argument("--model", default="haiku")
    s = sub.add_parser("ablate"); s.add_argument("sample"); s.add_argument("out"); s.add_argument("--net", default=NET)
    s = sub.add_parser("aa"); s.add_argument("sample"); s.add_argument("labels"); s.add_argument("audit_dir"); s.add_argument("belief_parts"); s.add_argument("out_dir")
    s.add_argument("--net", default=NET)
    s = sub.add_parser("squeeze-train"); s.add_argument("pours", nargs="+"); s.add_argument("--out", required=True)
    s.add_argument("--widths", required=True, help="128x64,32x16,..."); s.add_argument("--seeds", default="0")
    s.add_argument("--epochs", type=int, default=60); s.add_argument("--device", default="mlx")
    s = sub.add_parser("formation-train"); s.add_argument("pours", nargs="+"); s.add_argument("--out", required=True)
    s.add_argument("--widths", required=True); s.add_argument("--seeds", default="0,1,2"); s.add_argument("--epochs", type=int, default=60)
    s.add_argument("--device", default="mlx")
    s = sub.add_parser("formation-report"); s.add_argument("audit_root"); s.add_argument("runs_root"); s.add_argument("sample")
    s.add_argument("labels"); s.add_argument("q_dir"); s.add_argument("out")
    s = sub.add_parser("squeeze-families"); s.add_argument("sample"); s.add_argument("labels"); s.add_argument("audit_dir"); s.add_argument("out")
    s.add_argument("--net", default=NET)
    s = sub.add_parser("list"); s.add_argument("out")
    s = sub.add_parser("fidelity-collect"); s.add_argument("sample"); s.add_argument("pour"); s.add_argument("k", type=int); s.add_argument("out")
    s = sub.add_parser("fidelity"); s.add_argument("sample"); s.add_argument("labels"); s.add_argument("q_dir"); s.add_argument("out_dir")
    s.add_argument("--net", default=NET)
    s = sub.add_parser("belief"); s.add_argument("sample"); s.add_argument("out_dir"); s.add_argument("--net", default=NET)
    s.add_argument("--fit", type=int, default=20000); s.add_argument("--eval", type=int, default=20000)
    s.add_argument("--parts-dir", default=None, help="assemble the targets from belief-targets parts instead of computing them")
    s = sub.add_parser("belief-targets"); s.add_argument("sample"); s.add_argument("out_dir")
    s.add_argument("--part", type=int, required=True); s.add_argument("--parts", type=int, required=True)
    s.add_argument("--fit", type=int, default=20000); s.add_argument("--eval", type=int, default=20000)
    a = ap.parse_args(argv)

    if a.cmd == "sample":
        from lem.probe.sample import build
        build(a.pour, a.out, a.every, log, a.layout, a.offset)
    elif a.cmd == "merge":
        from lem.probe.sample import merge
        merge(a.out, a.parts, log)
    elif a.cmd == "concepts":
        from lem.probe import concepts
        from lem.probe.sample import Sample
        sample = Sample(a.sample)
        cl = concepts.LISTS[a.list]
        labels = concepts.evaluate(sample, a.start, a.stop, log=log, concepts=cl)
        concepts.save(a.out, labels, cl)
        log(f"{a.out}: {labels.shape} ({concepts.identity(cl)[:12]}, list {a.list})")
    elif a.cmd == "concepts-merge":
        from lem.probe import concepts
        parts = [np.load(p) for p in a.parts]
        ids = {str(p["identity"]) for p in parts}
        assert len(ids) == 1, "every part was labelled with one concept list"
        cl, labels = concepts.list_of(ids.pop()), np.concatenate([p["labels"] for p in parts])
        if a.as_list:
            want = concepts.LISTS[a.as_list]
            assert cl[:len(want)] == want, "the target list is a prefix of the labelled one"
            cl, labels = want, labels[:, :len(want)]
        concepts.save(a.out, labels, cl)
    elif a.cmd == "audit":
        from lem.probe import audit, concepts
        from lem.probe.net import forward, load_net
        from lem.probe.sample import Sample
        sample = Sample(a.sample)
        with np.load(a.labels) as z:
            concepts.list_of(str(z["identity"]))                     # a known list (v1 or v2), or refuse
            labels, names = z["labels"], [str(n) for n in z["names"]]
        assert len(labels) == len(sample), "one label row per sample row"
        net = load_net(a.net)
        assert net.layout == sample.layout, f"the net is {net.layout}, the sample {sample.layout}"
        log(f"forward over {len(sample):,} rows")
        acts = forward(net, sample.x)
        result = audit.run(sample, acts, labels, names, log, a.shuffles)
        result["net"] = a.net
        result["concepts"] = str(np.load(a.labels)["identity"])
        audit.write(result, a.out_dir)
        concepts.write_list(Path(a.out_dir) / "concepts.json")
        log(json.dumps(result["shares"]))
        log(json.dumps(result["shuffled"]))
    elif a.cmd == "cards":
        from lem.probe import readers
        from lem.probe.cards import card
        from lem.probe.sample import Sample
        sample = Sample(a.sample)
        result = json.loads((Path(a.audit_dir) / "audit.json").read_text())
        rng = random.Random(readers.SEED)
        poles = []
        for row in result["rows"]:
            if row["kind"] == "dead":
                continue
            top = row["top"]
            explain = top[:readers.EXPLAIN]
            score_top = top[readers.EXPLAIN:readers.EXPLAIN + readers.SCORE]
            score_random = rng.sample(range(len(sample)), readers.SCORE)
            poles.append({"name": row["name"], "pole": row["pole"], "label": row["label"],
                          "explain": [card(sample, r) for r in explain],
                          "score_top": [card(sample, r) for r in score_top],
                          "score_random": [card(sample, r) for r in score_random]})
        Path(a.out).write_text(json.dumps(poles, indent=1))
        log(f"{a.out}: cards for {len(poles)} live poles")
    elif a.cmd == "readers":
        from lem.probe import readers
        results = readers.run(a.cards, a.out_dir, a.workers, a.model, log)
        summary = readers.summarize(results)
        (Path(a.out_dir) / "summary.json").write_text(json.dumps(summary, indent=1))
        log(json.dumps(summary))
    elif a.cmd == "ablate":
        from lem.probe import ablate
        from lem.probe.net import forward, load_net
        from lem.probe.sample import Sample
        sample = Sample(a.sample)
        net = load_net(a.net)
        acts = forward(net, sample.x)
        result = ablate.run(sample, acts, net, log=log)
        ablate.write(result, a.out)
        log(f"knee at k={result['knee']} (eval fidelity >= {ablate.KNEE})")
    elif a.cmd == "belief":
        from lem.probe import belief
        from lem.probe.net import forward, load_net
        from lem.probe.sample import Sample
        sample = Sample(a.sample)
        net = load_net(a.net)
        acts = forward(net, sample.x)
        result = belief.run(sample, acts, a.fit, a.eval, log, a.parts_dir)
        result["net"] = a.net
        belief.write(result, a.out_dir)
        log(json.dumps({k: {n: round(v["r2_moved"], 3) for n, v in result["probes"].items()} for k in ("r2_moved",)}))
    elif a.cmd == "belief-targets":
        from lem.probe import belief
        from lem.probe.sample import Sample
        belief.targets_part(Sample(a.sample), a.out_dir, a.part, a.parts, a.fit, a.eval, log)
    elif a.cmd == "fidelity-collect":
        from lem.probe import fidelity
        from lem.probe.sample import Sample
        d = fidelity.collect(Sample(a.sample), a.pour, a.k, log)
        np.savez_compressed(a.out, **d)
        log(f"{a.out}: {len(d['rows']):,} rows")
    elif a.cmd == "fidelity":
        from lem.probe import concepts, fidelity
        from lem.probe.net import forward, load_net
        from lem.probe.sample import Sample
        sample = Sample(a.sample)
        with np.load(a.labels) as z:
            concepts.list_of(str(z["identity"]))                     # a known list (v1 or v2), or refuse
            labels, names = z["labels"], [str(n) for n in z["names"]]
        net = load_net(a.net)
        acts = forward(net, sample.x)
        result = fidelity.run(sample, acts, labels, names, a.q_dir, log)
        result["net"] = a.net
        fidelity.write(result, a.out_dir)
        log(json.dumps({"overall": result["overall"], "high": result["ranked_high"][:5], "low": result["ranked_low"][:5]}))
    elif a.cmd == "aa":
        from lem.probe import aa, concepts
        from lem.probe.net import forward, load_net
        from lem.probe.sample import Sample
        sample = Sample(a.sample)
        with np.load(a.labels) as z:
            concepts.list_of(str(z["identity"]))                     # a known list (v1 or v2), or refuse
            labels, names = z["labels"], [str(n) for n in z["names"]]
        audit = json.loads((Path(a.audit_dir) / "audit.json").read_text())
        net = load_net(a.net)
        acts = forward(net, sample.x)
        log("audit halves")
        result = {"net": a.net, "audit": aa.audit_aa(sample, acts, labels, names, audit, log)}
        log(json.dumps({k: v for k, v in result["audit"].items() if k in ("wilson", "bootstrap", "clean_share")}))
        if a.belief_parts != "-":                                   # "-": the audit half only
            log("belief, both ways")
            result["belief"] = aa.belief_aa(sample, acts, a.belief_parts, log=log)
        aa.write(result, a.out_dir)
    elif a.cmd == "squeeze-train":
        from lem.probe import squeeze
        widths = [tuple(int(v) for v in w.split("x")) for w in a.widths.split(",")]
        squeeze.train_family(a.pours, a.out, widths, [int(v) for v in a.seeds.split(",")], a.epochs, a.device, log)
    elif a.cmd == "formation-train":
        from lem.probe import squeeze
        widths = [tuple(int(v) for v in w.split("x")) for w in a.widths.split(",")]
        squeeze.train_formation(a.pours, a.out, widths, [int(v) for v in a.seeds.split(",")], a.epochs, a.device, log)
    elif a.cmd == "formation-report":
        from lem.probe import formation
        Path(a.out).write_text(json.dumps(formation.report(a.audit_root, a.runs_root, a.sample, a.labels, a.q_dir, log), indent=1))
    elif a.cmd == "squeeze-families":
        from lem.probe import concepts, squeeze
        from lem.probe.net import forward, load_net
        from lem.probe.sample import Sample
        sample = Sample(a.sample)
        with np.load(a.labels) as z:
            concepts.list_of(str(z["identity"]))                     # a known list (v1 or v2), or refuse
            labels, names = z["labels"], [str(n) for n in z["names"]]
        audit = json.loads((Path(a.audit_dir) / "audit.json").read_text())
        acts = forward(load_net(a.net), sample.x)
        out = squeeze.families(labels, names, acts, audit, log)
        out["net"] = a.net
        Path(a.out).write_text(json.dumps(out, indent=1))
    elif a.cmd == "list":
        from lem.probe import concepts
        concepts.write_list(a.out)
        log(f"{a.out}: {len(concepts.CONCEPTS)} concepts, identity {concepts.identity()}")


if __name__ == "__main__":
    main()
