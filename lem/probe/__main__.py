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
    s.add_argument("--condition", choices=("n_plays",), default=None, help="centre each unit within each play count, pooled scale (#226, #233)")
    s.add_argument("--baseline", choices=("n_plays",), default=None, help="labels must beat what the play count alone predicts (#233)")
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
    s = sub.add_parser("tree"); s.add_argument("sample"); s.add_argument("narrow_net"); s.add_argument("narrow_audit")
    s.add_argument("wide_net"); s.add_argument("wide_audit"); s.add_argument("out")
    s = sub.add_parser("squeeze-families"); s.add_argument("sample"); s.add_argument("labels"); s.add_argument("audit_dir"); s.add_argument("out")
    s.add_argument("--net", default=NET)
    s = sub.add_parser("families-u"); s.add_argument("sample"); s.add_argument("labels"); s.add_argument("audit_dir"); s.add_argument("out")
    s.add_argument("--net", default=NET); s.add_argument("--condition", choices=("n_plays",), default=None)
    s.add_argument("--shuffle-rows", type=int, default=None, help="calibration: permute the label rows under this seed")
    s = sub.add_parser("coarse"); s.add_argument("sample"); s.add_argument("labels_v2"); s.add_argument("audit_v1_dir"); s.add_argument("donor_audit_v2_dir"); s.add_argument("out")
    s.add_argument("--net", default=NET); s.add_argument("--folds", action="store_true", help="also read planted folds (the positive controls)")
    s = sub.add_parser("origin"); s.add_argument("sample"); s.add_argument("labels"); s.add_argument("run_dir"); s.add_argument("out")
    s = sub.add_parser("candidates"); s.add_argument("sample"); s.add_argument("labels_v2"); s.add_argument("out")
    s.add_argument("--unit", action="append", required=True, help="NET:J:B"); s.add_argument("--recur", nargs="+", required=True)
    s = sub.add_parser("list"); s.add_argument("out")
    s = sub.add_parser("fidelity-collect"); s.add_argument("sample"); s.add_argument("pour"); s.add_argument("k", type=int); s.add_argument("out")
    s = sub.add_parser("fidelity"); s.add_argument("sample"); s.add_argument("labels"); s.add_argument("q_dir"); s.add_argument("out_dir")
    s.add_argument("--net", default=NET)
    s = sub.add_parser("causal"); s.add_argument("sample"); s.add_argument("labels"); s.add_argument("audit_dir"); s.add_argument("q_dir"); s.add_argument("out")
    s.add_argument("--net", default=NET); s.add_argument("--steer", action="store_true", help="add k·sigma doses instead of ablate/clamp")
    s.add_argument("--matched", action="store_true", help="#205: own unit vs random units, stratified by play count")
    s = sub.add_parser("unit"); s.add_argument("sample"); s.add_argument("labels"); s.add_argument("net"); s.add_argument("j", type=int); s.add_argument("out")
    s.add_argument("--twin", required=True); s.add_argument("--audit", required=True, help="the net's audit dir (reader labels)")
    s.add_argument("--formation", default=None, help="a formation run dir of step_*.npz checkpoints")
    s.add_argument("--gate", default=None, help="an input bit that gates the unit (e.g. x:hand:14): the gate paper")
    s.add_argument("--layer", type=int, default=1, help="2: a layer-2 unit that acts on lead (the reader paper)")
    s.add_argument("--trick", action="store_true", help="a layer-1 unit shut by trumps on the open trick (paper 5)")
    s.add_argument("--winner", action="store_true", help="a layer-1 unit read by who wins the open trick, and with what (paper 6)")
    s.add_argument("--lead", action="store_true", help="a layer-1 unit shut on leads from trump length (paper 7)")
    s.add_argument("--count", action="store_true", help="a layer-2 unit read as a dial on putting count on the trick (paper 8)")
    s.add_argument("--double", action="store_true", help="a layer-2 unit read as leading the top off double under fives and sixes (paper 9)")
    s = sub.add_parser("route"); s.add_argument("sample"); s.add_argument("net"); s.add_argument("bit"); s.add_argument("tile", type=int); s.add_argument("out")
    s = sub.add_parser("gus-sample"); s.add_argument("sample"); s.add_argument("labels"); s.add_argument("out_dir")
    s = sub.add_parser("gus-audit"); s.add_argument("rows_dir"); s.add_argument("ckpt"); s.add_argument("out_dir")
    s.add_argument("--untrained", type=int, default=None, help="an untrained twin under this torch seed (ckpt gives the config)")
    s.add_argument("--shuffles", type=int, default=20)
    s.add_argument("--condition", choices=("n_plays",), default=None, help="centre each unit within each play count, pooled scale (#226, #233)")
    s.add_argument("--baseline", choices=("n_plays",), default=None, help="labels must beat what the play count alone predicts (#233)")
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
        if a.condition:
            acts = audit.condition(acts, sample.n_plays.astype(np.int64))
        result = audit.run(sample, acts, labels, names, log, a.shuffles,
                           strata=sample.n_plays.astype(np.int64) if a.baseline else None)
        result["net"] = a.net
        result["condition"] = a.condition
        result["baseline"] = a.baseline
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
    elif a.cmd == "causal":
        from lem.probe import causal, fidelity
        from lem.probe.net import Activations, load_net
        from lem.probe.sample import Sample
        sample, net = Sample(a.sample), load_net(a.net)
        with np.load(a.labels) as z:
            labels, names = z["labels"], [str(n) for n in z["names"]]
        audit = json.loads((Path(a.audit_dir) / "audit.json").read_text())
        z0 = np.zeros((0, 0), np.int64)
        regret_of = lambda out: fidelity.regret(sample, Activations(z0, z0, z0, z0, out), a.q_dir)[0]
        result = (causal.matched_steer if a.matched else causal.steer if a.steer else causal.run)(net, sample, labels, names, audit, regret_of, log)
        Path(a.out).write_text(json.dumps(result, indent=1))
        log(f"wrote {a.out}: {len(result['poles'])} named poles")
    elif a.cmd == "unit":
        from lem.probe import unit
        from lem.probe.sample import Sample
        with np.load(a.labels) as z:
            labels, names = z["labels"], [str(n) for n in z["names"]]
        audit = json.loads((Path(a.audit) / "audit.json").read_text())
        if a.double:
            result = unit.run_double(Sample(a.sample), labels, names, a.net, a.j, a.twin, audit, a.formation, log)
        elif a.count:
            result = unit.run_count(Sample(a.sample), labels, names, a.net, a.j, a.twin, audit, a.formation, log)
        elif a.lead:
            result = unit.run_lead(Sample(a.sample), labels, names, a.net, a.j, a.twin, audit, a.formation, log)
        elif a.winner:
            result = unit.run_winner(Sample(a.sample), labels, names, a.net, a.j, a.twin, audit, a.formation, log)
        elif a.trick:
            result = unit.run_trick(Sample(a.sample), labels, names, a.net, a.j, a.twin, audit, a.formation, log)
        elif a.layer == 2:
            result = unit.run_reader(Sample(a.sample), labels, names, a.net, a.j, a.twin, audit, a.formation, log)
        elif a.gate:
            result = unit.run_gate(Sample(a.sample), labels, names, a.net, a.j, a.gate, a.twin, audit, a.formation, log)
        else:
            result = unit.run(Sample(a.sample), labels, names, a.net, a.j, a.twin, audit, a.formation, log)
        Path(a.out).write_text(json.dumps(result, indent=1))
        log(f"wrote {a.out}")
    elif a.cmd == "route":
        from lem.probe import unit
        from lem.probe.sample import Sample
        result = unit.route(Sample(a.sample), a.net, a.bit, a.tile, log)
        Path(a.out).write_text(json.dumps(result, indent=1))
        log(f"wrote {a.out}")
    elif a.cmd == "gus-sample":
        from lem.probe import gus
        gus.build(a.sample, a.labels, a.out_dir, log)
    elif a.cmd == "gus-audit":
        import torch
        from lem.probe import audit, gus
        from lem.probe.sample import Sample
        torch.set_num_threads(max(1, int(__import__("os").environ.get("OMP_NUM_THREADS", "1"))))
        d = Path(a.rows_dir)
        sample = Sample(d / "sample.npz")
        with np.load(d / "labels.npz") as z:
            labels, names = z["labels"], [str(n) for n in z["names"]]
            identity = str(z["identity"])
        with np.load(d / "tokens.npz") as z:
            tokens, lengths = z["tokens"], z["lengths"]
        cfg = torch.load(a.ckpt, map_location="cpu", weights_only=True)["config"] if a.untrained is not None else None
        model, ident = gus.load_model(None if a.untrained is not None else a.ckpt, a.untrained, cfg)
        log(f"forward over {len(tokens):,} rows: {ident}")
        acts = gus.forward(model, tokens, lengths)
        raw_out = acts.out
        if a.condition:
            acts = audit.condition(acts, sample.n_plays.astype(np.int64))
        hit = float((raw_out.argmax(axis=1) == sample.chosen.astype(np.int64)).mean())
        log(f"Gus's top tile matches the recorded move on {hit:.4f} of rows")
        result = audit.run(sample, acts, labels, names, log, shuffles=a.shuffles,
                           strata=sample.n_plays.astype(np.int64) if a.baseline else None)
        result["baseline"] = a.baseline
        result["concepts"] = identity
        result["net"] = ident
        result["top1_agrees_with_recorded"] = hit
        result["condition"] = a.condition
        audit.write(result, a.out_dir)
        log(json.dumps(result["shares"]))
    elif a.cmd == "families-u":
        from lem.probe import audit, concepts, squeeze
        from lem.probe.net import forward, load_net
        from lem.probe.sample import Sample
        sample = Sample(a.sample)
        with np.load(a.labels) as z:
            concepts.list_of(str(z["identity"]))
            labels, names = z["labels"], [str(n) for n in z["names"]]
        acts = forward(load_net(a.net), sample.x)
        if a.condition:
            acts = audit.condition(acts, sample.n_plays.astype(np.int64))
        if a.shuffle_rows is not None:
            labels = labels[np.random.default_rng(a.shuffle_rows).permutation(len(labels))]
        res = squeeze.families_u(labels, names, acts, json.loads((Path(a.audit_dir) / "audit.json").read_text()), log=log)
        res.update(net=a.net, condition=a.condition, shuffle_rows=a.shuffle_rows)
        Path(a.out).write_text(json.dumps(res, indent=1))
    elif a.cmd == "coarse":
        from lem.probe import coarse, concepts
        from lem.probe.net import load_net
        from lem.probe.sample import Sample
        sample, net = Sample(a.sample), load_net(a.net)
        with np.load(a.labels_v2) as z:
            assert concepts.list_of(str(z["identity"])) is concepts.LISTS["v2"], "coarse reads list v2 (v1 is its first columns)"
            L2, n2 = z["labels"], [str(n) for n in z["names"]]
        n1 = len(concepts.LISTS["v1"])
        res = {"net": a.net,
               "broadened": coarse.broadened(sample, L2[:, :n1], n2[:n1], net, json.loads((Path(a.audit_v1_dir) / "audit.json").read_text()), log)}
        if a.folds:
            res["folds"] = coarse.folds(sample, L2, n2, net, json.loads((Path(a.donor_audit_v2_dir) / "audit.json").read_text()), log)
        Path(a.out).write_text(json.dumps(res, indent=1))
        log(f"wrote {a.out}: {len(res['broadened'])} broadened, {len(res.get('folds', []))} folds")
    elif a.cmd == "origin":
        from lem.probe import concepts, origin
        from lem.probe.sample import Sample
        with np.load(a.labels) as z:
            concepts.list_of(str(z["identity"]))
            labels, names = z["labels"], [str(n) for n in z["names"]]
        res = origin.run(Sample(a.sample), labels, names, a.run_dir, log)
        Path(a.out).write_text(json.dumps(res, indent=1))
        log(f"wrote {a.out}: {len(res['broadened'])} broadened, {len(res['clean'])} clean")
    elif a.cmd == "candidates":
        from lem.probe import candidates as cd, concepts
        from lem.probe.net import load_net
        from lem.probe.sample import Sample
        sample = Sample(a.sample)
        with np.load(a.labels_v2) as z:
            assert concepts.list_of(str(z["identity"])) is concepts.LISTS["v2"]
            L, names = z["labels"], [str(n) for n in z["names"]]
        n1 = len(concepts.LISTS["v1"])
        trick, other = 0, 0
        for path in a.recur:
            for u in cd.broadened_unions(sample, L[:, :n1], names[:n1], load_net(path)):
                if "lead" in u:
                    p = u[1] if u[0] == "lead" else u[0]
                    trick += p in cd.TRICK_STATE
                    other += p not in cd.TRICK_STATE
            log(f"  recurrence after {path}: lead+trick-state {trick}, lead+other {other}")
        recurs = trick > other
        res = {"recurrence": {"lead_trick_state": trick, "lead_other_family": other, "nets": a.recur, "recurs_above_baseline": recurs}, "units": []}
        for spec in a.unit:
            path, j, b = spec.rsplit(":", 2)
            rec = cd.read_unit(sample, L, names, load_net(path), int(j), b, log)
            rec["net"] = path
            rec["verdict"] = cd.verdict(rec, recurs)
            res["units"].append(rec)
            log(f"  verdict {path} unit {j}: {rec['verdict']}")
        Path(a.out).write_text(json.dumps(res, indent=1, default=str))
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
    elif a.cmd == "tree":
        from lem.probe import tree
        from lem.probe.net import forward, load_net
        from lem.probe.sample import Sample
        sample = Sample(a.sample)
        res = tree.match(forward(load_net(a.narrow_net), sample.x), json.loads((Path(a.narrow_audit) / "audit.json").read_text()),
                         forward(load_net(a.wide_net), sample.x), json.loads((Path(a.wide_audit) / "audit.json").read_text()), sample.deal, log)
        res.update(narrow=a.narrow_net, wide=a.wide_net)
        Path(a.out).write_text(json.dumps(res, indent=1))
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
