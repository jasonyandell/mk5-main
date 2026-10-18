# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""Gus depth campaign. Heavy units run in fresh bounded cpu-slot processes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch

from scenario.rules import mask_of
from .data import prepare
from .population import (active_pool, atomic_json, atomic_save, deepen, digest,
                         make_specs, seed_for)

DEPTHS = [2, 4, 8, 12, 15]


def plateau(score, best, stale, minimum_gain=.02, patience=4):
    if not np.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("invalid development score")
    improved = score >= best + minimum_gain
    return {"best": score if improved else best, "stale": 0 if improved else stale+1,
            "improved": bool(improved), "plateau": bool(not improved and stale+1 >= patience)}


def identity(path):
    return {"path": str(Path(path).resolve()), "sha256": digest(path)}


def initialize(root, initial, source, field):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "campaign.json").exists():
        raise ValueError("campaign already initialized")
    torch.set_num_threads(2)
    saved = torch.load(initial, map_location="cpu", weights_only=True)
    if saved["config"] != {"width":64,"layers":2,"heads":2,"context":68,"vocabulary":46}:
        raise ValueError("campaign starts at the published G0 architecture")
    records = json.loads(Path(source).read_text())
    excluded = {tuple(r["hands"]) for r in records}
    shutil.copyfile(initial, root / "base.pt")
    base = identity(root / "base.pt")
    config = {"initial": base, "source_deals": identity(source), "zeb_field": str(Path(field).resolve()),
              "zeb_field_sha256": digest(field), "depths": DEPTHS,
              "zeb_spec": f"k0,n160,f=16,F={Path(field).resolve()},u=make,d=posterior,r=uniform:4194304:0.28,t=margin,s0",
              "blocks":8,"hands_per_block":512,"lr":3e-5,"panel_hands":256,"confirmation_hands":512,
              "plateau_minimum_gain":.02,"plateau_patience":4,
              "pretraining_data_sha256":saved["data_sha256"],
              "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
              "wandb_entity":"jasonyandell-forge42","wandb_project":"gus-population"}
    for label,count in [("development",256),("confirmation",512)]:
        specs = make_specs(count,f"gus-depth-20261018-{label}",["g000"],excluded)
        excluded.update(tuple(mask_of(h) for h in s["hands"]) for s in specs)
        atomic_json(root / f"{label}-specs.json", specs)
    np.save(root / "excluded.npy", np.asarray(sorted(excluded),dtype=np.uint32))
    validation = [r for r in records if r["split"] == "validation"]
    ids = np.random.default_rng(42).choice(len(validation),512,replace=False)
    probe = [validation[i] for i in ids]
    atomic_json(root / "prediction-deals.json",probe)
    np.savez_compressed(root / "prediction.npz", **prepare(probe))
    (root / "evaluation").mkdir()
    atomic_json(root / "evaluation/config.json", {**config,"pool":{"g000":base}})
    atomic_json(root / "campaign.json",config)
    atomic_json(root / "state.json",{"status":"ready","generation":0,"depth_index":0,
                                      "generations":[{"id":"g000",**base,"layers":2}],
                                      "stale":0,"best_score":None,"best_path":base["path"]})
    print(json.dumps({"initialized":str(root),"base":base,"config":config}),flush=True)


def prepare_generation(root, number, parent, layers):
    root = Path(root).resolve()
    campaign = json.loads((root / "campaign.json").read_text())
    state = json.loads((root / "state.json").read_text())
    name = f"g{number:03d}"
    run = root / name
    run.mkdir(exist_ok=True)
    if (run / "ready.json").exists():
        ready=json.loads((run / "ready.json").read_text())
        if ready["parent_sha256"] != digest(parent) or ready["layers"] != layers:
            raise ValueError("generation preparation changed")
        return
    roster = active_pool([g["id"] for g in state["generations"]])
    pool = {g["id"]:{k:g[k] for k in ("path","sha256")} for g in state["generations"] if g["id"] in roster}
    excluded = {tuple(map(int,h)) for h in np.load(root / "excluded.npy")}
    scheduled = []
    for label in ("development", "confirmation"):
        scheduled += json.loads((root/f"{label}-specs.json").read_text())
    for path in root.glob("g[0-9][0-9][0-9]/specs.json"):
        if path.parent != run:
            earlier = json.loads(path.read_text())
            scheduled += earlier
            excluded.update(tuple(mask_of(h) for h in s["hands"]) for s in earlier)
    specs = make_specs(campaign["blocks"]*campaign["hands_per_block"],f"gus-depth-20261018-{name}",roster,excluded)
    saved = torch.load(parent,map_location="cpu",weights_only=True)
    if layers > saved["config"]["layers"]:
        saved = deepen(saved,layers,seed_for(name,"growth"))
    elif layers != saved["config"]["layers"]:
        raise ValueError("cannot shrink this campaign")
    initial = {k:saved[k] for k in ("model","config","tokenizer","data_sha256")}
    initial.update(step=0,rl_updates=0,parent_sha256=digest(parent))
    if "growth" in saved:
        initial["growth"] = saved["growth"]
    atomic_save(run / "initial.pt",initial)
    atomic_json(run / "specs.json",specs)
    atomic_json(run / "scheduled-deals.json",[{"hands":[mask_of(h) for h in s["hands"]]} for s in scheduled+specs])
    atomic_json(run / "config.json",{**campaign,"generation":name,"layers":layers,"parent":identity(parent),
                                      "pool":pool,"initial":identity(run / "initial.pt"),
                                      "specs_sha256":digest(run / "specs.json"),"training_overlap":0})
    atomic_json(run / "ready.json",{"parent_sha256":digest(parent),"layers":layers})


def unit(root, label, arguments, timeout=300):
    """One live subprocess is the authority; terminal failure preserves logs."""
    root = Path(root)
    log = root / f"{label}.log"
    command = ["scripts/cpu-slot","--threads","2","--timeout",str(timeout),"--",
               sys.executable,"-u",*map(str,arguments)]
    print(f"START {label}",flush=True)
    with log.open("a") as stream:
        result = subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{label} failed ({result.returncode}); see {log}")
    print(f"DONE {label}",flush=True)


def panel(root, checkpoint, label, specs="development", reference=None):
    root = Path(root)
    out = root / label
    receipt = out / "report.json"
    if receipt.exists():
        report=json.loads(receipt.read_text())
        if report["checkpoint_sha256"] != digest(checkpoint):
            raise ValueError("panel checkpoint changed")
        return report
    args=["-m","lem.observer.reward","panel","--run",root/"evaluation",
          "--checkpoint",checkpoint,"--specs",root/f"{specs}-specs.json","--out",out,
          "--prediction",root/"prediction.npz","--purpose",specs]
    if reference:
        args += ["--reference",reference]
    unit(root,label.replace("/","-"),args)
    return json.loads(receipt.read_text())


def run_campaign(root):
    import wandb
    from .release import publish_generation
    root = Path(root).resolve()
    config = json.loads((root / "campaign.json").read_text())
    state = json.loads((root / "state.json").read_text())
    if state["status"] == "complete":
        print("Campaign already complete",flush=True)
        return
    base = config["initial"]["path"]
    baseline = panel(root,base,"baseline")
    if state["best_score"] is None:
        state.update(best_score=baseline["team_win_rate"],status="training")
        atomic_json(root/"state.json",state)
    while state["status"] != "confirming":
        number = state["generation"]+1
        name = f"g{number:03d}"
        run = root/name
        depth = DEPTHS[state["depth_index"]]
        parent = state.get("growth_parent") or state["generations"][-1]["path"]
        unit(root,f"prepare-{name}",["-m","lem.observer.campaign","prepare","--root",root,
                                       "--generation",number,"--parent",parent,"--layers",depth])
        if state.get("growth_parent"):
            state["best_path"] = str(run/"initial.pt")
        recipe = json.loads((run/"config.json").read_text())
        tracking = run/"wandb-run.json"
        track_id = json.loads(tracking.read_text())["id"] if tracking.exists() else wandb.util.generate_id()
        with wandb.init(project=config["wandb_project"],entity=config["wandb_entity"],
                        name=f"gus-{name}-depth{depth}",id=track_id,resume="allow",dir=str(run),
                        group="depth-ladder-20261018",job_type="population-ppo",
                        config=recipe,tags=["gus","population","formation",f"depth_{depth}"]) as tracker:
            atomic_json(tracking,{"id":tracker.id,"url":tracker.url})
            tracker.define_metric("hands_seen")
            tracker.define_metric("*",step_metric="hands_seen")
            print(f"LIVE_RUN {tracker.url}",flush=True)
            for block in range(config["blocks"]):
                unit(root,f"{name}-block-{block:02d}",["-m","lem.observer.reward","block","--run",run,"--index",block])
                report=json.loads((run/f"block-{block:02d}/complete.json").read_text())
                values={"hands_seen":(block+1)*config["hands_per_block"],"block":block+1,
                        "train_team_win_rate":report["outcome"]["team_win_rate"],
                        "collection_seconds":report["collection_seconds"],"optimizer_seconds":report["optimizer_seconds"],
                        **{f"ppo_{k}":v for k,v in report["optimizer"]["passes"][-1].items()}}
                for key,stats in report["outcome"]["groups"].items():
                    values[f"train_{key}_win_rate"]=stats["win_rate"]
                tracker.log(values)
            checkpoint=run/f"block-{config['blocks']-1:02d}/checkpoint.pt"
            result=panel(root,checkpoint,f"{name}/development",reference=root/"baseline/report.json")
            decision=plateau(result["team_win_rate"],state.get("progress_score",state["best_score"]),state["stale"],
                             config["plateau_minimum_gain"],config["plateau_patience"])
            tracker.log({"hands_seen":4096,"development_win_rate":result["team_win_rate"],
                         "development_vs_g0":result["vs_reference"]["win_rate_difference"],
                         "prediction_all_accuracy":result["prediction"]["all"]["top1"],
                         "prediction_all_nll":result["prediction"]["all"]["nll"],
                         "plateau_stale_generations":decision["stale"]})
            for label,opponent in [("parent",parent),("zeb","walt:"+config["zeb_spec"])]:
                smoke=run/f"smoke-{label}"
                if not (smoke/"report.json").exists():
                    if smoke.exists():
                        raise RuntimeError(f"incomplete smoke retained at {smoke}; inspect before retry")
                    unit(root,f"{name}-smoke-{label}",["-m","lem.observer.h2h","--candidate",checkpoint,
                          "--opponent",opponent,"--games",2,"--seed",seed_for(name,"smoke",label),
                          "--exclude-deals",config["source_deals"]["path"],
                          "--exclude-deals",run/"scheduled-deals.json","--out",smoke])
            atomic_json(run/"result.json",{"development":result,"decision":decision,"checkpoint":identity(checkpoint),
                                          "parent":recipe["parent"],"wandb":tracker.url})
            release=publish_generation(root,run,checkpoint)
            tracker.summary.update({"status_note":"complete; development-selected exploratory candidate",
                                    "hf_revision":release["model_revision"],"development_win_rate":result["team_win_rate"]})
        state["generations"].append({"id":name,**identity(checkpoint),"layers":depth,
                                     "development_win_rate":result["team_win_rate"]})
        improved_any = result["team_win_rate"] > state["best_score"]
        state.update(generation=number,best_score=max(state["best_score"],result["team_win_rate"]),
                     progress_score=decision["best"],stale=decision["stale"])
        state.pop("growth_parent",None)
        if improved_any:
            state["best_path"]=str(checkpoint)
        if decision["plateau"]:
            if state["depth_index"] == len(DEPTHS)-1:
                state["status"]="confirming"
                atomic_json(root/"state.json",state)
                break
            state.update(depth_index=state["depth_index"]+1,stale=0,growth_parent=state["best_path"],
                         progress_score=state["best_score"])
        atomic_json(root/"state.json",state)
        print(json.dumps({"generation":name,"layers":depth,"win_rate":result["team_win_rate"],"decision":decision}),flush=True)
    # Fixed fresh panel is used once, after all depth/weight selection.
    base_final=panel(root,base,"confirmation-g0",specs="confirmation")
    final=panel(root,state["best_path"],"confirmation-selected",specs="confirmation",
                reference=root/"confirmation-g0/report.json")
    atomic_json(root/"final.json",{"selected":identity(state["best_path"]),"baseline":base_final,
                                  "confirmation":final,"state":state})
    state["status"]="complete"
    atomic_json(root/"state.json",state)
    print("CAMPAIGN_COMPLETE",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=("init","prepare","run"))
    p.add_argument("--root",required=True)
    p.add_argument("--initial");p.add_argument("--source");p.add_argument("--field")
    p.add_argument("--generation",type=int);p.add_argument("--parent");p.add_argument("--layers",type=int)
    a=p.parse_args()
    if a.command == "init": initialize(a.root,a.initial,a.source,a.field)
    elif a.command == "prepare": prepare_generation(a.root,a.generation,a.parent,a.layers)
    else: run_campaign(a.root)


if __name__ == "__main__":
    main()
