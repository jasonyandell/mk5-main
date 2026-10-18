# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""Publish versioned Gus candidates and allowlisted, reproducible evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import torch

from .data import label
from .population import atomic_json, digest, load_model

MODEL_REPO = "jasonyandell/gus"
DATA_REPO = "jasonyandell/texas-42-zeb-research-2026-10"
ARCHIVE = "lem/gus-depth-campaign-2026-10-18"


def publish_generation(root, run, checkpoint):
    from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download, ModelCard
    from safetensors.torch import save_file, load_file
    root,run=Path(root),Path(run)
    receipt=run/"published.json"
    if receipt.exists():
        return json.loads(receipt.read_text())
    config=json.loads((run/"config.json").read_text())
    result=json.loads((run/"result.json").read_text())
    name=config["generation"]
    stage=run/"release";stage.mkdir(exist_ok=True)
    model,saved=load_model(checkpoint,"cpu")
    state={k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()}
    save_file(state,str(stage/"model.safetensors"),metadata={"format":"pt","name":"Gus","generation":name})
    inference={k:saved[k] for k in ("config","model","step","tokenizer","data_sha256")}
    torch.save(inference,stage/"gus.pt")
    atomic_json(stage/"config.json",saved["config"])
    atomic_json(stage/"tokenizer.json",{"encoding":saved["tokenizer"],"input_tokens":[{"id":i,"label":label(i)} for i in range(46)],
                "tile_outputs":28,"context":68,"prefix":"BOS BID30 relative_bidder DECL HAND own_7_sorted_tiles PLAY",
                "history":"relative_actor tile pairs; end at next actor","own_actor_id":28,
                "unsupported_id":40,"note":"Game-token schema, not a text tokenizer."})
    atomic_json(stage/"provenance.json",{"generation":name,"source_checkpoint_sha256":digest(checkpoint),
                "training":config,"result":result,"parameters":sum(p.numel() for p in model.parameters()),
                "archive":f"https://huggingface.co/datasets/{DATA_REPO}/tree/main/{ARCHIVE}/{name}"})
    atomic_json(stage/"evaluation.json",{"development":result["development"],
                "smokes":{s:json.loads((run/f"smoke-{s}/report.json").read_text()) for s in ("parent","zeb")}})
    example={"tokens":[42,45,28,38,43,0,2,5,9,14,20,27,44,28],"legal_tile_ids":[0,2,5,9,14,20,27]}
    torch.set_num_threads(2)
    with torch.inference_mode():
        logits=model(torch.tensor([example["tokens"]]))[0,-1]
    example.update(expected_last_logits=logits.tolist(),expected_argmax_legal_tile=max(example["legal_tile_ids"],key=lambda t:float(logits[t])))
    atomic_json(stage/"example.json",example)
    atomic_json(stage/"export-verification.json",{"source_checkpoint_sha256":digest(checkpoint),
                "parameter_count":sum(p.numel() for p in model.parameters()),
                "all_exported_tensors_match_source":all(torch.equal(state[k],saved["model"][k]) for k in state)})
    (stage/"LICENSE").write_text(Path("LICENSES/CC0-1.0.txt").read_text())
    params=sum(p.numel() for p in model.parameters())
    card=f'''---
license: cc0-1.0
library_name: pytorch
model_name: Gus
tags: [texas-42, dominoes, nanogpt, causal-transformer, reinforcement-learning, interpretability, custom-code]
---
# Gus — population candidate {name}

**{params:,} parameters, {saved['config']['layers']} blocks, width 64, two attention heads.**
This is an exploratory reward-trained candidate in the Gus depth campaign.
Latest does not mean strongest. The initial pretrained model remains frozen at
[`pretrained-v0`](https://huggingface.co/{MODEL_REPO}/tree/pretrained-v0).

## Inputs and outputs

Unchanged original own seven tiles, public bid-30 contract/bidder/declaration,
relative seats and public actor/tile history. No hidden hands, player identities,
planner values, legal-action list or match score are inputs. There are 46 input
tokens, 68 context positions and 28 tile logits. Tile outputs are masked by the
acting seat's lawful legal moves when playing. No bidding, Nello, value or splat
head is present. Game-token definitions are in `tokenizer.json`.

## Learning and scope

Starts from {config['parent']['sha256']}. This generation collected 4,096 fresh
hands, eight batches of 512. One learner seat samples legal softmax; each other
seat is independently 25% pinned Zeb and 75% uniform among the recent-five Gus
pool. Frozen Gus seats use legal argmax. PPO uses only the learner's own actions,
actual team make/set return, a prior-batch scalar baseline, and no hidden critic.
Bid 30 and pip declarations throughout. Exact settings and model/field pins are
in `provenance.json`; full episodes and optimizer checkpoints are archived.

The development score is {100*result['development']['team_win_rate']:.2f}% on a
fixed 256-hand G0/Zeb mixed table. This panel is reused for adaptive selection;
it is exploratory and is not a general win rate or independent confirmation.
Two full-game smokes versus the parent and two versus Zeb check integration only.
Native-neuron meanings remain ungraded. The depth stopping rule is four
generations without a two-percentage-point development improvement.

[Live/training run]({result['wandb']}).
[Protocol](https://github.com/jasonyandell/mk5-main/blob/main/wiki/decisions/gus-population-iteration.md).

## Load

Use the repository's `lem.observer.model.Observer` and `Config` classes with
`config.json` and `model.safetensors`, or pass downloaded `gus.pt` to
`lem.observer.play.ObserverPlay`. The code and loading pattern are the same as
the [pretrained release](https://huggingface.co/{MODEL_REPO}/blob/pretrained-v0/README.md);
set the Hub revision to **`gus-{name}`** and read this release's example's
`expected_argmax_legal_tile` rather than assuming six-six. This is custom PyTorch
code, not a Transformers AutoModel or prose model. `gus.pt` is inference-only,
compatible with `torch.load(..., weights_only=True)`; optimizer states are in
the [research archive](https://huggingface.co/datasets/{DATA_REPO}/tree/main/{ARCHIVE}/{name}).

Weights/artifacts: CC0-1.0, Jason Yandell. The nanoGPT-derived implementation is
MIT, Andrej Karpathy and Jason Yandell; retain its license. `SHA256SUMS` covers
release files. The historical April oracle-era Gus is a separate model lineage.
'''
    (stage/"README.md").write_text(card)
    ModelCard.load(stage/"README.md").validate()
    for p in stage.iterdir():
        if p.suffix in (".md",".json"):
            if re.search(r"hf_[A-Za-z0-9]{20,}|BEGIN .*PRIVATE KEY",p.read_text()):
                raise ValueError("credential-shaped release content")
    (stage/"SHA256SUMS").write_text("".join(f"{digest(p)}  {p.name}\n" for p in sorted(stage.iterdir()) if p.name!="SHA256SUMS"))
    api=HfApi()
    phases=run/"publication-progress.json"
    progress=json.loads(phases.read_text()) if phases.exists() else {}
    if "model_revision" not in progress:
        commit=api.create_commit(repo_id=MODEL_REPO,repo_type="model",commit_message=f"Gus {name}: depth {saved['config']['layers']} population candidate",
                operations=[CommitOperationAdd(path_in_repo=p.name,path_or_fileobj=str(p)) for p in sorted(stage.iterdir())])
        progress["model_revision"]=commit.oid;atomic_json(phases,progress)
    api.create_tag(MODEL_REPO,tag=f"gus-{name}",revision=progress["model_revision"],exist_ok=True)
    if api.model_info(MODEL_REPO,revision=f"gus-{name}").sha != progress["model_revision"]:
        raise ValueError("generation tag was already used for a different release")
    if "archive_revision" not in progress:
        files=[run/p for p in ["config.json","specs.json","scheduled-deals.json","initial.pt","ready.json","result.json","wandb-run.json"]]
        for directory in sorted(run.glob("block-*")):
            files += [directory/p for p in ["checkpoint.pt","decisions.npz","hands.jsonl.gz","complete.json"]]
        files += [run/"development"/p for p in ["report.json","hands.jsonl.gz"]]
        for opponent in ("parent","zeb"):
            files += [run/f"smoke-{opponent}"/p for p in ["manifest.json","report.json","games.jsonl.gz"]]
        total=sum(p.stat().st_size for p in files)
        if total > 2_000_000_000:
            raise ValueError("generation archive exceeds the 2GB per-upload review limit")
        common=[root/p for p in ["campaign.json","development-specs.json","confirmation-specs.json","prediction-deals.json","prediction.npz"]]
        common += [root / "baseline" / p for p in ["report.json", "hands.jsonl.gz"]]
        common += list(root.glob(f"{name}-*.log")) + list(root.glob(f"{name}-*-execution.json"))
        common += list(root.glob(f"prepare-{name}*"))
        operations=[CommitOperationAdd(path_in_repo=f"{ARCHIVE}/{name}/{p.relative_to(run)}",path_or_fileobj=str(p)) for p in files]
        operations += [CommitOperationAdd(path_in_repo=f"{ARCHIVE}/{p.relative_to(root)}",path_or_fileobj=str(p)) for p in common]
        commit=api.create_commit(repo_id=DATA_REPO,repo_type="dataset",commit_message=f"Gus {name}: complete formation and lawful population trajectories",
                                operations=operations)
        progress.update(archive_revision=commit.oid,archive_bytes=total);atomic_json(phases,progress)
    downloaded=Path(hf_hub_download(MODEL_REPO,"model.safetensors",revision=progress["model_revision"],token=False,local_dir=run/"verify-model"))
    if digest(downloaded) != digest(stage/"model.safetensors"):
        raise ValueError("published model hash mismatch")
    restored=load_file(str(downloaded))
    if any(not torch.equal(state[k],restored[k]) for k in state):
        raise ValueError("published tensors differ")
    sample=run/"block-00/hands.jsonl.gz"
    remote=Path(hf_hub_download(DATA_REPO,f"{ARCHIVE}/{name}/block-00/hands.jsonl.gz",repo_type="dataset",
                    revision=progress["archive_revision"],token=False,local_dir=run/"verify-evidence"))
    if digest(remote)!=digest(sample):
        raise ValueError("published replay hash mismatch")
    result={**progress,"model_url":f"https://huggingface.co/{MODEL_REPO}/tree/gus-{name}",
            "anonymous_model_and_replay_verified":True,"weights_sha256":digest(downloaded)}
    atomic_json(receipt,result)
    return result
